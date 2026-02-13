import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers.activations import ACT2FN
from typing import Optional, Union, Tuple, List

# Try to import flash-attn, fallback to standard attention if not available
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None

class LVRHead(nn.Module):
    """
        The simplest mlp w/o up_proj
    """
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.ln_q = LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x))
        return x


class LVRHeadGLU(nn.Module):
    ''' 
        The Gated Liner Unit MLP
    '''
    def __init__(self, hidden_size, intermediate_size, hidden_act, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        # 11008 for 3b; 18944 for 7b; 27648 for 32b
        self.intermediate_size = intermediate_size  
        self.hidden_act = hidden_act
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[self.hidden_act]    #silu

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
    

class LVRHeadAttention(nn.Module):
    """
    Attention-based LVR Head (Memory Optimized)
    Uses cross-attention mechanism to focus on relevant image tokens.
    Query is projected from hidden_state via MLP, then attends to image_embeds.
    
    Memory optimizations:
    1. Chunked processing: Process long image sequences in chunks to avoid large intermediate tensors
    2. Online softmax: Compute weighted sum without storing full attention scores
    3. Gradient checkpointing support: Can be used with torch.utils.checkpoint
    """
    def __init__(self, hidden_size: int, query_dim: int = None, num_heads: int = 12, 
                 num_layers: int = 1, dropout: float = 0.1, mlp_ratio: float = 1.0,
                 chunk_size: Optional[int] = None, use_flash_attention: bool = False,
                 use_sparse_attention: bool = False, sparse_mode: str = "topk", 
                 top_k: Optional[int] = None, sparse_ratio: float = 0.25,
                 grad_norm_threshold: Optional[float] = None, skip_on_grad_explosion: bool = True,
                 output_value_threshold: Optional[float] = None):
        super().__init__()
        self.hidden_size = hidden_size
        # query_dim, num_heads, num_layers, dropout are kept for backward compatibility but not used
        
        # Use mlp_ratio to control projection dimension
        # mlp_ratio=1.0: hidden_size -> hidden_size (minimal params)
        # mlp_ratio=0.5: hidden_size -> hidden_size/2 (fewer params, but need projection back)
        # mlp_ratio=0.25: hidden_size -> hidden_size/4 (even fewer params)
        proj_dim = int(hidden_size * mlp_ratio)
        
        # Minimal projection: single linear layer
        # Using bias=False to save parameters
        self.query_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        
        # If projection dimension is smaller than hidden_size, we need to project back
        # for computing attention scores with image_embeds
        if proj_dim < hidden_size:
            self.query_expand = nn.Linear(proj_dim, hidden_size, bias=False)
        else:
            self.query_expand = None
        
        # Optional: scale parameter for attention temperature (single parameter)
        self.scale = nn.Parameter(torch.ones(1) * (hidden_size ** -0.5))
        
        # Output normalization (minimal, can be removed if needed)
        self.output_norm = LayerNorm(hidden_size, eps=1e-6)
        
        # Chunk size for memory-efficient processing
        # If None, will auto-determine based on available memory or use full sequence
        # Recommended: 512-1024 for typical GPU memory constraints
        # Use smaller default for memory-constrained environments
        self.chunk_size = chunk_size if chunk_size is not None else 512
        self._original_chunk_size = self.chunk_size  # Store original for reset
        
        # Flash Attention support
        # Note: Flash Attention requires multi-head format, so we use num_heads=1
        # and treat hidden_size as head_dim
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE
        if self.use_flash_attention:
            print(f"[LVRHeadAttention] Flash Attention enabled (num_heads=1, head_dim={hidden_size})")
        elif use_flash_attention and not FLASH_ATTN_AVAILABLE:
            print("[LVRHeadAttention] Warning: Flash Attention requested but not available. Falling back to standard attention.")
        
        # Auto mode selection threshold
        # Automatically choose between standard/chunked/flash attention based on sequence length
        self.auto_mode_threshold_standard = 512  # Use standard attention if num_tokens <= this
        self.auto_mode_threshold_chunked = 2048  # Use chunked if num_tokens > standard but <= this
        # Flash attention is used if enabled and num_tokens > chunked threshold
        
        # Sparse attention support for further memory optimization
        # Options: "topk" (top-k tokens), "window" (local window), "block" (block sparse)
        self.use_sparse_attention = use_sparse_attention
        self.sparse_mode = sparse_mode if sparse_mode in ["topk", "window", "block"] else "topk"
        self.sparse_ratio = sparse_ratio  # Ratio of tokens to keep (for topk mode)
        # top_k can be specified directly, or computed from sparse_ratio
        self.top_k = top_k
        
        if self.use_sparse_attention:
            print(f"[LVRHeadAttention] Sparse attention enabled: mode={self.sparse_mode}, "
                  f"sparse_ratio={self.sparse_ratio}, top_k={self.top_k}")
        
        # Gradient explosion detection and skip mechanism for ZeRO-3 compatibility
        # grad_norm_threshold: If None, defaults to 100.0 (reasonable threshold for most cases)
        # Set to a lower value (e.g., 10.0) for stricter control, or higher (e.g., 1000.0) for more tolerance
        self.grad_norm_threshold = grad_norm_threshold if grad_norm_threshold is not None else 100.0
        self.skip_on_grad_explosion = skip_on_grad_explosion
        # output_value_threshold: Maximum absolute value in output tensor before considering it abnormal
        # If None, defaults to 1e6 (very large value, effectively disabled)
        self.output_value_threshold = output_value_threshold if output_value_threshold is not None else 1e6
        
        # Track if current forward pass should skip gradient update
        self._skip_gradient = False
        # Register backward hook to detect gradient explosion
        if self.skip_on_grad_explosion:
            # Try to use register_full_backward_hook (PyTorch 1.9+)
            # Fallback to register_backward_hook for older versions
            try:
                self.register_full_backward_hook(self._gradient_explosion_hook)
            except AttributeError:
                # Fallback for older PyTorch versions
                self.register_backward_hook(self._gradient_explosion_hook_legacy)
            print(f"[LVRHeadAttention] Gradient explosion detection enabled: threshold={self.grad_norm_threshold}, "
                  f"output_value_threshold={self.output_value_threshold}")
    
    def _gradient_explosion_hook(self, module, grad_input, grad_output):
        """
        Backward hook to detect gradient explosion (full backward hook, PyTorch 1.9+).
        If gradient norm exceeds threshold, zero out gradients to prevent NCCL timeout.
        
        Args:
            module: The module that registered this hook
            grad_input: Gradients w.r.t. inputs
            grad_output: Gradients w.r.t. outputs (tuple of one element)
        
        Returns:
            Modified grad_output tuple or None (to keep original gradients)
        """
        if not self.skip_on_grad_explosion or grad_output is None:
            return None
        
        if len(grad_output) == 0 or grad_output[0] is None:
            return None
        
        grad = grad_output[0]
        
        # Check for NaN or Inf in gradients
        if grad.isnan().any() or grad.isinf().any():
            import os
            if os.getenv("LVR_DEBUG", "0") == "1":
                print(f"[LVRHeadAttention] WARNING: NaN/Inf detected in gradients, zeroing out gradients")
            self._skip_gradient = True
            return (torch.zeros_like(grad),)
        
        # Compute gradient norm
        try:
            grad_norm = torch.norm(grad, p=2).item()
            
            # Check if gradient norm exceeds threshold
            if grad_norm > self.grad_norm_threshold:
                import os
                if os.getenv("LVR_DEBUG", "0") == "1":
                    print(f"[LVRHeadAttention] WARNING: Gradient explosion detected (norm={grad_norm:.2f} > "
                          f"threshold={self.grad_norm_threshold:.2f}), zeroing out gradients to prevent NCCL timeout")
                self._skip_gradient = True
                # Zero out gradients to prevent NCCL timeout
                return (torch.zeros_like(grad),)
        except Exception as e:
            # If norm computation fails (e.g., due to memory issues), zero out gradients
            import os
            if os.getenv("LVR_DEBUG", "0") == "1":
                print(f"[LVRHeadAttention] WARNING: Failed to compute gradient norm: {e}, zeroing out gradients")
            self._skip_gradient = True
            return (torch.zeros_like(grad),)
        
        return None
    
    def _gradient_explosion_hook_legacy(self, module, grad_output):
        """
        Backward hook for older PyTorch versions (only receives grad_output).
        If gradient norm exceeds threshold, zero out gradients to prevent NCCL timeout.
        
        Args:
            module: The module that registered this hook
            grad_output: Gradients w.r.t. outputs (tuple of one element)
        
        Returns:
            Modified grad_output tuple or None (to keep original gradients)
        """
        if not self.skip_on_grad_explosion or grad_output is None:
            return None
        
        if len(grad_output) == 0 or grad_output[0] is None:
            return None
        
        grad = grad_output[0]
        
        # Check for NaN or Inf in gradients
        if grad.isnan().any() or grad.isinf().any():
            import os
            if os.getenv("LVR_DEBUG", "0") == "1":
                print(f"[LVRHeadAttention] WARNING: NaN/Inf detected in gradients, zeroing out gradients")
            self._skip_gradient = True
            return (torch.zeros_like(grad),)
        
        # Compute gradient norm
        try:
            grad_norm = torch.norm(grad, p=2).item()
            
            # Check if gradient norm exceeds threshold
            if grad_norm > self.grad_norm_threshold:
                import os
                if os.getenv("LVR_DEBUG", "0") == "1":
                    print(f"[LVRHeadAttention] WARNING: Gradient explosion detected (norm={grad_norm:.2f} > "
                          f"threshold={self.grad_norm_threshold:.2f}), zeroing out gradients to prevent NCCL timeout")
                self._skip_gradient = True
                # Zero out gradients to prevent NCCL timeout
                return (torch.zeros_like(grad),)
        except Exception as e:
            # If norm computation fails (e.g., due to memory issues), zero out gradients
            import os
            if os.getenv("LVR_DEBUG", "0") == "1":
                print(f"[LVRHeadAttention] WARNING: Failed to compute gradient norm: {e}, zeroing out gradients")
            self._skip_gradient = True
            return (torch.zeros_like(grad),)
        
        return None
    
    def _check_output_abnormal(self, output: torch.Tensor) -> bool:
        """
        Check if output tensor contains abnormal values (NaN, Inf, or extremely large values).
        
        Args:
            output: Output tensor to check
        
        Returns:
            True if abnormal values detected, False otherwise
        """
        # Check for NaN or Inf
        if output.isnan().any() or output.isinf().any():
            return True
        
        # Check for extremely large values
        if self.output_value_threshold < float('inf'):
            max_abs_value = output.abs().max().item()
            if max_abs_value > self.output_value_threshold:
                return True
        
        return False
    
    def _sparse_attention_topk_2d(
        self,
        query: torch.Tensor,
        image_embeds: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """
        Top-K sparse attention for 2D image_embeds.
        Only attends to top-k most relevant tokens, significantly reducing memory.
        
        Args:
            query: (batch_size, hidden_size)
            image_embeds: (num_image_tokens, hidden_size)
            top_k: Number of top tokens to attend to
            
        Returns:
            v_focal: (batch_size, hidden_size)
        """
        import time
        import os
        debug_enabled = os.getenv("LVR_DEBUG", "0") == "1"
        start_time = time.time() if debug_enabled else None
        
        batch_size = query.shape[0]
        num_tokens = image_embeds.shape[0]
        hidden_size = image_embeds.shape[1]
        device = query.device
        dtype = query.dtype
        
        # Clamp top_k to valid range
        top_k = min(top_k, num_tokens)
        
        if debug_enabled:
            print(f"[_sparse_attention_topk_2d] START: num_tokens={num_tokens}, top_k={top_k}, "
                  f"sparsity={1.0 - top_k/num_tokens:.2%}")
        
        # Step 1: Compute attention scores for all tokens (coarse-grained)
        # Use a lightweight projection to reduce computation
        # Memory: O(batch_size × num_tokens) - but we'll only keep top-k
        attention_scores = torch.matmul(query, image_embeds.t())  # (batch_size, num_tokens)
        
        # Step 2: Select top-k tokens for each batch item
        # Memory: O(batch_size × top_k) instead of O(batch_size × num_tokens)
        top_k_scores, top_k_indices = torch.topk(attention_scores, k=top_k, dim=-1)  # (batch_size, top_k)
        
        # Free full attention scores immediately
        del attention_scores
        
        # Step 3: Gather top-k image embeddings
        # For batched top-k selection, we need to handle each batch item separately
        # or use advanced indexing
        top_k_embeds_list = []
        for i in range(batch_size):
            top_k_embeds_list.append(image_embeds[top_k_indices[i]])  # (top_k, hidden_size)
        top_k_embeds = torch.stack(top_k_embeds_list, dim=0)  # (batch_size, top_k, hidden_size)
        
        # Step 4: Recompute attention scores for top-k tokens (more accurate)
        # This ensures we have accurate scores for the selected tokens
        query_expanded = query.unsqueeze(1)  # (batch_size, 1, hidden_size)
        top_k_scores_refined = torch.bmm(
            query_expanded,
            top_k_embeds.transpose(1, 2)  # (batch_size, hidden_size, top_k)
        ).squeeze(1)  # (batch_size, top_k)
        
        # Step 5: Compute softmax over top-k tokens
        attention_weights = torch.softmax(top_k_scores_refined, dim=-1)  # (batch_size, top_k)
        
        # Step 6: Weighted sum over top-k tokens
        attention_weights_expanded = attention_weights.unsqueeze(1)  # (batch_size, 1, top_k)
        v_focal = torch.bmm(attention_weights_expanded, top_k_embeds).squeeze(1)  # (batch_size, hidden_size)
        
        if debug_enabled:
            print(f"[_sparse_attention_topk_2d] COMPLETE: elapsed={time.time()-start_time:.3f}s, "
                  f"memory_reduction={1.0 - top_k/num_tokens:.2%}")
        
        return v_focal
    
    def _sparse_attention_topk_3d(
        self,
        query: torch.Tensor,
        image_embeds: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor],
        top_k: int
    ) -> torch.Tensor:
        """
        Top-K sparse attention for 3D image_embeds (batched format).
        Only attends to top-k most relevant tokens per batch item.
        
        Args:
            query: (batch_size, hidden_size)
            image_embeds: (batch_size, max_num_tokens, hidden_size)
            image_attention_mask: Optional (batch_size, max_num_tokens)
            top_k: Number of top tokens to attend to
            
        Returns:
            v_focal: (batch_size, hidden_size)
        """
        import time
        import os
        debug_enabled = os.getenv("LVR_DEBUG", "0") == "1"
        start_time = time.time() if debug_enabled else None
        
        batch_size = query.shape[0]
        max_num_tokens = image_embeds.shape[1]
        hidden_size = image_embeds.shape[2]
        device = query.device
        dtype = query.dtype
        
        # Clamp top_k to valid range
        top_k = min(top_k, max_num_tokens)
        
        if debug_enabled:
            print(f"[_sparse_attention_topk_3d] START: batch_size={batch_size}, max_num_tokens={max_num_tokens}, "
                  f"top_k={top_k}, sparsity={1.0 - top_k/max_num_tokens:.2%}")
        
        # Step 1: Compute attention scores for all tokens
        attention_scores = torch.bmm(
            query.unsqueeze(1),  # (batch_size, 1, hidden_size)
            image_embeds.transpose(1, 2)  # (batch_size, hidden_size, max_num_tokens)
        ).squeeze(1)  # (batch_size, max_num_tokens)
        
        # Apply mask if provided (set padding to -inf before top-k)
        if image_attention_mask is not None:
            attention_scores = attention_scores.masked_fill(~image_attention_mask, float('-inf'))
        
        # Step 2: Select top-k tokens for each batch item
        top_k_scores, top_k_indices = torch.topk(attention_scores, k=top_k, dim=-1)  # (batch_size, top_k)
        
        # Free full attention scores immediately
        del attention_scores
        
        # Step 3: Gather top-k image embeddings using advanced indexing
        # Create batch indices for advanced indexing
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, top_k)  # (batch_size, top_k)
        top_k_embeds = image_embeds[batch_indices, top_k_indices]  # (batch_size, top_k, hidden_size)
        
        # Step 4: Recompute attention scores for top-k tokens (more accurate)
        top_k_scores_refined = torch.bmm(
            query.unsqueeze(1),  # (batch_size, 1, hidden_size)
            top_k_embeds.transpose(1, 2)  # (batch_size, hidden_size, top_k)
        ).squeeze(1)  # (batch_size, top_k)
        
        # Step 5: Compute softmax over top-k tokens
        attention_weights = torch.softmax(top_k_scores_refined, dim=-1)  # (batch_size, top_k)
        
        # Step 6: Weighted sum over top-k tokens
        v_focal = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, top_k)
            top_k_embeds  # (batch_size, top_k, hidden_size)
        ).squeeze(1)  # (batch_size, hidden_size)
        
        if debug_enabled:
            print(f"[_sparse_attention_topk_3d] COMPLETE: elapsed={time.time()-start_time:.3f}s, "
                  f"memory_reduction={1.0 - top_k/max_num_tokens:.2%}")
        
        return v_focal
    
    def _chunked_attention_2d(
        self,
        query: torch.Tensor,
        image_embeds: torch.Tensor,
        chunk_size: int
    ) -> torch.Tensor:
        """
        Memory-efficient chunked attention for 2D image_embeds (legacy format).
        Uses online softmax algorithm to avoid storing full attention scores.
        Optimized: Reduced numerical stability checks for better performance.
        
        Args:
            query: (batch_size, hidden_size)
            image_embeds: (num_image_tokens, hidden_size)
            chunk_size: Size of each chunk
            
        Returns:
            v_focal: (batch_size, hidden_size)
        """
        import time
        import os
        debug_enabled = os.getenv("LVR_DEBUG", "0") == "1"
        start_time = time.time() if debug_enabled else None
        
        batch_size = query.shape[0]
        num_tokens = image_embeds.shape[0]
        hidden_size = image_embeds.shape[1]
        device = query.device
        dtype = query.dtype
        
        if debug_enabled:
            print(f"[_chunked_attention_2d] START: num_tokens={num_tokens}, chunk_size={chunk_size}, "
                  f"num_chunks={(num_tokens + chunk_size - 1) // chunk_size}")
        
        # Initialize online softmax accumulators
        # Using numerically stable online softmax algorithm
        max_score = torch.full((batch_size,), float('-inf'), device=device, dtype=dtype)
        exp_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        weighted_sum = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        
        # Optimized: Pre-compute clamp bounds to avoid repeated tensor creation
        CLAMP_MIN, CLAMP_MAX = -50.0, 50.0
        CLAMP_MIN_SCORE, CLAMP_MAX_SCORE = -1e4, 1e4
        
        # Process in chunks
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size
        for chunk_idx, chunk_start in enumerate(range(0, num_tokens, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, num_tokens)
            if debug_enabled and chunk_idx % 10 == 0:
                print(f"[_chunked_attention_2d] Processing chunk {chunk_idx+1}/{num_chunks} "
                      f"(tokens {chunk_start}-{chunk_end}), elapsed={time.time()-start_time:.3f}s")
            chunk_embeds = image_embeds[chunk_start:chunk_end]  # (chunk_size, hidden_size)
            
            # Compute attention scores for this chunk
            chunk_scores = torch.matmul(query, chunk_embeds.t())  # (batch_size, chunk_size)
            
            # Update max_score (optimized: reduced checks)
            chunk_max = chunk_scores.max(dim=-1)[0]  # (batch_size,)
            # Only clamp if values are extreme (most cases won't need this)
            if chunk_max.isnan().any() or (chunk_max < CLAMP_MIN_SCORE).any() or (chunk_max > CLAMP_MAX_SCORE).any():
                chunk_max = torch.nan_to_num(chunk_max, nan=float('-inf'))
                chunk_max = torch.clamp(chunk_max, min=CLAMP_MIN_SCORE, max=CLAMP_MAX_SCORE)
            
            old_max = max_score
            max_score = torch.maximum(max_score, chunk_max)
            # Only clamp if needed
            if (max_score < CLAMP_MIN_SCORE).any() or (max_score > CLAMP_MAX_SCORE).any():
                max_score = torch.clamp(max_score, min=CLAMP_MIN_SCORE, max=CLAMP_MAX_SCORE)
            
            # Adjust previous accumulators by exp(old_max - new_max)
            if chunk_start > 0:
                diff = old_max - max_score
                # Only apply nan_to_num and clamp if values are extreme
                if diff.isnan().any() or (diff < CLAMP_MIN).any() or (diff > CLAMP_MAX).any():
                    diff = torch.nan_to_num(diff, nan=0.0, posinf=CLAMP_MAX, neginf=CLAMP_MIN)
                    diff = torch.clamp(diff, min=CLAMP_MIN, max=CLAMP_MAX)
                exp_adjust = torch.exp(diff)  # (batch_size,)
                exp_sum = exp_sum * exp_adjust
                weighted_sum = weighted_sum * exp_adjust.unsqueeze(-1)
            
            # Compute exp for current chunk (relative to current max_score)
            scores_diff = chunk_scores - max_score.unsqueeze(-1)
            # Optimized: Only clamp if values are extreme
            if scores_diff.isnan().any() or (scores_diff < CLAMP_MIN).any() or (scores_diff > CLAMP_MAX).any():
                scores_diff = torch.nan_to_num(scores_diff, nan=-50.0, posinf=CLAMP_MAX, neginf=CLAMP_MIN)
                scores_diff = torch.clamp(scores_diff, min=CLAMP_MIN, max=CLAMP_MAX)
            
            chunk_exp = torch.exp(scores_diff)  # (batch_size, chunk_size)
            # Only check for NaN/Inf if there's a possibility (rare in practice)
            if chunk_exp.isnan().any() or chunk_exp.isinf().any():
                chunk_exp = torch.nan_to_num(chunk_exp, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure sum result maintains the same dtype
            chunk_exp_sum = chunk_exp.sum(dim=-1)  # (batch_size,)
            if chunk_exp_sum.dtype != dtype:
                chunk_exp_sum = chunk_exp_sum.to(dtype=dtype)
            exp_sum = exp_sum + chunk_exp_sum  # (batch_size,)
            
            # Accumulate weighted sum
            # Free chunk_scores first to free memory
            del chunk_scores
            
            # Try matmul without making contiguous first (saves memory)
            try:
                weighted_chunk = torch.matmul(chunk_exp, chunk_embeds)  # (batch_size, hidden_size)
            except RuntimeError as e:
                # If matmul fails (e.g., memory error), try making contiguous
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if not chunk_exp.is_contiguous():
                    chunk_exp = chunk_exp.contiguous()
                if not chunk_embeds.is_contiguous():
                    chunk_embeds = chunk_embeds.contiguous()
                weighted_chunk = torch.matmul(chunk_exp, chunk_embeds)  # (batch_size, hidden_size)
            
            # Only check for NaN/Inf if there's a possibility
            if weighted_chunk.isnan().any() or weighted_chunk.isinf().any():
                weighted_chunk = torch.nan_to_num(weighted_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            if weighted_chunk.dtype != dtype:
                weighted_chunk = weighted_chunk.to(dtype=dtype)
            weighted_sum = weighted_sum + weighted_chunk
            
            # Free chunk tensors immediately
            del chunk_exp, chunk_embeds
        
        # Normalize to get final weighted sum (attention-weighted image features)
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        exp_sum_safe = torch.clamp(exp_sum, min=eps.item())
        v_focal = weighted_sum / (exp_sum_safe.unsqueeze(-1) + eps)  # (batch_size, hidden_size)
        
        # Final safety check: only if needed
        if v_focal.isnan().any() or v_focal.isinf().any():
            v_focal = torch.nan_to_num(v_focal, nan=0.0, posinf=0.0, neginf=0.0)
        
        if debug_enabled:
            print(f"[_chunked_attention_2d] COMPLETE: processed {num_chunks} chunks, total elapsed={time.time()-start_time:.3f}s")
        
        return v_focal
    
    def _chunked_attention_3d(
        self,
        query: torch.Tensor,
        image_embeds: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor],
        chunk_size: int
    ) -> torch.Tensor:
        """
        Memory-efficient chunked attention for 3D image_embeds (batched format).
        Uses online softmax algorithm to avoid storing full attention scores.
        Optimized: Reduced numerical stability checks for better performance.
        
        Args:
            query: (batch_size, hidden_size)
            image_embeds: (batch_size, max_num_tokens, hidden_size)
            image_attention_mask: Optional (batch_size, max_num_tokens)
            chunk_size: Size of each chunk
            
        Returns:
            v_focal: (batch_size, hidden_size)
        """
        import time
        import os
        debug_enabled = os.getenv("LVR_DEBUG", "0") == "1"
        start_time = time.time() if debug_enabled else None
        
        batch_size = query.shape[0]
        max_num_tokens = image_embeds.shape[1]
        hidden_size = image_embeds.shape[2]
        device = query.device
        dtype = query.dtype
        
        if debug_enabled:
            print(f"[_chunked_attention_3d] START: batch_size={batch_size}, max_num_tokens={max_num_tokens}, "
                  f"chunk_size={chunk_size}, num_chunks={(max_num_tokens + chunk_size - 1) // chunk_size}")
        
        # Initialize online softmax accumulators
        max_score = torch.full((batch_size,), float('-inf'), device=device, dtype=dtype)
        exp_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        weighted_sum = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        
        # Optimized: Pre-compute clamp bounds to avoid repeated tensor creation
        CLAMP_MIN, CLAMP_MAX = -50.0, 50.0
        CLAMP_MIN_SCORE, CLAMP_MAX_SCORE = -1e4, 1e4
        
        # Process in chunks
        num_chunks = (max_num_tokens + chunk_size - 1) // chunk_size
        for chunk_idx, chunk_start in enumerate(range(0, max_num_tokens, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, max_num_tokens)
            if debug_enabled and chunk_idx % 10 == 0:
                print(f"[_chunked_attention_3d] Processing chunk {chunk_idx+1}/{num_chunks} "
                      f"(tokens {chunk_start}-{chunk_end}), elapsed={time.time()-start_time:.3f}s")
            chunk_embeds = image_embeds[:, chunk_start:chunk_end]  # (batch_size, chunk_size, hidden_size)
            
            # Compute attention scores for this chunk
            chunk_scores = torch.bmm(
                query.unsqueeze(1),  # (batch_size, 1, hidden_size)
                chunk_embeds.transpose(1, 2)  # (batch_size, hidden_size, chunk_size)
            ).squeeze(1)  # (batch_size, chunk_size)
            
            # Apply padding mask if provided
            if image_attention_mask is not None:
                chunk_mask = image_attention_mask[:, chunk_start:chunk_end]  # (batch_size, chunk_size)
                chunk_scores = chunk_scores.masked_fill(~chunk_mask, float('-inf'))
            
            # Update max_score (optimized: reduced checks)
            chunk_max = chunk_scores.max(dim=-1)[0]  # (batch_size,)
            # Only clamp if values are extreme
            if chunk_max.isnan().any() or (chunk_max < CLAMP_MIN_SCORE).any() or (chunk_max > CLAMP_MAX_SCORE).any():
                chunk_max = torch.nan_to_num(chunk_max, nan=float('-inf'))
                chunk_max = torch.clamp(chunk_max, min=CLAMP_MIN_SCORE, max=CLAMP_MAX_SCORE)
            
            old_max = max_score
            max_score = torch.maximum(max_score, chunk_max)
            # Only clamp if needed
            if (max_score < CLAMP_MIN_SCORE).any() or (max_score > CLAMP_MAX_SCORE).any():
                max_score = torch.clamp(max_score, min=CLAMP_MIN_SCORE, max=CLAMP_MAX_SCORE)
            
            # Adjust previous accumulators by exp(old_max - new_max)
            if chunk_start > 0:
                diff = old_max - max_score
                # Only apply nan_to_num and clamp if values are extreme
                if diff.isnan().any() or (diff < CLAMP_MIN).any() or (diff > CLAMP_MAX).any():
                    diff = torch.nan_to_num(diff, nan=0.0, posinf=CLAMP_MAX, neginf=CLAMP_MIN)
                    diff = torch.clamp(diff, min=CLAMP_MIN, max=CLAMP_MAX)
                exp_adjust = torch.exp(diff)  # (batch_size,)
                exp_sum = exp_sum * exp_adjust
                weighted_sum = weighted_sum * exp_adjust.unsqueeze(-1)
            
            # Compute exp for current chunk (relative to current max_score)
            scores_diff = chunk_scores - max_score.unsqueeze(-1)
            # Optimized: Only clamp if values are extreme
            if scores_diff.isnan().any() or (scores_diff < CLAMP_MIN).any() or (scores_diff > CLAMP_MAX).any():
                scores_diff = torch.nan_to_num(scores_diff, nan=-50.0, posinf=CLAMP_MAX, neginf=CLAMP_MIN)
                scores_diff = torch.clamp(scores_diff, min=CLAMP_MIN, max=CLAMP_MAX)
            
            chunk_exp = torch.exp(scores_diff)  # (batch_size, chunk_size)
            # Only check for NaN/Inf if there's a possibility
            if chunk_exp.isnan().any() or chunk_exp.isinf().any():
                chunk_exp = torch.nan_to_num(chunk_exp, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply mask to exp (set to 0 for padding)
            if image_attention_mask is not None:
                mask_tensor = chunk_mask.to(dtype=dtype)
                chunk_exp = chunk_exp * mask_tensor
            
            # Ensure sum result maintains the same dtype
            chunk_exp_sum = chunk_exp.sum(dim=-1)  # (batch_size,)
            if chunk_exp_sum.dtype != dtype:
                chunk_exp_sum = chunk_exp_sum.to(dtype=dtype)
            exp_sum = exp_sum + chunk_exp_sum  # (batch_size,)
            
            # Accumulate weighted sum
            del chunk_scores
            
            # Try bmm without making contiguous first (saves memory)
            chunk_exp_unsqueezed = chunk_exp.unsqueeze(1)  # (batch_size, 1, chunk_size)
            try:
                weighted_chunk = torch.bmm(chunk_exp_unsqueezed, chunk_embeds).squeeze(1)  # (batch_size, hidden_size)
            except RuntimeError as e:
                del chunk_exp_unsqueezed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                chunk_exp_unsqueezed = chunk_exp.unsqueeze(1)
                if not chunk_exp_unsqueezed.is_contiguous():
                    chunk_exp_unsqueezed = chunk_exp_unsqueezed.contiguous()
                if not chunk_embeds.is_contiguous():
                    chunk_embeds = chunk_embeds.contiguous()
                weighted_chunk = torch.bmm(chunk_exp_unsqueezed, chunk_embeds).squeeze(1)  # (batch_size, hidden_size)
            
            # Only check for NaN/Inf if there's a possibility
            if weighted_chunk.isnan().any() or weighted_chunk.isinf().any():
                weighted_chunk = torch.nan_to_num(weighted_chunk, nan=0.0, posinf=0.0, neginf=0.0)
            if weighted_chunk.dtype != dtype:
                weighted_chunk = weighted_chunk.to(dtype=dtype)
            weighted_sum = weighted_sum + weighted_chunk
            
            # Free chunk tensors immediately
            del chunk_exp, chunk_exp_unsqueezed, chunk_embeds
        
        # Normalize
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        exp_sum_safe = torch.clamp(exp_sum, min=eps.item())
        v_focal = weighted_sum / (exp_sum_safe.unsqueeze(-1) + eps)  # (batch_size, hidden_size)
        
        # Final safety check: only if needed
        if v_focal.isnan().any() or v_focal.isinf().any():
            v_focal = torch.nan_to_num(v_focal, nan=0.0, posinf=0.0, neginf=0.0)
        
        if debug_enabled:
            print(f"[_chunked_attention_3d] COMPLETE: processed {num_chunks} chunks, total elapsed={time.time()-start_time:.3f}s")
        
        return v_focal
    
    def _flash_attention_2d(
        self,
        query: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Use Flash Attention for 2D image_embeds (legacy format).
        Memory-optimized: processes batch items separately to avoid large tensor allocation.
        
        Args:
            query: (batch_size, hidden_size)
            image_embeds: (num_image_tokens, hidden_size)
            
        Returns:
            v_focal: (batch_size, hidden_size)
        """
        batch_size = query.shape[0]
        num_tokens = image_embeds.shape[0]
        hidden_size = query.shape[1]
        device = query.device
        
        # Optimized: Process batch items separately to avoid repeat() memory overhead
        # This reduces peak memory from O(batch_size × num_tokens × hidden_size) 
        # to O(num_tokens × hidden_size) per iteration
        if batch_size == 1:
            # Single batch item: direct processing
            q = query.unsqueeze(1).unsqueeze(2)  # (1, 1, 1, hidden_size)
            if not q.is_contiguous():
                q = q.contiguous()
            
            k = image_embeds.unsqueeze(0).unsqueeze(2)  # (1, num_tokens, 1, hidden_size)
            if not k.is_contiguous():
                k = k.contiguous()
            v = k.clone()
            
            softmax_scale = self.scale.item() if self.scale.numel() == 1 else None
            
            output = flash_attn_func(
                q, k, v,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
                return_attn_probs=False
            )  # (1, 1, 1, hidden_size)
            
            v_focal = output.squeeze(2).squeeze(1)  # (1, hidden_size)
        else:
            # Multiple batch items: process separately to avoid large tensor allocation
            # Prepare image_embeds once (shared across batch items)
            k_base = image_embeds.unsqueeze(0).unsqueeze(2)  # (1, num_tokens, 1, hidden_size)
            if not k_base.is_contiguous():
                k_base = k_base.contiguous()
            
            softmax_scale = self.scale.item() if self.scale.numel() == 1 else None
            
            # Process each batch item separately
            outputs = []
            for i in range(batch_size):
                q_i = query[i:i+1].unsqueeze(1).unsqueeze(2)  # (1, 1, 1, hidden_size)
                if not q_i.is_contiguous():
                    q_i = q_i.contiguous()
                
                # Clone k/v for this batch item (Flash Attention may modify them)
                k_i = k_base.clone()
                v_i = k_i.clone()
                
                output_i = flash_attn_func(
                    q_i, k_i, v_i,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=False,
                    return_attn_probs=False
                )  # (1, 1, 1, hidden_size)
                
                outputs.append(output_i.squeeze(2).squeeze(1))  # (1, hidden_size)
                
                # Free intermediate tensors
                del q_i, k_i, v_i, output_i
            
            # Concatenate results
            v_focal = torch.cat(outputs, dim=0)  # (batch_size, hidden_size)
            del k_base, outputs
        
        # Only make contiguous if necessary
        if not v_focal.is_contiguous():
            v_focal = v_focal.contiguous()
        
        return v_focal
    
    def _flash_attention_3d(
        self,
        query: torch.Tensor,
        image_embeds: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Use Flash Attention for 3D image_embeds (batched format).
        Memory-optimized: avoids unnecessary clone operations when possible.
        
        Args:
            query: (batch_size, hidden_size)
            image_embeds: (batch_size, max_num_tokens, hidden_size)
            image_attention_mask: Optional (batch_size, max_num_tokens)
            
        Returns:
            v_focal: (batch_size, hidden_size)
        """
        batch_size = query.shape[0]
        
        # Reshape for Flash Attention: (batch_size, seq_len, num_heads, head_dim)
        # Query: (batch_size, 1, num_heads, head_dim)
        q = query.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, hidden_size)
        if not q.is_contiguous():
            q = q.contiguous()
        
        # Key/Value: (batch_size, max_num_tokens, num_heads, head_dim)
        k = image_embeds.unsqueeze(2)  # (batch_size, max_num_tokens, 1, hidden_size)
        # Only make contiguous if necessary to save memory
        if not k.is_contiguous():
            k = k.contiguous()
        # Flash Attention requires separate memory for k and v (may modify them)
        # Only clone if k is not already a copy (check if it shares memory with image_embeds)
        # For memory efficiency, we can try to avoid clone if image_embeds won't be used again
        # But to be safe, we still clone since Flash Attention may modify the tensors
        v = k.clone()
        
        # Flash Attention expects (batch, seqlen, num_heads, head_dim)
        # Use softmax_scale for attention temperature
        softmax_scale = self.scale.item() if self.scale.numel() == 1 else None
        
        # Note: Flash Attention doesn't directly support arbitrary attention masks
        # If mask is needed, this method should not be called (fallback to standard attention)
        
        # Call Flash Attention
        output = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,  # Cross-attention, not causal
            return_attn_probs=False
        )  # (batch_size, 1, num_heads, head_dim)
        
        # Free k, v immediately after use
        del k, v
        
        # Reshape back: (batch_size, hidden_size)
        v_focal = output.squeeze(2).squeeze(1)  # (batch_size, hidden_size)
        # Only make contiguous if necessary
        if not v_focal.is_contiguous():
            v_focal = v_focal.contiguous()
        
        return v_focal
    
    def forward(
        self, 
        hidden_state: torch.Tensor, 
        image_embeds: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: (batch_size, hidden_size) - LLM hidden state h_t
            image_embeds: Can be either:
                - (num_image_tokens, hidden_size) - Single batch item (legacy format)
                - (batch_size, max_num_tokens, hidden_size) - Batched format (padded)
            image_attention_mask: Optional (batch_size, max_num_tokens) - True for valid tokens, False for padding
                                 Only used when image_embeds is batched format
            
        Returns:
            v_focal: (batch_size, hidden_size) - Focused image features
        """
        import time
        import os
        debug_enabled = os.getenv("LVR_DEBUG", "0") == "1"
        start_time = time.time() if debug_enabled else None
        
        # Reset skip gradient flag at the start of each forward pass
        self._skip_gradient = False
        
        batch_size = hidden_state.shape[0]
        
        if debug_enabled:
            print(f"[LVRHeadAttention.forward] START: batch_size={batch_size}, image_embeds.shape={image_embeds.shape}, "
                  f"image_attention_mask={'None' if image_attention_mask is None else image_attention_mask.shape}, "
                  f"use_flash_attention={self.use_flash_attention}, use_sparse_attention={self.use_sparse_attention}")
        
        # Step 0: Check input for abnormal values before processing
        if self.skip_on_grad_explosion:
            if self._check_output_abnormal(hidden_state):
                import os
                if os.getenv("LVR_DEBUG", "0") == "1":
                    print(f"[LVRHeadAttention.forward] WARNING: Abnormal input detected in hidden_state, "
                          f"returning zero output to prevent gradient explosion")
                self._skip_gradient = True
                return torch.zeros(batch_size, self.hidden_size, device=hidden_state.device, dtype=hidden_state.dtype)
            
            if image_embeds.dim() >= 2 and self._check_output_abnormal(image_embeds):
                import os
                if os.getenv("LVR_DEBUG", "0") == "1":
                    print(f"[LVRHeadAttention.forward] WARNING: Abnormal input detected in image_embeds, "
                          f"returning zero output to prevent gradient explosion")
                self._skip_gradient = True
                return torch.zeros(batch_size, self.hidden_size, device=hidden_state.device, dtype=hidden_state.dtype)
        
        # Step 1: Project hidden state to query representation
        if debug_enabled:
            print(f"[LVRHeadAttention.forward] Step 1: Projecting query...")
        query = self.query_proj(hidden_state)  # (batch_size, proj_dim)
        
        # If projection dimension is smaller, expand back to hidden_size for attention computation
        if self.query_expand is not None:
            query = self.query_expand(query)  # (batch_size, hidden_size)
        
        # Apply scale for attention temperature
        query = query * self.scale  # (batch_size, hidden_size)
        
        if debug_enabled:
            print(f"[LVRHeadAttention.forward] Step 1 done: query.shape={query.shape}, elapsed={time.time()-start_time:.3f}s")
        
        # Use Flash Attention if available and enabled
        if self.use_flash_attention:
            if debug_enabled:
                print(f"[LVRHeadAttention.forward] Attempting Flash Attention...")
            try:
                if image_embeds.dim() == 2:
                    # Legacy format: (num_image_tokens, hidden_size)
                    if debug_enabled:
                        print(f"[LVRHeadAttention.forward] Using _flash_attention_2d...")
                    v_focal = self._flash_attention_2d(query, image_embeds)
                    # Apply output normalization
                    v_focal = self.output_norm(v_focal)
                    if debug_enabled:
                        print(f"[LVRHeadAttention.forward] Flash Attention 2D done: elapsed={time.time()-start_time:.3f}s")
                    return v_focal
                elif image_embeds.dim() == 3:
                    # Batched format: (batch_size, max_num_tokens, hidden_size)
                    # Note: Flash Attention has limited support for arbitrary masks
                    # If mask is needed, fall back to standard attention
                    if image_attention_mask is None:
                        # No mask, can use Flash Attention
                        if debug_enabled:
                            print(f"[LVRHeadAttention.forward] Using _flash_attention_3d...")
                        v_focal = self._flash_attention_3d(query, image_embeds, None)
                        # Apply output normalization
                        v_focal = self.output_norm(v_focal)
                        if debug_enabled:
                            print(f"[LVRHeadAttention.forward] Flash Attention 3D done: elapsed={time.time()-start_time:.3f}s")
                        return v_focal
                    # If mask is present, fall through to standard attention
            except Exception as e:
                # If Flash Attention fails, fall back to standard attention
                print(f"[LVRHeadAttention] Flash Attention failed, falling back to standard attention: {e}")
                import traceback
                traceback.print_exc()
                pass
        
        # Sparse attention: Use if enabled and sequence is long enough to benefit
        if self.use_sparse_attention:
            # Determine sequence length and compute top_k
            if image_embeds.dim() == 2:
                num_tokens = image_embeds.shape[0]
                # Only use sparse attention if sequence is long enough
                if num_tokens > self.auto_mode_threshold_standard:
                    # Compute top_k if not specified
                    if self.top_k is None:
                        top_k = max(int(num_tokens * self.sparse_ratio), 32)  # At least 32 tokens
                    else:
                        top_k = self.top_k
                    
                    if debug_enabled:
                        print(f"[LVRHeadAttention.forward] Using sparse attention (top-k): "
                              f"num_tokens={num_tokens}, top_k={top_k}, sparsity={1.0-top_k/num_tokens:.2%}")
                    
                    v_focal = self._sparse_attention_topk_2d(query, image_embeds, top_k)
                    v_focal = self.output_norm(v_focal)
                    if debug_enabled:
                        print(f"[LVRHeadAttention.forward] Sparse attention done: elapsed={time.time()-start_time:.3f}s")
                    return v_focal
            elif image_embeds.dim() == 3:
                max_num_tokens = image_embeds.shape[1]
                # Only use sparse attention if sequence is long enough
                if max_num_tokens > self.auto_mode_threshold_standard:
                    # Compute top_k if not specified
                    if self.top_k is None:
                        top_k = max(int(max_num_tokens * self.sparse_ratio), 32)  # At least 32 tokens
                    else:
                        top_k = self.top_k
                    
                    if debug_enabled:
                        print(f"[LVRHeadAttention.forward] Using sparse attention (top-k): "
                              f"max_num_tokens={max_num_tokens}, top_k={top_k}, sparsity={1.0-top_k/max_num_tokens:.2%}")
                    
                    v_focal = self._sparse_attention_topk_3d(query, image_embeds, image_attention_mask, top_k)
                    v_focal = self.output_norm(v_focal)
                    if debug_enabled:
                        print(f"[LVRHeadAttention.forward] Sparse attention done: elapsed={time.time()-start_time:.3f}s")
                    return v_focal
        
        # Auto mode selection: Choose optimal attention mode based on sequence length
        # Standard attention: num_tokens <= threshold_standard (fastest, but higher memory)
        # Chunked attention: threshold_standard < num_tokens <= threshold_chunked (balanced)
        # Flash attention: num_tokens > threshold_chunked (if enabled, best for long sequences)
        if debug_enabled:
            print(f"[LVRHeadAttention.forward] Using standard attention (Flash Attention not used or failed)")
        
        # Determine sequence length and optimal mode
        if image_embeds.dim() == 2:
            num_tokens = image_embeds.shape[0]
            # Auto-select mode based on thresholds
            if num_tokens <= self.auto_mode_threshold_standard:
                use_chunked = False
                mode = "standard"
            elif num_tokens <= self.auto_mode_threshold_chunked:
                use_chunked = True
                mode = "chunked"
            else:
                # For very long sequences, prefer chunked even if flash attention failed
                use_chunked = True
                mode = "chunked (long sequence)"
            if debug_enabled:
                print(f"[LVRHeadAttention.forward] 2D format: num_tokens={num_tokens}, "
                      f"chunk_size={self.chunk_size}, mode={mode}, use_chunked={use_chunked}")
        elif image_embeds.dim() == 3:
            max_num_tokens = image_embeds.shape[1]
            # Auto-select mode based on thresholds
            if max_num_tokens <= self.auto_mode_threshold_standard:
                use_chunked = False
                mode = "standard"
            elif max_num_tokens <= self.auto_mode_threshold_chunked:
                use_chunked = True
                mode = "chunked"
            else:
                use_chunked = True
                mode = "chunked (long sequence)"
            if debug_enabled:
                print(f"[LVRHeadAttention.forward] 3D format: max_num_tokens={max_num_tokens}, "
                      f"chunk_size={self.chunk_size}, mode={mode}, use_chunked={use_chunked}")
        
        # Check if image_embeds is in legacy format (2D) or batched format (3D)
        if image_embeds.dim() == 2:
            # Legacy format: (num_image_tokens, hidden_size) for single batch item
            if use_chunked:
                # Use memory-efficient chunked processing
                if debug_enabled:
                    print(f"[LVRHeadAttention.forward] Calling _chunked_attention_2d...")
                v_focal = self._chunked_attention_2d(query, image_embeds, self.chunk_size)
                if debug_enabled:
                    print(f"[LVRHeadAttention.forward] _chunked_attention_2d done: elapsed={time.time()-start_time:.3f}s")
            else:
                # Original implementation for short sequences
                if debug_enabled:
                    print(f"[LVRHeadAttention.forward] Using standard 2D attention (non-chunked)...")
                attention_scores = torch.matmul(query, image_embeds.t())  # (batch_size, num_image_tokens)
                attention_mask = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_image_tokens)
                v_focal = torch.matmul(attention_mask, image_embeds)  # (batch_size, hidden_size)
                del attention_scores, attention_mask
                if debug_enabled:
                    print(f"[LVRHeadAttention.forward] Standard 2D attention done: elapsed={time.time()-start_time:.3f}s")
            
        elif image_embeds.dim() == 3:
            # Batched format: (batch_size, max_num_tokens, hidden_size)
            if use_chunked:
                # Use memory-efficient chunked processing
                if debug_enabled:
                    print(f"[LVRHeadAttention.forward] Calling _chunked_attention_3d...")
                v_focal = self._chunked_attention_3d(query, image_embeds, image_attention_mask, self.chunk_size)
                if debug_enabled:
                    print(f"[LVRHeadAttention.forward] _chunked_attention_3d done: elapsed={time.time()-start_time:.3f}s")
            else:
                # Original implementation for short sequences
                if debug_enabled:
                    print(f"[LVRHeadAttention.forward] Using standard 3D attention (non-chunked)...")
                attention_scores = torch.bmm(
                    query.unsqueeze(1),  # (batch_size, 1, hidden_size)
                    image_embeds.transpose(1, 2)  # (batch_size, hidden_size, max_num_tokens)
                ).squeeze(1)  # (batch_size, max_num_tokens)
                
                if image_attention_mask is not None:
                    attention_scores = attention_scores.masked_fill(~image_attention_mask, float('-inf'))
                
                attention_mask = torch.softmax(attention_scores, dim=-1)  # (batch_size, max_num_tokens)
                v_focal = torch.bmm(
                    attention_mask.unsqueeze(1),  # (batch_size, 1, max_num_tokens)
                    image_embeds  # (batch_size, max_num_tokens, hidden_size)
                ).squeeze(1)  # (batch_size, hidden_size)
                del attention_scores, attention_mask
                if debug_enabled:
                    print(f"[LVRHeadAttention.forward] Standard 3D attention done: elapsed={time.time()-start_time:.3f}s")
            
        else:
            raise ValueError(f"image_embeds must be 2D or 3D, got {image_embeds.dim()}D")
        
        # Step 2: Output normalization
        if debug_enabled:
            print(f"[LVRHeadAttention.forward] Applying output normalization...")
        v_focal = self.output_norm(v_focal)  # (batch_size, hidden_size)
        
        # Step 3: Check for abnormal output values (NaN, Inf, or extremely large values)
        # If detected, return zero output to prevent gradient explosion and NCCL timeout
        if self.skip_on_grad_explosion and self._check_output_abnormal(v_focal):
            import os
            if os.getenv("LVR_DEBUG", "0") == "1":
                print(f"[LVRHeadAttention.forward] WARNING: Abnormal output detected (NaN/Inf/large values), "
                      f"returning zero output to prevent gradient explosion and NCCL timeout")
            self._skip_gradient = True
            # Return zero output with same shape and device
            v_focal = torch.zeros_like(v_focal)
        
        if debug_enabled:
            print(f"[LVRHeadAttention.forward] COMPLETE: total elapsed={time.time()-start_time:.3f}s, v_focal.shape={v_focal.shape}")
        
        return v_focal


class LVRHeadImplicitVisualRouting(nn.Module):
    """
    Implicit Visual Routing (IVR) - 基于胶囊网络路由思想的轻量实现
    
    核心特点：
    1. 完全无参数（除了可选的输出归一化层）
    2. 通过迭代优化实现动态聚焦
    
    算法流程：
    1. 初始化路由系数 b
    2. 迭代更新（iterations次）：
       a. 计算注意力系数 c = softmax(b)
       b. 生成当前聚焦特征 s = weighted_sum(visual_tokens, c)
       c. 与语言状态对比更新路由 b = b + similarity(visual_tokens, s)
    3. 返回最终聚焦特征 s
    
    内存优化：
    - 使用在线计算，避免存储大型中间张量
    - 支持chunked处理长序列
    - 无梯度爆炸风险（纯迭代算法）
    """
    def __init__(
        self,
        hidden_size: int,
        iterations: int = 3,
        chunk_size: Optional[int] = None,
        use_output_norm: bool = True,
        temperature: float = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.iterations = iterations
        self.temperature = temperature
        
        # Chunk size for memory-efficient processing of long sequences
        # If None, will auto-select based on sequence length (default: 512)
        # This ensures long sequences are processed efficiently
        self.chunk_size = chunk_size if chunk_size is not None else 512
        self._original_chunk_size = chunk_size  # Store original None for auto-selection
        
        # Optional output normalization (minimal parameter overhead)
        self.use_output_norm = use_output_norm
        if use_output_norm:
            self.output_norm = LayerNorm(hidden_size, eps=1e-6)
        else:
            self.output_norm = None
    
    def _visual_routing_2d(
        self,
        lang_state: torch.Tensor,
        visual_tokens: torch.Tensor,
        iterations: int,
        data_id: str = "unknown"
    ) -> torch.Tensor:
        """
        Implicit Visual Routing for 2D format (legacy).
        
        Args:
            lang_state: (batch_size, hidden_size) - LLM hidden state
            visual_tokens: (num_image_tokens, hidden_size) - Image tokens
            iterations: Number of routing iterations
            data_id: Data identifier for tracking (includes fingerprint and content hash)
            
        Returns:
            s: (batch_size, hidden_size) - Focused visual features
        """
        import time
        batch_size = lang_state.shape[0]
        num_tokens = visual_tokens.shape[0]
        device = lang_state.device
        dtype = lang_state.dtype
        
        iter_start_time = time.time()
        
        # Initialize routing coefficients: b (batch_size, num_tokens)
        # Start with similarity between lang_state and visual_tokens
        # b = lang_state @ visual_tokens.T / temperature
        b = torch.matmul(lang_state, visual_tokens.t()) / self.temperature  # (batch_size, num_tokens)
        
        # Iterative routing updates
        for iter_idx in range(iterations):
            # Step 1: Compute attention coefficients from routing coefficients
            c = F.softmax(b, dim=1)  # (batch_size, num_tokens)
            
            # Step 2: Generate current focused feature
            # s = weighted_sum(visual_tokens, c)
            s = torch.matmul(c, visual_tokens)  # (batch_size, hidden_size)
            
            # Step 3: Update routing coefficients by comparing visual_tokens with focused feature
            # b = b + similarity(visual_tokens, s)
            # Similarity: visual_tokens @ s.T
            similarity_update = torch.matmul(visual_tokens, s.t())  # (num_tokens, batch_size)
            b = b + similarity_update.t() / self.temperature  # (batch_size, num_tokens)
        
        # Final focused feature (use last iteration's attention)
        c_final = F.softmax(b, dim=1)  # (batch_size, num_tokens)
        
        s_final = torch.matmul(c_final, visual_tokens)  # (batch_size, hidden_size)
        
        return s_final
    
    def _visual_routing_2d_chunked(
        self,
        lang_state: torch.Tensor,
        visual_tokens: torch.Tensor,
        iterations: int,
        chunk_size: int,
        data_id: str = "unknown"
    ) -> torch.Tensor:
        """
        Memory-efficient chunked version for 2D format.
        Processes visual tokens in chunks to avoid large intermediate tensors.
        
        Args:
            lang_state: (batch_size, hidden_size)
            visual_tokens: (num_image_tokens, hidden_size)
            iterations: Number of routing iterations
            chunk_size: Size of each chunk
            data_id: Data identifier for tracking (includes fingerprint and content hash)
            
        Returns:
            s: (batch_size, hidden_size) - Focused visual features
        """
        import time
        batch_size = lang_state.shape[0]
        num_tokens = visual_tokens.shape[0]
        hidden_size = lang_state.shape[1]
        device = lang_state.device
        dtype = lang_state.dtype
        
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size
        total_start_time = time.time()
        
        # Initialize routing coefficients with chunked computation
        # Compute initial similarity in chunks
        # Use torch.cat to avoid inplace operations that break gradient computation
        chunk_similarities = []
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, num_tokens)
            chunk_tokens = visual_tokens[chunk_start:chunk_end]  # (chunk_size, hidden_size)
            chunk_similarity = torch.matmul(lang_state, chunk_tokens.t()) / self.temperature
            chunk_similarities.append(chunk_similarity)
        b = torch.cat(chunk_similarities, dim=1)  # (batch_size, num_tokens)
        
        # Iterative routing updates (chunked)
        for iter_idx in range(iterations):
            # Step 1: Compute attention coefficients
            c = F.softmax(b, dim=1)  # (batch_size, num_tokens)
            
            # Step 2: Generate focused feature (chunked weighted sum)
            s = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, num_tokens)
                chunk_tokens = visual_tokens[chunk_start:chunk_end]
                chunk_c = c[:, chunk_start:chunk_end]  # (batch_size, chunk_size)
                s += torch.matmul(chunk_c, chunk_tokens)  # (batch_size, hidden_size)
            
            # Step 3: Update routing coefficients (chunked)
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, num_tokens)
                chunk_tokens = visual_tokens[chunk_start:chunk_end]
                chunk_similarity = torch.matmul(chunk_tokens, s.t())  # (chunk_size, batch_size)
                b[:, chunk_start:chunk_end] += chunk_similarity.t() / self.temperature
        
        # Final focused feature
        c_final = F.softmax(b, dim=1)
        
        s_final = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, num_tokens)
            chunk_tokens = visual_tokens[chunk_start:chunk_end]
            chunk_c = c_final[:, chunk_start:chunk_end]
            s_final += torch.matmul(chunk_c, chunk_tokens)
        
        return s_final
    
    def _visual_routing_3d(
        self,
        lang_state: torch.Tensor,
        visual_tokens: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor],
        iterations: int,
        data_id: str = "unknown"
    ) -> torch.Tensor:
        """
        Implicit Visual Routing for 3D format (batched).
        
        Args:
            lang_state: (batch_size, hidden_size)
            visual_tokens: (batch_size, max_num_tokens, hidden_size)
            image_attention_mask: Optional (batch_size, max_num_tokens) - True for valid tokens
            iterations: Number of routing iterations
            data_id: Data identifier for tracking (includes fingerprint and content hash)
            
        Returns:
            s: (batch_size, hidden_size) - Focused visual features
        """
        import time
        batch_size = lang_state.shape[0]
        max_num_tokens = visual_tokens.shape[1]
        device = lang_state.device
        dtype = lang_state.dtype
        
        total_start_time = time.time()
        
        # Detach visual_tokens to avoid unnecessary gradient computation (vision tower is typically frozen)
        visual_tokens = visual_tokens.detach()
        
        # Initialize routing coefficients: b (batch_size, max_num_tokens)
        # b = lang_state @ visual_tokens.T / temperature
        b = torch.bmm(
            lang_state.unsqueeze(1),  # (batch_size, 1, hidden_size)
            visual_tokens.transpose(1, 2)  # (batch_size, hidden_size, max_num_tokens)
        ).squeeze(1) / self.temperature  # (batch_size, max_num_tokens)
        
        # Apply mask if provided (set padding to -inf before softmax)
        if image_attention_mask is not None:
            b = b.masked_fill(~image_attention_mask, float('-inf'))
        
        # Iterative routing updates
        for iter_idx in range(iterations):
            # Step 1: Compute attention coefficients
            c = F.softmax(b, dim=1)  # (batch_size, max_num_tokens)
            
            # Step 2: Generate current focused feature
            s = torch.bmm(
                c.unsqueeze(1),  # (batch_size, 1, max_num_tokens)
                visual_tokens  # (batch_size, max_num_tokens, hidden_size)
            ).squeeze(1)  # (batch_size, hidden_size)
            
            # Step 3: Update routing coefficients
            similarity_update = torch.bmm(
                visual_tokens,  # (batch_size, max_num_tokens, hidden_size)
                s.unsqueeze(2)  # (batch_size, hidden_size, 1)
            ).squeeze(2) / self.temperature  # (batch_size, max_num_tokens)
            
            b = b + similarity_update
            
            # Re-apply mask
            if image_attention_mask is not None:
                b = b.masked_fill(~image_attention_mask, float('-inf'))
        
        # Final focused feature
        c_final = F.softmax(b, dim=1)
        
        s_final = torch.bmm(
            c_final.unsqueeze(1),
            visual_tokens
        ).squeeze(1)
        
        return s_final
    
    def _visual_routing_3d_chunked(
        self,
        lang_state: torch.Tensor,
        visual_tokens: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor],
        iterations: int,
        chunk_size: int,
        data_id: str = "unknown"
    ) -> torch.Tensor:
        """
        Memory-efficient chunked version for 3D format (batched).
        Processes visual tokens in chunks to avoid large intermediate tensors.
        
        Args:
            lang_state: (batch_size, hidden_size)
            visual_tokens: (batch_size, max_num_tokens, hidden_size)
            image_attention_mask: Optional (batch_size, max_num_tokens) - True for valid tokens
            iterations: Number of routing iterations
            chunk_size: Size of each chunk
            
        Returns:
            s: (batch_size, hidden_size) - Focused visual features
        """
        import time
        batch_size = lang_state.shape[0]
        max_num_tokens = visual_tokens.shape[1]
        hidden_size = lang_state.shape[1]
        device = lang_state.device
        dtype = lang_state.dtype
        
        num_chunks = (max_num_tokens + chunk_size - 1) // chunk_size
        total_start_time = time.time()
        
        # Detach visual_tokens to avoid unnecessary gradient computation (vision tower is typically frozen)
        visual_tokens = visual_tokens.detach()
        
        # Initialize routing coefficients with chunked computation
        # Compute initial similarity in chunks
        # Use torch.cat to avoid inplace operations that break gradient computation
        chunk_similarities = []
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, max_num_tokens)
            chunk_tokens = visual_tokens[:, chunk_start:chunk_end]  # (batch_size, chunk_size, hidden_size)
            # Compute similarity: lang_state @ chunk_tokens.T
            chunk_similarity = torch.bmm(
                lang_state.unsqueeze(1),  # (batch_size, 1, hidden_size)
                chunk_tokens.transpose(1, 2)  # (batch_size, hidden_size, chunk_size)
            ).squeeze(1) / self.temperature  # (batch_size, chunk_size)
            chunk_similarities.append(chunk_similarity)
        b = torch.cat(chunk_similarities, dim=1)  # (batch_size, max_num_tokens)
        
        # Apply mask if provided (set padding to -inf before softmax)
        if image_attention_mask is not None:
            b = b.masked_fill(~image_attention_mask, float('-inf'))
        
        # Iterative routing updates (chunked)
        for iter_idx in range(iterations):
            # Step 1: Compute attention coefficients
            c = F.softmax(b, dim=1)  # (batch_size, max_num_tokens)
            
            # Step 2: Generate focused feature (chunked weighted sum)
            s = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, max_num_tokens)
                chunk_tokens = visual_tokens[:, chunk_start:chunk_end]  # (batch_size, chunk_size, hidden_size)
                chunk_c = c[:, chunk_start:chunk_end].unsqueeze(1)  # (batch_size, 1, chunk_size)
                s += torch.bmm(chunk_c, chunk_tokens).squeeze(1)  # (batch_size, hidden_size)
            
            # Step 3: Update routing coefficients (chunked)
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, max_num_tokens)
                chunk_tokens = visual_tokens[:, chunk_start:chunk_end]  # (batch_size, chunk_size, hidden_size)
                # Compute similarity: chunk_tokens @ s
                chunk_similarity = torch.bmm(
                    chunk_tokens,  # (batch_size, chunk_size, hidden_size)
                    s.unsqueeze(2)  # (batch_size, hidden_size, 1)
                ).squeeze(2) / self.temperature  # (batch_size, chunk_size)
                b[:, chunk_start:chunk_end] += chunk_similarity
            
            # Re-apply mask
            if image_attention_mask is not None:
                b = b.masked_fill(~image_attention_mask, float('-inf'))
        
        # Final focused feature (chunked)
        c_final = F.softmax(b, dim=1)
        
        s_final = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, max_num_tokens)
            chunk_tokens = visual_tokens[:, chunk_start:chunk_end]  # (batch_size, chunk_size, hidden_size)
            chunk_c = c_final[:, chunk_start:chunk_end].unsqueeze(1)  # (batch_size, 1, chunk_size)
            s_final += torch.bmm(chunk_c, chunk_tokens).squeeze(1)  # (batch_size, hidden_size)
        
        return s_final
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        image_embeds: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: (batch_size, hidden_size) - LLM hidden state h_t
            image_embeds: Can be either:
                - (num_image_tokens, hidden_size) - Single batch item (legacy format)
                - (batch_size, max_num_tokens, hidden_size) - Batched format (padded)
            image_attention_mask: Optional (batch_size, max_num_tokens) - True for valid tokens, False for padding
                                 Only used when image_embeds is batched format
            
        Returns:
            v_focal: (batch_size, hidden_size) - Focused image features
        """
        batch_size = hidden_state.shape[0]
        
        # Create data identifier (simplified, no stats computation)
        data_id = "unknown"
    
        
        # Handle 2D format (legacy)
        if image_embeds.dim() == 2:
            num_tokens = image_embeds.shape[0]
            
            # Removed format detection logging
            
            # Auto-select chunked or non-chunked based on sequence length
            # For long sequences (>512 tokens), always use chunked processing for efficiency
            # This prevents slowdowns on long sequences
            effective_chunk_size = self.chunk_size
            if num_tokens > 512 and batch_size == 1:
                # For sequences longer than 512, use chunked processing
                use_chunked = True
                if effective_chunk_size is None or effective_chunk_size > num_tokens:
                    # Auto-select appropriate chunk size
                    if num_tokens > 2048:
                        effective_chunk_size = 512  # Smaller chunks for very long sequences
                    elif num_tokens > 1024:
                        effective_chunk_size = 512
                    else:
                        effective_chunk_size = 256
            else:
                use_chunked = False
            
            if use_chunked:
                v_focal = self._visual_routing_2d_chunked(
                    hidden_state, image_embeds, self.iterations, effective_chunk_size, data_id
                )
            else:
                v_focal = self._visual_routing_2d(
                    hidden_state, image_embeds, self.iterations, data_id
                )
        
        # Handle 3D format (batched)
        elif image_embeds.dim() == 3:
            max_num_tokens = image_embeds.shape[1]
            
            # Compute valid tokens count if mask is provided
            valid_tokens_info = "unknown"
            if image_attention_mask is not None:
                valid_tokens_per_batch = image_attention_mask.sum(dim=1).cpu().tolist()
                valid_tokens_info = f"valid_tokens={valid_tokens_per_batch}"
            
            # Removed format detection loggin
            
            # Auto-select chunked or non-chunked based on sequence length
            # For long sequences (>512 tokens), always use chunked processing for efficiency
            # This prevents OOM on long sequences
            effective_chunk_size = self.chunk_size
            if max_num_tokens > 512:
                # For sequences longer than 512, use chunked processing
                use_chunked = True
                if effective_chunk_size is None or effective_chunk_size > max_num_tokens:
                    # Auto-select appropriate chunk size
                    if max_num_tokens > 2048:
                        effective_chunk_size = 512  # Smaller chunks for very long sequences
                    elif max_num_tokens > 1024:
                        effective_chunk_size = 512
                    else:
                        effective_chunk_size = 256
            else:
                use_chunked = False
            
            if use_chunked:
                v_focal = self._visual_routing_3d_chunked(
                    hidden_state, image_embeds, image_attention_mask, self.iterations, 
                    effective_chunk_size, data_id
                )
            else:
                v_focal = self._visual_routing_3d(
                    hidden_state, image_embeds, image_attention_mask, self.iterations, data_id
                )
        
        else:
            raise ValueError(f"image_embeds must be 2D or 3D, got {image_embeds.dim()}D")
        
        # Apply output normalization if enabled
        if self.use_output_norm:
            v_focal = self.output_norm(v_focal)
        
        # Removed shape and stats logging - focus on gradients only
        
        return v_focal


class LVRHeadGatedFocus(nn.Module):
    """
    Gated Feature Reweighting (GFR) - 轻量级门控特征重加权机制
    
    核心特点：
    1. 极少的参数量（仅约4.2M参数）
    2. 通过门控机制动态重加权视觉特征
    3. 内存高效，避免显存和NCCL内存溢出问题
    
    算法流程：
    1. 使用语言状态的最后一个token生成门控向量
    2. 将门控向量应用到所有视觉token
    3. 对重加权后的特征进行归一化
    4. 聚合为单一向量输出
    
    内存优化：
    - 仅使用单个线性层进行投影
    - 支持chunked处理长序列
    - 避免存储大型中间张量
    - 自动对image tokens进行detach，避免不必要的梯度计算（vision tower通常被冻结）
    """
    def __init__(
        self,
        hidden_size: int,
        visual_dim: Optional[int] = None,
        use_output_norm: bool = True,
        chunk_size: Optional[int] = None,
        save_activation_maps: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # visual_dim 默认等于 hidden_size，但可以设置为更小的值以节省参数
        self.visual_dim = visual_dim if visual_dim is not None else hidden_size
        
        # 门控投影层：text_dim -> visual_dim
        # 参数数量：hidden_size * visual_dim（约4.2M for 7B model）
        self.gate_proj = nn.Linear(hidden_size, self.visual_dim, bias=False)
        
        # 归一化层
        self.norm = LayerNorm(self.visual_dim, eps=1e-6)
        
        # 如果visual_dim != hidden_size，需要投影visual_tokens到visual_dim
        if self.visual_dim != hidden_size:
            self.visual_proj = nn.Linear(hidden_size, self.visual_dim, bias=False)
        else:
            self.visual_proj = None
        
        # 输出归一化（可选）
        self.use_output_norm = use_output_norm
        if use_output_norm and self.visual_dim != hidden_size:
            # 如果visual_dim != hidden_size，需要投影回hidden_size
            self.output_proj = nn.Linear(self.visual_dim, hidden_size, bias=False)
            self.output_norm = LayerNorm(hidden_size, eps=1e-6)
        elif use_output_norm:
            self.output_proj = None
            self.output_norm = LayerNorm(hidden_size, eps=1e-6)
        else:
            self.output_proj = None
            self.output_norm = None
        
        # Chunk size for memory-efficient processing of long sequences
        self.chunk_size = chunk_size if chunk_size is not None else 512
        
        # Activation map visualization
        self.save_activation_maps = save_activation_maps
    
    def _gated_focus_2d(
        self,
        lang_state: torch.Tensor,
        visual_tokens: torch.Tensor,
        return_activations: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Gated Feature Reweighting for 2D format (legacy).
        
        Args:
            lang_state: (batch_size, hidden_size) - LLM hidden state
            visual_tokens: (num_image_tokens, hidden_size) - Image tokens
                          Note: 如果visual_dim != hidden_size，visual_tokens需要先投影
            return_activations: If True, return activation weights for visualization
            
        Returns:
            focused_features: (batch_size, hidden_size) - Focused visual features
            activations: (batch_size, num_tokens) - Optional activation weights per token
        """
        batch_size = lang_state.shape[0]
        num_tokens = visual_tokens.shape[0]
        device = lang_state.device
        dtype = lang_state.dtype
        
        # Detach visual_tokens to avoid unnecessary gradient computation (vision tower is typically frozen)
        visual_tokens = visual_tokens.detach()
        
        # Step 1: 使用语言状态生成门控向量
        gate = torch.sigmoid(self.gate_proj(lang_state))  # (batch_size, visual_dim)
        
        # Step 2: 如果visual_tokens的维度不匹配，需要投影
        if self.visual_proj is not None:
            visual_tokens_proj = self.visual_proj(visual_tokens)  # (num_image_tokens, visual_dim)
        else:
            visual_tokens_proj = visual_tokens  # (num_image_tokens, visual_dim)
        
        # Step 3: 应用门控到所有视觉token
        # gate: (batch_size, visual_dim)
        # visual_tokens_proj: (num_image_tokens, visual_dim)
        # 需要扩展gate到 (batch_size, num_image_tokens, visual_dim)
        gate_expanded = gate.unsqueeze(1)  # (batch_size, 1, visual_dim)
        visual_tokens_expanded = visual_tokens_proj.unsqueeze(0)  # (1, num_image_tokens, visual_dim)
        
        # 应用门控
        focused_features = visual_tokens_expanded * gate_expanded  # (batch_size, num_image_tokens, visual_dim)
        
        # Compute activation weights for visualization (magnitude of gated features per token)
        activations = None
        if return_activations:
            # Compute L2 norm of gated features for each token: (batch_size, num_image_tokens)
            activations = torch.norm(focused_features, dim=-1)
        
        # Step 4: 归一化
        # Ensure dtype compatibility: LayerNorm requires input dtype to match weight dtype
        focused_features = self.norm(focused_features.to(self.norm.weight.dtype))  # (batch_size, num_image_tokens, visual_dim)
        
        # Step 5: 聚合为单一向量（平均池化）
        focused_features = focused_features.mean(dim=1)  # (batch_size, visual_dim)
        
        # Step 6: 投影回hidden_size（如果需要）
        if self.output_proj is not None:
            focused_features = self.output_proj(focused_features)  # (batch_size, hidden_size)
        
        # Step 7: 输出归一化（如果启用）
        if self.use_output_norm:
            focused_features = self.output_norm(focused_features)
        
        return focused_features if not return_activations else (focused_features, activations)


class LVRHeadIntrinsicSimilarity(nn.Module):
    """
    内生相似度映射 (Intrinsic Similarity Gating) - 完全零参数的视觉路由机制
    
    核心特点：
    1. 完全零参数增加（除了可选的输出归一化层）
    2. 直接利用 LLM 隐状态 h_t 和视觉特征 Z_grid 的内生相似度
    3. 数值稳定性极佳（Softmax 保证了输出范围）
    4. 假设 h_t 的语义空间与 Z_grid 的特征空间已经对齐（通过映射层）
    
    算法流程：
    1. 计算 h_t 与所有 Z_i 的点积：similarity = h_t @ Z_i^T
    2. 对相似度进行 Softmax 归一化：attention_weights = Softmax(similarity)
    3. 加权求和得到聚焦特征：V_focal^t = sum(attention_weights * Z_i)
    
    公式：
    V_focal^t = sum_{i=1}^{N} Softmax(h_t · Z_i^T) Z_i
    
    内存优化：
    - 支持 chunked 处理长序列
    - 避免存储大型中间张量
    - 自动对 image tokens 进行 detach，避免不必要的梯度计算（vision tower 通常被冻结）
    """
    def __init__(
        self,
        hidden_size: int,
        use_output_norm: bool = True,
        chunk_size: Optional[int] = None,
        save_activation_maps: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Chunk size for memory-efficient processing of long sequences
        # If None, will auto-select based on sequence length (default: 512)
        self.chunk_size = chunk_size if chunk_size is not None else 512
        self._original_chunk_size = chunk_size  # Store original None for auto-selection
        
        # Optional output normalization (minimal parameter overhead)
        self.use_output_norm = use_output_norm
        if use_output_norm:
            self.output_norm = LayerNorm(hidden_size, eps=1e-6)
        else:
            self.output_norm = None
        
        # Activation map visualization
        self.save_activation_maps = save_activation_maps
    
    def _intrinsic_similarity_2d(
        self,
        h_t: torch.Tensor,
        Z_grid: torch.Tensor,
        return_activations: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        内生相似度映射 for 2D format (legacy).
        
        Args:
            h_t: (batch_size, hidden_size) - LLM hidden state
            Z_grid: (num_image_tokens, hidden_size) - Image tokens
            return_activations: If True, return attention weights for visualization
            
        Returns:
            V_focal: (batch_size, hidden_size) - Focused visual features
            attention_weights: (batch_size, num_image_tokens) - Optional attention weights per token
        """
        # Detach visual_tokens to avoid unnecessary gradient computation (vision tower is typically frozen)
        Z_grid = Z_grid.detach()
        
        # Compute similarity: h_t @ Z_grid^T
        # h_t: (batch_size, hidden_size)
        # Z_grid: (num_image_tokens, hidden_size)
        # similarity: (batch_size, num_image_tokens)
        similarity = torch.matmul(h_t, Z_grid.t())
        
        # Softmax normalization
        attention_weights = F.softmax(similarity, dim=1)  # (batch_size, num_image_tokens)
        
        # Weighted sum: V_focal = sum(attention_weights * Z_grid)
        # attention_weights: (batch_size, num_image_tokens)
        # Z_grid: (num_image_tokens, hidden_size)
        # V_focal: (batch_size, hidden_size)
        V_focal = torch.matmul(attention_weights, Z_grid)
        
        if return_activations:
            return V_focal, attention_weights
        return V_focal
    
    def _intrinsic_similarity_2d_chunked(
        self,
        h_t: torch.Tensor,
        Z_grid: torch.Tensor,
        chunk_size: int
    ) -> torch.Tensor:
        """
        Memory-efficient chunked version for 2D format.
        Uses online softmax algorithm to avoid storing full similarity tensor.
        
        Args:
            h_t: (batch_size, hidden_size)
            Z_grid: (num_image_tokens, hidden_size)
            chunk_size: Size of each chunk
            
        Returns:
            V_focal: (batch_size, hidden_size) - Focused visual features
        """
        batch_size = h_t.shape[0]
        num_tokens = Z_grid.shape[0]
        hidden_size = h_t.shape[1]
        device = h_t.device
        dtype = h_t.dtype
        
        # Detach visual_tokens to avoid unnecessary gradient computation
        Z_grid = Z_grid.detach()
        
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size
        
        # Online softmax algorithm: avoid storing full similarity tensor
        max_scores = torch.full((batch_size,), float('-inf'), device=device, dtype=dtype)
        exp_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        weighted_sum = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        
        # First pass: compute max scores and exp sums
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, num_tokens)
            chunk_tokens = Z_grid[chunk_start:chunk_end]  # (chunk_size, hidden_size)
            
            chunk_similarity = torch.matmul(h_t, chunk_tokens.t())  # (batch_size, chunk_size)
            
            chunk_max = chunk_similarity.max(dim=1)[0]  # (batch_size,)
            old_max = max_scores
            max_scores = torch.maximum(max_scores, chunk_max)
            
            # Adjust previous accumulators
            if chunk_idx > 0:
                exp_adjust = torch.exp(old_max - max_scores)  # (batch_size,)
                exp_sum = exp_sum * exp_adjust
                weighted_sum = weighted_sum * exp_adjust.unsqueeze(-1)
            
            # Compute exp for current chunk
            scores_diff = chunk_similarity - max_scores.unsqueeze(-1)  # (batch_size, chunk_size)
            chunk_exp = torch.exp(scores_diff)  # (batch_size, chunk_size)
            
            exp_sum += chunk_exp.sum(dim=1)  # (batch_size,)
            weighted_sum += torch.matmul(chunk_exp, chunk_tokens)  # (batch_size, hidden_size)
        
        # Normalize
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        V_focal = weighted_sum / (exp_sum.unsqueeze(-1) + eps)
        
        return V_focal
    
    def _intrinsic_similarity_3d(
        self,
        h_t: torch.Tensor,
        Z_grid: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor],
        return_activations: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        内生相似度映射 for 3D format (batched).
        
        Args:
            h_t: (batch_size, hidden_size)
            Z_grid: (batch_size, max_num_tokens, hidden_size)
            image_attention_mask: Optional (batch_size, max_num_tokens) - True for valid tokens
            return_activations: If True, return attention weights for visualization
            
        Returns:
            V_focal: (batch_size, hidden_size) - Focused visual features
            attention_weights: (batch_size, max_num_tokens) - Optional attention weights per token
        """
        # Detach visual_tokens to avoid unnecessary gradient computation (vision tower is typically frozen)
        Z_grid = Z_grid.detach()
        
        # Compute similarity: h_t @ Z_grid^T
        # h_t: (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
        # Z_grid: (batch_size, max_num_tokens, hidden_size) -> (batch_size, hidden_size, max_num_tokens)
        # similarity: (batch_size, max_num_tokens)
        similarity = torch.bmm(
            h_t.unsqueeze(1),  # (batch_size, 1, hidden_size)
            Z_grid.transpose(1, 2)  # (batch_size, hidden_size, max_num_tokens)
        ).squeeze(1)  # (batch_size, max_num_tokens)
        
        # Apply mask if provided (set padding to -inf before softmax)
        if image_attention_mask is not None:
            similarity = similarity.masked_fill(~image_attention_mask, float('-inf'))
        
        # Softmax normalization
        attention_weights = F.softmax(similarity, dim=1)  # (batch_size, max_num_tokens)
        
        # Weighted sum: V_focal = sum(attention_weights * Z_grid)
        # attention_weights: (batch_size, max_num_tokens) -> (batch_size, 1, max_num_tokens)
        # Z_grid: (batch_size, max_num_tokens, hidden_size)
        # V_focal: (batch_size, hidden_size)
        V_focal = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, max_num_tokens)
            Z_grid  # (batch_size, max_num_tokens, hidden_size)
        ).squeeze(1)  # (batch_size, hidden_size)
        
        if return_activations:
            return V_focal, attention_weights
        return V_focal
    
    def _intrinsic_similarity_3d_chunked(
        self,
        h_t: torch.Tensor,
        Z_grid: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor],
        chunk_size: int
    ) -> torch.Tensor:
        """
        Memory-efficient chunked version for 3D format (batched).
        Uses online softmax algorithm to avoid storing full similarity tensor.
        
        Args:
            h_t: (batch_size, hidden_size)
            Z_grid: (batch_size, max_num_tokens, hidden_size)
            image_attention_mask: Optional (batch_size, max_num_tokens) - True for valid tokens
            chunk_size: Size of each chunk
            
        Returns:
            V_focal: (batch_size, hidden_size) - Focused visual features
        """
        batch_size = h_t.shape[0]
        max_num_tokens = Z_grid.shape[1]
        hidden_size = h_t.shape[1]
        device = h_t.device
        dtype = h_t.dtype
        
        num_chunks = (max_num_tokens + chunk_size - 1) // chunk_size
        
        # Detach visual_tokens to avoid unnecessary gradient computation (vision tower is typically frozen)
        Z_grid = Z_grid.detach()
        
        # Online softmax algorithm: avoid storing full similarity tensor
        max_scores = torch.full((batch_size,), float('-inf'), device=device, dtype=dtype)
        exp_sum = torch.zeros(batch_size, device=device, dtype=dtype)
        weighted_sum = torch.zeros(batch_size, hidden_size, device=device, dtype=dtype)
        
        # First pass: compute max scores and exp sums
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, max_num_tokens)
            chunk_tokens = Z_grid[:, chunk_start:chunk_end]  # (batch_size, chunk_size, hidden_size)
            
            # Compute similarity: h_t @ chunk_tokens^T
            chunk_similarity = torch.bmm(
                h_t.unsqueeze(1),  # (batch_size, 1, hidden_size)
                chunk_tokens.transpose(1, 2)  # (batch_size, hidden_size, chunk_size)
            ).squeeze(1)  # (batch_size, chunk_size)
            
            # Apply mask if provided (set padding to -inf before computing max)
            if image_attention_mask is not None:
                chunk_mask = image_attention_mask[:, chunk_start:chunk_end]  # (batch_size, chunk_size)
                chunk_similarity = chunk_similarity.masked_fill(~chunk_mask, float('-inf'))
            
            chunk_max = chunk_similarity.max(dim=1)[0]  # (batch_size,)
            old_max = max_scores
            max_scores = torch.maximum(max_scores, chunk_max)
            
            # Adjust previous accumulators
            if chunk_idx > 0:
                exp_adjust = torch.exp(old_max - max_scores)  # (batch_size,)
                exp_sum = exp_sum * exp_adjust
                weighted_sum = weighted_sum * exp_adjust.unsqueeze(-1)
            
            # Compute exp for current chunk
            scores_diff = chunk_similarity - max_scores.unsqueeze(-1)  # (batch_size, chunk_size)
            chunk_exp = torch.exp(scores_diff)  # (batch_size, chunk_size)
            
            # Apply mask to exp (set to 0 for padding)
            if image_attention_mask is not None:
                chunk_mask = image_attention_mask[:, chunk_start:chunk_end]
                chunk_exp = chunk_exp * chunk_mask.to(dtype)
            
            exp_sum += chunk_exp.sum(dim=1)  # (batch_size,)
            
            # Accumulate weighted sum
            weighted_chunk = torch.bmm(
                chunk_exp.unsqueeze(1),  # (batch_size, 1, chunk_size)
                chunk_tokens  # (batch_size, chunk_size, hidden_size)
            ).squeeze(1)  # (batch_size, hidden_size)
            weighted_sum += weighted_chunk
        
        # Normalize
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        V_focal = weighted_sum / (exp_sum.unsqueeze(-1) + eps)
        
        return V_focal
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        image_embeds: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor] = None,
        activation_map_save_dir: Optional[str] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        sample_idx: Optional[int] = None,
        step_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: (batch_size, hidden_size) - LLM hidden state h_t
            image_embeds: Can be either:
                - (num_image_tokens, hidden_size) - Single batch item (legacy format)
                - (batch_size, max_num_tokens, hidden_size) - Batched format (padded)
            image_attention_mask: Optional (batch_size, max_num_tokens) - True for valid tokens, False for padding
                                 Only used when image_embeds is batched format
            activation_map_save_dir: Optional directory to save activation maps
            image_grid_thw: Optional grid dimensions for visualization
            sample_idx: Optional sample index for naming saved files
            step_idx: Optional step index for multi-step inference
            
        Returns:
            v_focal: (batch_size, hidden_size) - Focused image features
        """
        import os
        debug_enabled = os.getenv("LVR_DEBUG", "0") == "1"
        
        batch_size = hidden_state.shape[0]
        
        # Check if visualization is enabled
        save_activations = (self.save_activation_maps and 
                           activation_map_save_dir is not None and 
                           sample_idx is not None)
        
        # Handle 2D format (legacy)
        if image_embeds.dim() == 2:
            num_tokens = image_embeds.shape[0]
            
            # Auto-select chunked or non-chunked based on sequence length and batch size
            # For batch_size > 1, always use chunked to save memory
            # However, if visualization is needed, prefer non-chunked to get attention weights
            effective_chunk_size = self.chunk_size
            if num_tokens > 256 or batch_size > 1:  # Lower threshold, force chunked for batch_size > 1
                # For sequences longer than 256 or batch_size > 1, use chunked processing
                use_chunked = True
                if effective_chunk_size is None or effective_chunk_size > num_tokens:
                    # Auto-select appropriate chunk size (smaller for larger batch_size)
                    if batch_size > 1:
                        # Use smaller chunks when batch_size > 1 to reduce memory pressure
                        if num_tokens > 2048:
                            effective_chunk_size = 256
                        elif num_tokens > 1024:
                            effective_chunk_size = 256
                        else:
                            effective_chunk_size = 128
                    else:
                        if num_tokens > 2048:
                            effective_chunk_size = 512
                        elif num_tokens > 1024:
                            effective_chunk_size = 512
                        else:
                            effective_chunk_size = 256
            else:
                use_chunked = False
            
            # If visualization is needed and chunked would be used, use non-chunked instead
            if save_activations and use_chunked:
                if debug_enabled:
                    print(f"[LVRHeadIntrinsicSimilarity.forward] Visualization enabled, using non-chunked 2D for attention weights")
                result = self._intrinsic_similarity_2d(
                    hidden_state, image_embeds, return_activations=True
                )
            elif use_chunked:
                v_focal = self._intrinsic_similarity_2d_chunked(
                    hidden_state, image_embeds, effective_chunk_size
                )
                result = v_focal
            else:
                result = self._intrinsic_similarity_2d(
                    hidden_state, image_embeds, return_activations=save_activations
                )
            
            if save_activations and isinstance(result, tuple):
                v_focal, activations = result
            else:
                v_focal = result
                activations = None
        
        # Handle 3D format (batched)
        elif image_embeds.dim() == 3:
            max_num_tokens = image_embeds.shape[1]
            
            # Auto-select chunked or non-chunked based on sequence length and batch size
            # For batch_size > 1, always use chunked to save memory
            # However, if visualization is needed, prefer non-chunked to get attention weights
            effective_chunk_size = self.chunk_size
            if max_num_tokens > 256 or batch_size > 1:  # Lower threshold, force chunked for batch_size > 1
                # For sequences longer than 256 or batch_size > 1, use chunked processing
                use_chunked = True
                if effective_chunk_size is None or effective_chunk_size > max_num_tokens:
                    # Auto-select appropriate chunk size (smaller for larger batch_size)
                    if batch_size > 1:
                        # Use smaller chunks when batch_size > 1 to reduce memory pressure
                        if max_num_tokens > 2048:
                            effective_chunk_size = 256
                        elif max_num_tokens > 1024:
                            effective_chunk_size = 256
                        else:
                            effective_chunk_size = 128
                    else:
                        if max_num_tokens > 2048:
                            effective_chunk_size = 512
                        elif max_num_tokens > 1024:
                            effective_chunk_size = 512
                        else:
                            effective_chunk_size = 256
            else:
                use_chunked = False
            
            # If visualization is needed and chunked would be used, use non-chunked instead
            if save_activations and use_chunked:
                if debug_enabled:
                    print(f"[LVRHeadIntrinsicSimilarity.forward] Visualization enabled, using non-chunked 3D for attention weights")
                result = self._intrinsic_similarity_3d(
                    hidden_state, image_embeds, image_attention_mask, return_activations=True
                )
            elif use_chunked:
                v_focal = self._intrinsic_similarity_3d_chunked(
                    hidden_state, image_embeds, image_attention_mask, effective_chunk_size
                )
                result = v_focal
            else:
                result = self._intrinsic_similarity_3d(
                    hidden_state, image_embeds, image_attention_mask, return_activations=save_activations
                )
            
            if save_activations and isinstance(result, tuple):
                v_focal, activations = result
            else:
                v_focal = result
                activations = None
        
        else:
            raise ValueError(f"image_embeds must be 2D or 3D, got {image_embeds.dim()}D")
        
        # Apply output normalization if enabled
        if self.use_output_norm:
            v_focal = self.output_norm(v_focal)
        
        # Save activation maps if enabled
        if save_activations and activations is not None:
            try:
                from src.model.lvr_visualization import visualize_lvr_activations
                visualize_lvr_activations(
                    activations=activations,
                    save_dir=activation_map_save_dir,
                    sample_idx=sample_idx,
                    step_idx=step_idx,
                    head_type='intrinsic_similarity',
                    image_grid_thw=image_grid_thw,
                    image_attention_mask=image_attention_mask,
                    batch_idx=0  # Visualize first batch item
                )
            except Exception as e:
                if debug_enabled:
                    print(f"[LVRHeadIntrinsicSimilarity.forward] Warning: Failed to save activation map: {e}")
        
        return v_focal


class BoxFeatureResampler(nn.Module):
    """
    Resamples variable-length bbox visual features into fixed num_queries latent tokens
    via learnable queries and cross-attention. Used as target for MSE loss (output detached).
    """
    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 8,
        num_heads: Optional[int] = None,
        vision_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        num_heads = num_heads or min(8, hidden_size // 64)
        self.num_heads = num_heads
        vision_dim = vision_dim or hidden_size
        self.vision_dim = vision_dim
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        if vision_dim != hidden_size:
            self.vision_proj = nn.Linear(vision_dim, hidden_size, bias=False)
        else:
            self.vision_proj = None
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = LayerNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        bbox_region_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            bbox_region_features: (L, N, D) L=num_bboxes, N=variable tokens per bbox, D=vision_dim or hidden_size
            key_padding_mask: (L, N) True for padding positions (ignore), False for valid. Optional.
        Returns:
            (L, num_queries, hidden_size)
        """
        L, N, D = bbox_region_features.shape
        if self.vision_proj is not None:
            bbox_region_features = self.vision_proj(bbox_region_features)
        q = self.queries.expand(L, -1, -1)
        attn_out, _ = self.cross_attn(
            q,
            bbox_region_features,
            bbox_region_features,
            key_padding_mask=key_padding_mask,
        )
        return self.output_norm(attn_out)


class LVRBboxMLP(nn.Module):
    """
    Thin wrapper around BoxFeatureResampler for backward compatibility with
    use_fixed_num_lvr_tokens + lvr_bbox_mlp. Maps bbox image token features to
    fixed_num_lvr_tokens vectors via cross-attention resampling.
    """
    def __init__(self, hidden_size: int, fixed_num_lvr_tokens: int = 16, vision_dim: Optional[int] = None):
        super().__init__()
        self.resampler = BoxFeatureResampler(
            hidden_size=hidden_size,
            num_queries=fixed_num_lvr_tokens,
            vision_dim=vision_dim,
        )
        self.fixed_num_lvr_tokens = fixed_num_lvr_tokens

    def forward(
        self,
        bbox_region_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.resampler(bbox_region_features, key_padding_mask=key_padding_mask)


# Optional diffusers for DiT reconstruction head
try:
    from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    AutoencoderKL = None
    DDPMScheduler = None
    DDIMScheduler = None


class DiTReconstructionHead(nn.Module):
    """
    DiT-XL-2 pixel reconstruction head.
    Train: bbox crop -> VAE encode -> add noise -> DiT denoise (conditioned on LLM 8 tokens) -> decode -> MSE loss.
    Infer: LLM 8 tokens -> DiT denoise from noise -> decode -> image.
    """
    patch_size = 2
    in_channels = 4
    out_channels = 8  # DiT-XL-2 with learn_sigma=True: out_channels = in_channels * 2
    hidden_size = 1152  # DiT-XL-2
    depth = 28  # DiT-XL-2
    num_heads = 16  # DiT-XL-2
    num_condition_tokens = 8

    def __init__(
        self,
        llm_hidden_size: int = 3584,
        dit_hidden_size: int = 1152,  # DiT-XL-2 default
        vae_repo: str = "stabilityai/sd-vae-ft-mse",
        dit_pretrained_path: Optional[str] = None,
        crop_size: int = 128,
    ):
        super().__init__()
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("DiTReconstructionHead requires diffusers. Install with: pip install diffusers")
        self.llm_hidden_size = llm_hidden_size
        self.dit_hidden_size = dit_hidden_size
        self.crop_size = crop_size
        self.latent_size = crop_size // 8  # VAE 8x downsampling: 128->16, 256->32
        self.n_patch = self.latent_size // self.patch_size
        self.num_patches = self.n_patch * self.n_patch
        # Use out_channels for final_layer to match official DiT-XL-2 (learn_sigma=True)
        # During training we only use the first 4 channels (noise prediction)
        self.patch_dim = self.patch_size * self.patch_size * self.out_channels  # 2*2*8 = 32

        # Defer VAE loading to avoid DeepSpeed ZeRO-3 meta tensor issues
        # ZeRO-3 intercepts nn.Module.__init__ and creates meta tensors, which breaks from_pretrained
        # Solution: Load VAE lazily on first forward pass, outside of ZeRO-3 init context
        self._load_vae_outside_zero3(vae_repo)  # Sets up lazy loading

        # Condition adapter: maps LLM tokens to DiT hidden size
        self.condition_proj = nn.Linear(llm_hidden_size, dit_hidden_size)
        
        # x_embedder: patch embedding using Conv2d (matching official DiT-XL-2)
        # Input: (B, 4, H, W) latent -> Output: (B, hidden_size, H//patch_size, W//patch_size)
        # Then flattened to (B, num_patches, hidden_size)
        self.x_embedder = nn.Conv2d(
            self.in_channels, 
            dit_hidden_size, 
            kernel_size=self.patch_size, 
            stride=self.patch_size,
            bias=True
        )
        
        # Legacy proj_in for backward compatibility (will be replaced by x_embedder in forward)
        # Keep for now but will use x_embedder instead
        self.proj_in = None  # Deprecated, use x_embedder instead
        # Timestep embedding: official DiT-XL-2 uses fixed 256-dim sinusoidal embedding
        # then maps to hidden_size via MLP: 256 -> hidden -> hidden (NOT 256 -> hidden*4 -> hidden)
        self.t_embed_dim = 256  # Fixed dimension matching official DiT-XL-2
        self.t_embed_mlp = nn.Sequential(
            nn.Linear(self.t_embed_dim, dit_hidden_size),
            nn.SiLU(),
            nn.Linear(dit_hidden_size, dit_hidden_size),
        )

        self.blocks = nn.ModuleList([
            _DiTBlock(dit_hidden_size, num_heads=self.num_heads)
            for _ in range(self.depth)
        ])
        self.norm = LayerNorm(dit_hidden_size, eps=1e-6)
        
        # final_layer: output projection back to patch space
        self.final_layer = nn.Linear(dit_hidden_size, self.patch_dim)
        # Legacy name for backward compatibility
        self.proj_out = self.final_layer

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            beta_start=0.0001,
            beta_end=0.02,
            prediction_type="epsilon",
            clip_sample=False,
        )

        # DDIMScheduler for faster inference sampling.
        # IMPORTANT: Use from_config to ensure alpha/beta values match training DDPMScheduler exactly.
        # Mismatch causes DDIM sampling to produce garbage (one-step denoise works but generate fails).
        # timestep_spacing="trailing" per "Common Diffusion Noise Schedules and Sample Steps are Flawed"
        self.inference_scheduler = DDIMScheduler.from_config(
            self.noise_scheduler.config,
            timestep_spacing="trailing",
        )

        # Defer pretrained weight loading to outside ZeRO-3 init context.
        # ZeRO-3 creates meta tensors during __init__, so all parameter sizes are 0.
        # We must load weights after model initialization is complete.
        self._dit_pretrained_path = dit_pretrained_path
        self._dit_weights_loaded = False

    def _load_vae_outside_zero3(self, vae_repo: str):
        """Load VAE while bypassing DeepSpeed ZeRO-3's nn.Module.__init__ wrapper.
        
        ZeRO-3 wraps nn.Module.__init__ to create meta tensors and then tries to move
        them to device, which fails because meta tensors have no data.
        
        Solution: Don't load VAE in __init__. Instead, defer loading to first forward pass
        when DeepSpeed context is no longer active. Store vae_repo for later use.
        """
        # We can't load VAE here because we're inside DeepSpeed's ZeRO-3 init context.
        # Return None and load lazily on first use.
        self._vae_repo = vae_repo
        self._vae_loaded = False
        return None
    
    def _ensure_vae_loaded(self):
        """Lazy-load VAE on first use, outside of DeepSpeed init context."""
        if self._vae_loaded:
            return
        
        # Now we're outside the __init__ context, safe to load
        print(f"[DiTReconstructionHead] Lazy-loading VAE from {self._vae_repo}...", flush=True)
        vae = AutoencoderKL.from_pretrained(self._vae_repo)
        for p in vae.parameters():
            p.requires_grad = False
        
        # Move VAE to same device as model parameters
        # In DeepSpeed ZeRO-3, parameters might be on meta device, so check for a valid device
        try:
            device = next(p.device for p in self.parameters() if p.device.type != 'meta')
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        vae = vae.to(device)
        
        # CRITICAL: Use object.__setattr__ to bypass nn.Module's submodule registration
        # This prevents DeepSpeed ZeRO-3 from trying to manage VAE parameters
        # (VAE is frozen and doesn't need gradient sync)
        object.__setattr__(self, '_vae_model', vae)
        
        print(f"[DiTReconstructionHead] VAE loaded and moved to {device}", flush=True)
        self._vae_loaded = True
    
    @property
    def vae(self):
        """Access VAE model (stored outside nn.Module to avoid DeepSpeed management)."""
        return getattr(self, '_vae_model', None)

    def _ensure_dit_weights_loaded(self):
        """Lazy-load pretrained DiT weights on first forward, outside ZeRO-3 init context."""
        if self._dit_weights_loaded:
            return
        self._dit_weights_loaded = True  # Set early to avoid re-entry
        
        path = self._dit_pretrained_path
        if not path or path.strip() == "":
            return
        
        import os
        if not os.path.exists(path):
            print(f"[DiTReconstructionHead] Pretrained weights not found: {path}, training from scratch", flush=True)
            return
        
        self._load_pretrained_dit(path)

    def _load_pretrained_dit(self, path: str):
        """
        Load pretrained DiT-XL-2 weights from .pt checkpoint.
        
        Official DiT-XL-2 key structure:
          x_embedder.proj.weight/bias        -> Conv2d patch embedding
          t_embedder.mlp.0.weight/bias       -> timestep MLP layer 1 (256 -> hidden*4)
          t_embedder.mlp.2.weight/bias       -> timestep MLP layer 2 (hidden*4 -> hidden)
          y_embedder.embedding_table.weight   -> class embedding (SKIP)
          pos_embed                           -> positional embedding (SKIP)
          blocks.N.attn.qkv.weight/bias      -> fused QKV  -> our attn.in_proj_weight/bias
          blocks.N.attn.proj.weight/bias      -> attn out   -> our attn.out_proj.weight/bias
          blocks.N.mlp.fc1.weight/bias        -> MLP layer1 -> our mlp.0.weight/bias
          blocks.N.mlp.fc2.weight/bias        -> MLP layer2 -> our mlp.2.weight/bias
          blocks.N.adaLN_modulation.*         -> adaptive LN (SKIP, no equivalent)
          final_layer.adaLN_modulation.*      -> adaptive LN (SKIP)
          final_layer.linear.weight/bias      -> output proj -> our final_layer.weight/bias
        """
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # Handle different checkpoint formats (ema key, model key, raw state_dict)
            if isinstance(checkpoint, dict):
                if 'ema' in checkpoint:
                    state_dict = checkpoint['ema']
                    print(f"[DiTReconstructionHead] Using EMA weights from checkpoint", flush=True)
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            loaded_count = 0
            skipped_keys = []
            
            def _safe_copy(target_param, source_tensor, key_name):
                """Copy source tensor to target parameter with shape validation."""
                nonlocal loaded_count
                if target_param.shape == source_tensor.shape:
                    target_param.data.copy_(source_tensor)
                    loaded_count += 1
                    return True
                else:
                    print(f"[DiTReconstructionHead] Shape mismatch for {key_name}: "
                          f"ours={target_param.shape}, pretrained={source_tensor.shape}", flush=True)
                    return False
            
            # ---- x_embedder (patch embedding Conv2d) ----
            # Official key: x_embedder.proj.weight / x_embedder.proj.bias
            for src_prefix in ['x_embedder.proj', 'x_embedder']:
                w_key = f'{src_prefix}.weight'
                b_key = f'{src_prefix}.bias'
                if w_key in state_dict:
                    _safe_copy(self.x_embedder.weight, state_dict[w_key], w_key)
                    if b_key in state_dict:
                        _safe_copy(self.x_embedder.bias, state_dict[b_key], b_key)
                    break
            
            # ---- t_embedder (timestep MLP: 256 -> hidden*4 -> hidden) ----
            # Official: t_embedder.mlp.0 / t_embedder.mlp.2
            t_map = {
                't_embedder.mlp.0.weight': (self.t_embed_mlp, '0', 'weight'),
                't_embedder.mlp.0.bias':   (self.t_embed_mlp, '0', 'bias'),
                't_embedder.mlp.2.weight': (self.t_embed_mlp, '2', 'weight'),
                't_embedder.mlp.2.bias':   (self.t_embed_mlp, '2', 'bias'),
            }
            for src_key, (container, idx, attr) in t_map.items():
                if src_key in state_dict:
                    target = getattr(container[int(idx)], attr)
                    _safe_copy(target, state_dict[src_key], src_key)
            
            # ---- Transformer blocks ----
            # Official DiT block key mapping:
            #   attn.qkv.{w,b}           -> our attn.in_proj_{weight,bias}
            #   attn.proj.{w,b}          -> our attn.out_proj.{weight,bias}
            #   mlp.fc1.{w,b}            -> our mlp.0.{weight,bias}  (== mlp[0])
            #   mlp.fc2.{w,b}            -> our mlp.2.{weight,bias}  (== mlp[2])
            #   adaLN_modulation.*        -> SKIP (no equivalent in our cross-attn design)
            block_loaded = 0
            block_skipped = 0
            for block_idx, block in enumerate(self.blocks):
                prefix = f'blocks.{block_idx}'
                
                # Self-attention: qkv -> in_proj, proj -> out_proj
                for src_attr, dst_obj, dst_attr in [
                    (f'{prefix}.attn.qkv.weight', block.attn, 'in_proj_weight'),
                    (f'{prefix}.attn.qkv.bias',   block.attn, 'in_proj_bias'),
                    (f'{prefix}.attn.proj.weight', block.attn.out_proj, 'weight'),
                    (f'{prefix}.attn.proj.bias',   block.attn.out_proj, 'bias'),
                    # MLP: fc1 -> mlp[0], fc2 -> mlp[2]
                    (f'{prefix}.mlp.fc1.weight', block.mlp[0], 'weight'),
                    (f'{prefix}.mlp.fc1.bias',   block.mlp[0], 'bias'),
                    (f'{prefix}.mlp.fc2.weight', block.mlp[2], 'weight'),
                    (f'{prefix}.mlp.fc2.bias',   block.mlp[2], 'bias'),
                ]:
                    if src_attr in state_dict:
                        target = getattr(dst_obj, dst_attr)
                        if _safe_copy(target, state_dict[src_attr], src_attr):
                            block_loaded += 1
                        else:
                            block_skipped += 1
                
                # Skip: adaLN_modulation, norm (official uses adaLN, not separate LayerNorm)
                for k in state_dict:
                    if k.startswith(prefix) and ('adaLN' in k):
                        skipped_keys.append(k)
            
            # ---- final_layer output projection ----
            # Official: final_layer.linear.weight/bias
            for src_key, target_param in [
                ('final_layer.linear.weight', self.final_layer.weight),
                ('final_layer.linear.bias',   self.final_layer.bias),
            ]:
                if src_key in state_dict:
                    _safe_copy(target_param, state_dict[src_key], src_key)
            
            # ---- Skip: y_embedder, pos_embed, final_layer.adaLN_modulation ----
            for k in state_dict:
                if any(skip in k for skip in ['y_embedder', 'pos_embed', 'adaLN']):
                    if k not in skipped_keys:
                        skipped_keys.append(k)
            
            total = loaded_count + block_loaded
            print(f"[DiTReconstructionHead] Loaded {total} params from {path} "
                  f"(global: {loaded_count}, blocks: {block_loaded})", flush=True)
            
            if skipped_keys:
                print(f"[DiTReconstructionHead] Skipped {len(skipped_keys)} keys "
                      f"(y_embedder/pos_embed/adaLN): {skipped_keys[:5]}...", flush=True)
                
        except Exception as e:
            import traceback
            print(f"[DiTReconstructionHead] Error loading weights from {path}: {e}", flush=True)
            traceback.print_exc()
            print(f"[DiTReconstructionHead] Training from scratch", flush=True)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        # Dynamically calculate num_patches based on actual latent shape
        # This handles cases where VAE output size differs from expected crop_size//8
        num_patches_h = H // p
        num_patches_w = W // p
        num_patches = num_patches_h * num_patches_w
        # Verify channel count matches expected in_channels
        if C != self.in_channels:
            raise ValueError(
                f"VAE latent channels {C} does not match expected in_channels {self.in_channels}. "
                f"Latent shape: {x.shape}, crop_size: {self.crop_size}, "
                f"expected latent_size: {self.latent_size}, expected channels: {self.in_channels}"
            )
        # Verify spatial dimensions are divisible by patch_size
        if H % p != 0 or W % p != 0:
            raise ValueError(
                f"Latent spatial dimensions ({H}, {W}) must be divisible by patch_size {p}. "
                f"Latent shape: {x.shape}, crop_size: {self.crop_size}"
            )
        # Patchify: (B, in_channels, H, W) -> (B, num_patches, patch_size^2 * in_channels)
        # Note: This method is for legacy/fallback use. forward() uses x_embedder (Conv2d) instead.
        x = x.reshape(B, C, num_patches_h, p, num_patches_w, p)
        # Permute: (B, C, h, p, w, p) -> (B, h, w, C, p, p) -> (B, h*w, C*p*p)
        patch_dim_input = p * p * C  # For input latent: 2*2*4 = 16
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, num_patches, patch_dim_input)
        return x

    def _unpatchify(self, x: torch.Tensor, H_patch: Optional[int] = None, W_patch: Optional[int] = None) -> torch.Tensor:
        """
        Convert patch tokens back to spatial latent format.
        
        Args:
            x: (B, num_patches, patch_dim) patch tokens
            H_patch, W_patch: Optional spatial dimensions. If None, infer from num_patches assuming square.
        
        Returns:
            (B, out_channels, H, W) spatial latent (full output with sigma if learn_sigma=True)
        """
        B, N, D = x.shape
        p = self.patch_size
        
        # Verify patch_dim matches
        if D != self.patch_dim:
            raise ValueError(
                f"Patch dimension {D} does not match expected patch_dim {self.patch_dim}"
            )
        
        # Determine spatial dimensions
        if H_patch is not None and W_patch is not None:
            h, w = H_patch, W_patch
            if h * w != N:
                raise ValueError(
                    f"num_patches {N} does not match H_patch * W_patch = {h} * {w} = {h * w}"
                )
        else:
            # Infer from num_patches assuming square
            n_patch = int(N ** 0.5)
            if n_patch * n_patch != N:
                raise ValueError(
                    f"num_patches {N} is not a perfect square, cannot determine spatial dimensions. "
                    f"Please provide H_patch and W_patch explicitly."
                )
            h = w = n_patch
        
        # Reshape: (B, num_patches, patch_dim) -> (B, h, w, p, p, out_channels)
        # patch_dim = p * p * out_channels
        x = x.reshape(B, h, w, p, p, self.out_channels)
        # Permute and reshape: -> (B, out_channels, h*p, w*p)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_channels, h * p, w * p)
        return x

    def forward(
        self,
        cropped_images: torch.Tensor,
        llm_condition_tokens: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Args:
            cropped_images: (B, 3, crop_size, crop_size) RGB, normalized in [-1, 1]
            llm_condition_tokens: (B, 8, llm_hidden_size)
            timesteps: (B,) or None; if None, random for training
        Returns:
            loss (training) or pred_images (inference)
        """
        # Lazy-load VAE and pretrained DiT weights on first use
        # (deferred from __init__ to avoid DeepSpeed ZeRO-3 meta tensor issues)
        self._ensure_vae_loaded()
        self._ensure_dit_weights_loaded()
        
        B = cropped_images.shape[0]
        device = cropped_images.device
        dtype = cropped_images.dtype

        # VAE is in float32, convert input to float32 for encoding
        with torch.no_grad():
            cropped_images_f32 = cropped_images.float()  # Convert bf16 -> float32
            latent = self.vae.encode(cropped_images_f32).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
            latent = latent.to(dtype)  # Convert back to original dtype for DiT processing
        
        # Debug: Check latent shape matches expectations
        # VAE should output (B, 4, H, W) where H=W=crop_size//8
        if latent.shape[1] != self.in_channels:
            # Try to handle different VAE output formats
            # Some VAEs might output (B, C, H, W) with C != 4
            # If spatial dims match but channels don't, we need to adapt
            B_latent, C_latent, H_latent, W_latent = latent.shape
            expected_H = self.latent_size
            expected_W = self.latent_size
            
            if H_latent == expected_H and W_latent == expected_W and C_latent != self.in_channels:
                # Spatial dimensions match but channels don't - this is an error
                raise ValueError(
                    f"VAE latent channels mismatch: got {C_latent}, expected {self.in_channels}. "
                    f"Latent shape: {latent.shape}, crop_size: {self.crop_size}, "
                    f"expected latent_size: {self.latent_size}, expected shape: (B, {self.in_channels}, {expected_H}, {expected_W})"
                )
            elif H_latent != expected_H or W_latent != expected_W:
                # Spatial dimensions don't match - update expected dimensions
                import warnings
                warnings.warn(
                    f"VAE latent spatial dimensions ({H_latent}, {W_latent}) differ from expected ({expected_H}, {expected_W}). "
                    f"Latent shape: {latent.shape}, crop_size: {self.crop_size}. "
                    f"This may cause issues with patchify/unpatchify operations."
                )

        if self.training and timesteps is None:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
        elif not self.training and timesteps is None:
            timesteps = torch.zeros(B, device=device, dtype=torch.long)

        noise = torch.randn_like(latent, device=device, dtype=dtype)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)

        cond = self.condition_proj(llm_condition_tokens)  # (B, 8, dit_hidden_size)
        t_emb = self._timestep_embed(timesteps, device, dtype)  # (B, dit_hidden_size)
        
        # Use x_embedder (Conv2d) for patch embedding, matching official DiT-XL-2
        # Input: (B, 4, H, W) -> Output: (B, hidden_size, H//patch_size, W//patch_size)
        x = self.x_embedder(noisy_latent)  # (B, hidden_size, H_patch, W_patch)
        B_x, C_x, H_patch, W_patch = x.shape
        # Flatten spatial dimensions: (B, hidden_size, H_patch, W_patch) -> (B, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        
        # Add timestep embedding
        x = x + t_emb.unsqueeze(1)  # (B, num_patches, hidden_size)

        for blk in self.blocks:
            x = blk(x, cond)

        x = self.norm(x)
        noise_pred_patch = self.final_layer(x)  # (B, num_patches, patch_dim)
        
        # Unpatchify: convert back to spatial format
        # (B, num_patches, patch_dim) -> (B, out_channels, H, W)
        noise_pred_full = self._unpatchify(noise_pred_patch, H_patch, W_patch)
        
        # Extract noise prediction (first in_channels=4), ignore sigma (last 4 channels if learn_sigma)
        noise_pred = noise_pred_full[:, :self.in_channels, :, :]  # (B, 4, H, W)

        if self.training:
            # Latent Diffusion loss: predict noise in latent space (no VAE decode needed)
            loss = F.mse_loss(noise_pred, noise)
            return (loss, {"noise_pred": noise_pred}) if return_dict else loss
        else:
            # Non-training forward: return noise_pred directly; use generate() for actual inference
            return (noise_pred, {}) if return_dict else noise_pred

    @torch.no_grad()
    def generate(
        self,
        llm_condition_tokens: torch.Tensor,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate images from pure noise conditioned on LLM tokens (DDIM sampling).
        
        Args:
            llm_condition_tokens: (B, 8, llm_hidden_size) - LLM hidden states as condition
            num_inference_steps: Number of denoising steps (default: 20)
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
            return_intermediate: If True, return intermediate denoising steps
            
        Returns:
            pred_images: (B, 3, crop_size, crop_size) Generated images in [-1, 1]
            intermediates: (optional) List of intermediate images
        """
        self._ensure_vae_loaded()
        self._ensure_dit_weights_loaded()
        
        B = llm_condition_tokens.shape[0]
        device = llm_condition_tokens.device
        dtype = llm_condition_tokens.dtype
        
        # Project condition tokens
        cond = self.condition_proj(llm_condition_tokens)  # (B, 8, dit_hidden_size)
        
        # Start from pure Gaussian noise
        latent_shape = (B, self.in_channels, self.latent_size, self.latent_size)
        latent = torch.randn(latent_shape, device=device, dtype=dtype)
        
        # Set up DDIM scheduler for inference
        self.inference_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inference_scheduler.timesteps
        
        intermediates = [] if return_intermediate else None
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)
            
            # Get timestep embedding
            t_emb = self._timestep_embed(t_batch, device, dtype)
            
            # Use x_embedder for patch embedding
            x = self.x_embedder(latent)  # (B, hidden_size, H_patch, W_patch)
            B_x, C_x, H_patch, W_patch = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
            x = x + t_emb.unsqueeze(1)
            
            # DiT blocks with cross-attention to condition
            for blk in self.blocks:
                x = blk(x, cond)
            
            x = self.norm(x)
            noise_pred_patch = self.final_layer(x)  # (B, num_patches, patch_dim)
            noise_pred_full = self._unpatchify(noise_pred_patch, H_patch, W_patch)
            # Extract noise prediction (first 4 channels), ignore sigma
            noise_pred = noise_pred_full[:, :self.in_channels, :, :]
            
            # DDIM scheduler step: update latents
            latent = self.inference_scheduler.step(noise_pred, t, latent, return_dict=False)[0]
            
            # Save intermediate
            if return_intermediate and (i % max(num_inference_steps // 10, 1) == 0 or i == len(timesteps) - 1):
                intermediate_f32 = latent.float() / self.vae.config.scaling_factor
                intermediate_img = self.vae.decode(intermediate_f32).sample.to(dtype)
                intermediates.append(intermediate_img.clamp(-1, 1))
        
        # Decode final latent to image: undo VAE scaling before decoding
        latent_f32 = latent.float() / self.vae.config.scaling_factor
        pred_images = self.vae.decode(latent_f32).sample.to(dtype)
        pred_images = pred_images.clamp(-1, 1)
        
        if return_intermediate:
            return pred_images, intermediates
        return pred_images

    def _timestep_embed(self, t: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Use fixed 256-dim embedding matching official DiT-XL-2
        emb = _get_sinusoidal_timestep_embedding(t, self.t_embed_dim, device, dtype)
        emb = self.t_embed_mlp(emb)
        return emb


def _get_sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device, dtype=dtype) * -emb)
    emb = t[:, None].to(dtype) * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class _DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 6):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.0, batch_first=True)
        self.norm2 = LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.0, batch_first=True)
        self.norm3 = LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.cross_attn(self.norm2(x), cond, cond, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x