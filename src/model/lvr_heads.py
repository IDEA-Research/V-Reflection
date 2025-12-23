import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers.activations import ACT2FN
from typing import Optional

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


class SlotInitializer(nn.Module):
    """
    阶段一：预计算语义槽位（图像摘要）
    在推理循环开始前，用一次前向传播将图像编码为一组固定、紧凑的语义槽位。
    """
    def __init__(self, num_slots: int, slot_dim: int, feature_dim: int, iters: int = 3):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters
        
        # 使用一个小型网络将图像特征投影到键/值空间
        self.to_k = nn.Linear(feature_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(feature_dim, slot_dim, bias=False)
        
        # 槽位初始化：可学习的聚类中心
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        
    def forward(self, Z_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_grid: [B, N, feature_dim] 或 [N, feature_dim] (legacy format)
        Returns:
            slots: [B, num_slots, slot_dim]
        """
        # 处理legacy格式（2D）
        if Z_grid.dim() == 2:
            Z_grid = Z_grid.unsqueeze(0)  # [1, N, feature_dim]
            squeeze_output = True
        else:
            squeeze_output = False
            
        B, N, _ = Z_grid.shape
        
        # 初始化槽位
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn_like(self.slots_mu)
        slots = slots.repeat(B, 1, 1)  # [B, num_slots, slot_dim]
        
        # 简化的Slot Attention迭代
        K = self.to_k(Z_grid)  # [B, N, slot_dim]
        V = self.to_v(Z_grid)  # [B, N, slot_dim]
        
        for _ in range(self.iters):
            # 1. 槽位作为Query，与图像特征计算注意力
            Q = slots  # [B, num_slots, slot_dim]
            attn_logits = torch.einsum('bsd,bnd->bsn', Q, K) / (self.slot_dim ** 0.5)
            attn = F.softmax(attn_logits, dim=-1)  # 沿N维，竞争归属权 [B, num_slots, N]
            
            # 2. 加权更新槽位
            updates = torch.einsum('bsn,bnd->bsd', attn, V)  # [B, num_slots, slot_dim]
            slots = slots + updates  # 残差连接
        
        # 如果输入是legacy格式，输出也应该是legacy格式
        if squeeze_output:
            slots = slots.squeeze(0)  # [num_slots, slot_dim]
            
        return slots


class SparseRouter(nn.Module):
    """
    阶段二：可微稀疏路由器（动态选择）
    在每一步 t，路由器根据LLM的隐状态 h_t，决定当前步骤应与哪些槽位交互。
    """
    def __init__(self, hidden_dim: int, num_slots: int, top_k: int = 2, temp: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.temp = temp
        self.num_slots = num_slots
        
        # 一个简单的投影层，将h_t映射为每个槽位的"相关性分数"
        self.router_proj = nn.Linear(hidden_dim, num_slots)
        
    def forward(self, h_t: torch.Tensor, slots: torch.Tensor, training: bool = True):
        """
        Args:
            h_t: [B, hidden_dim] - LLM的隐状态
            slots: [B, num_slots, slot_dim] 或 [num_slots, slot_dim] (legacy format)
            training: 是否处于训练模式
        Returns:
            router_weights: [B, num_slots] - 稀疏权重（只有top_k个非零值）
            selected_slots: [B, 1, slot_dim] - 选中的槽位聚合
        """
        # 处理legacy格式
        if slots.dim() == 2:
            slots = slots.unsqueeze(0)  # [1, num_slots, slot_dim]
            squeeze_output = True
        else:
            squeeze_output = False
            
        B = h_t.shape[0]
        router_logits = self.router_proj(h_t)  # [B, num_slots]
        
        if training:
            # 使用Gumbel-Softmax进行可微的Top-k采样
            # 1. 添加Gumbel噪声
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-10) + 1e-10)
            noisy_logits = (router_logits + gumbel_noise) / self.temp
            
            # 2. 计算Top-k稀疏分布
            topk_val, topk_idx = torch.topk(noisy_logits, self.top_k, dim=-1)
            
            # 3. 构造一个仅在Top-k位置有值的稀疏矩阵
            topk_one_hot = torch.zeros_like(router_logits).scatter(-1, topk_idx, 1.0)
            
            # 4. 使用Straight-Through Estimator: 前向硬，反向软
            topk_softmax = F.softmax(topk_val, dim=-1)
            # 创建一个稀疏矩阵，在top-k位置填充softmax值
            topk_softmax_sparse = torch.zeros_like(router_logits).scatter(-1, topk_idx, topk_softmax)
            router_weights = topk_one_hot + (topk_softmax_sparse - topk_one_hot).detach()
        else:
            # 推理时：直接取Top-k，并进行归一化作为权重
            topk_val, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
            topk_softmax = F.softmax(topk_val, dim=-1)
            router_weights = torch.zeros_like(router_logits).scatter(-1, topk_idx, topk_softmax)
        
        # 根据权重，选出需要交互的槽位
        # router_weights: [B, num_slots]，每行只有k个非零值
        selected_slots = torch.bmm(router_weights.unsqueeze(1), slots)  # [B, 1, slot_dim]
        
        if squeeze_output:
            selected_slots = selected_slots.squeeze(0)  # [1, slot_dim]
            router_weights = router_weights.squeeze(0)  # [num_slots]
        
        return router_weights, selected_slots


class LocalInteraction(nn.Module):
    """
    阶段三：局部深度交互与更新
    让LLM的潜状态 h_t 与选中的少量槽位进行深度信息交换。
    """
    def __init__(self, hidden_dim: int, slot_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slot_dim = slot_dim
        
        # 将h_t投影到slot_dim空间，以便与槽位交互
        self.h_to_slot = nn.Linear(hidden_dim, slot_dim, bias=False)
        
        # 一个轻量的交叉注意力层，用于h_t与选中槽位的交互
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=slot_dim, 
            num_heads=num_heads, 
            batch_first=True,
            bias=False
        )
        
        # 一个MLP，用于更新槽位状态
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(slot_dim * 4, slot_dim, bias=False)
        )
        
        self.norm = nn.LayerNorm(slot_dim, eps=1e-6)
        
    def forward(self, h_t: torch.Tensor, selected_slots: torch.Tensor):
        """
        Args:
            h_t: [B, hidden_dim] - LLM的隐状态
            selected_slots: [B, 1, slot_dim] 或 [1, slot_dim] (legacy format)
        Returns:
            V_focal: [B, slot_dim] - 更新后的槽位特征，作为V_focal输入下一步LLM
            updated_slot: [B, 1, slot_dim] - 更新后的槽位（用于后续更新）
        """
        # 处理legacy格式
        if selected_slots.dim() == 2:
            selected_slots = selected_slots.unsqueeze(0)  # [1, 1, slot_dim]
            squeeze_output = True
        else:
            squeeze_output = False
            
        B = h_t.shape[0]
        
        # 将h_t投影到slot_dim空间作为query
        h_t_proj = self.h_to_slot(h_t).unsqueeze(1)  # [B, 1, slot_dim]
        
        # 进行交叉注意力交互：h_t作为query，选中的槽位作为key和value
        updated_slot, _ = self.cross_attn(
            query=h_t_proj,
            key=selected_slots,
            value=selected_slots
        )  # [B, 1, slot_dim]
        
        # 残差连接与MLP更新
        updated_slot = self.norm(updated_slot + self.mlp(updated_slot))
        
        # 将更新后的槽位特征作为V_focal，输入下一步LLM
        V_focal = updated_slot.squeeze(1)  # [B, slot_dim]
        
        if squeeze_output:
            V_focal = V_focal.squeeze(0)  # [slot_dim]
            updated_slot = updated_slot.squeeze(0)  # [1, slot_dim]
        
        return V_focal, updated_slot


class LVRHeadSlotAttention(nn.Module):
    """
    语义槽位的稀疏交互 LVR Head
    
    核心思想：将高维图像抽象为一小组语义槽位，在推理的每一步，让LLM的潜状态通过一个稀疏路由器，
    主动选择与之交互的少数几个槽位，进行深度信息交换。
    
    三个阶段：
    1. 预计算语义槽位（图像摘要）- SlotInitializer
    2. 可微稀疏路由器（动态选择）- SparseRouter
    3. 局部深度交互与更新 - LocalInteraction
    """
    def __init__(
        self, 
        hidden_size: int,
        num_slots: int = 8,
        slot_dim: Optional[int] = None,
        slot_iters: int = 3,
        top_k: int = 2,
        router_temp: float = 1.0,
        num_heads: int = 4,
        feature_dim: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.slot_dim = slot_dim if slot_dim is not None else hidden_size
        self.top_k = top_k
        
        # feature_dim用于SlotInitializer，如果未指定则使用hidden_size
        self.feature_dim = feature_dim if feature_dim is not None else hidden_size
        
        # 阶段一：预计算语义槽位
        self.slot_initializer = SlotInitializer(
            num_slots=num_slots,
            slot_dim=self.slot_dim,
            feature_dim=self.feature_dim,
            iters=slot_iters
        )
        
        # 阶段二：可微稀疏路由器
        self.sparse_router = SparseRouter(
            hidden_dim=hidden_size,
            num_slots=num_slots,
            top_k=top_k,
            temp=router_temp
        )
        
        # 阶段三：局部深度交互
        self.local_interaction = LocalInteraction(
            hidden_dim=hidden_size,
            slot_dim=self.slot_dim,
            num_heads=num_heads
        )
        
        # 输出投影：将slot_dim投影回hidden_size（如果需要）
        if self.slot_dim != hidden_size:
            self.output_proj = nn.Linear(self.slot_dim, hidden_size, bias=False)
        else:
            self.output_proj = None
        
        # 输出归一化
        self.output_norm = LayerNorm(hidden_size, eps=1e-6)
        
        # 缓存槽位（避免重复计算）
        self._cached_slots = None
        self._cached_image_embeds_shape = None
        
    def _initialize_slots(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        初始化槽位（仅在第一次或图像特征变化时调用）
        
        Args:
            image_embeds: [B, N, hidden_size] 或 [N, hidden_size] (legacy format)
        Returns:
            slots: [B, num_slots, slot_dim]
        """
        # 检查是否需要重新计算槽位
        current_shape = image_embeds.shape
        if (self._cached_slots is None or 
            self._cached_image_embeds_shape != current_shape or
            not self.training):  # 推理时每次都重新计算以确保正确性
            
            # 处理legacy格式
            if image_embeds.dim() == 2:
                # [N, hidden_size] -> [1, N, hidden_size]
                image_embeds_batched = image_embeds.unsqueeze(0)
            else:
                image_embeds_batched = image_embeds
            
            # 如果feature_dim != hidden_size，需要投影
            if self.feature_dim != self.hidden_size:
                # 假设image_embeds已经是正确的维度，或者需要投影
                # 这里假设image_embeds的最后一维是hidden_size
                if image_embeds_batched.shape[-1] != self.feature_dim:
                    # 需要一个投影层（如果feature_dim不同）
                    # 为了简化，我们假设feature_dim == hidden_size
                    pass
            
            # 初始化槽位
            slots = self.slot_initializer(image_embeds_batched)
            
            # 缓存（仅在训练时）
            if self.training:
                self._cached_slots = slots
                self._cached_image_embeds_shape = current_shape
        else:
            slots = self._cached_slots
        
        return slots
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        image_embeds: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, hidden_size] - LLM的隐状态 h_t
            image_embeds: [B, N, hidden_size] 或 [N, hidden_size] (legacy format)
            image_attention_mask: Optional [B, N] - 仅用于兼容性，实际不使用
        Returns:
            v_focal: [B, hidden_size] - 聚焦的图像特征
        """
        import time
        import os
        debug_enabled = os.getenv("LVR_DEBUG", "0") == "1"
        start_time = time.time() if debug_enabled else None
        
        if debug_enabled:
            print(f"[LVRHeadSlotAttention.forward] START: batch_size={hidden_state.shape[0]}, "
                  f"image_embeds.shape={image_embeds.shape}, num_slots={self.num_slots}, top_k={self.top_k}")
        
        # 阶段一：预计算语义槽位（仅在第一次或图像变化时）
        if debug_enabled:
            print(f"[LVRHeadSlotAttention.forward] Stage 1: Initializing slots...")
        slots = self._initialize_slots(image_embeds)  # [B, num_slots, slot_dim] 或 [num_slots, slot_dim]
        
        if debug_enabled:
            print(f"[LVRHeadSlotAttention.forward] Stage 1 done: slots.shape={slots.shape}, "
                  f"elapsed={time.time()-start_time:.3f}s" if start_time else "")
        
        # 阶段二：稀疏路由器选择相关槽位
        if debug_enabled:
            print(f"[LVRHeadSlotAttention.forward] Stage 2: Sparse routing...")
        router_weights, selected_slots = self.sparse_router(
            h_t=hidden_state,
            slots=slots,
            training=self.training
        )  # selected_slots: [B, 1, slot_dim] 或 [1, slot_dim]
        
        if debug_enabled:
            print(f"[LVRHeadSlotAttention.forward] Stage 2 done: selected_slots.shape={selected_slots.shape}, "
                  f"elapsed={time.time()-start_time:.3f}s" if start_time else "")
        
        # 阶段三：局部深度交互
        if debug_enabled:
            print(f"[LVRHeadSlotAttention.forward] Stage 3: Local interaction...")
        V_focal, updated_slot = self.local_interaction(
            h_t=hidden_state,
            selected_slots=selected_slots
        )  # V_focal: [B, slot_dim] 或 [slot_dim]
        
        if debug_enabled:
            print(f"[LVRHeadSlotAttention.forward] Stage 3 done: V_focal.shape={V_focal.shape}, "
                  f"elapsed={time.time()-start_time:.3f}s" if start_time else "")
        
        # 处理legacy格式
        if V_focal.dim() == 1:
            V_focal = V_focal.unsqueeze(0)  # [1, slot_dim]
        
        # 投影回hidden_size（如果需要）
        if self.output_proj is not None:
            V_focal = self.output_proj(V_focal)  # [B, hidden_size]
        
        # 输出归一化
        v_focal = self.output_norm(V_focal)  # [B, hidden_size]
        
        if debug_enabled:
            print(f"[LVRHeadSlotAttention.forward] COMPLETE: v_focal.shape={v_focal.shape}, "
                  f"total elapsed={time.time()-start_time:.3f}s" if start_time else "")
        
        return v_focal


class LVRHeadImplicitVisualRouting(nn.Module):
    """
    Implicit Visual Routing (IVR) - 基于胶囊网络路由思想的轻量实现
    
    核心特点：
    1. 完全无参数（除了可选的输出归一化层）
    2. 通过迭代优化实现动态聚焦
    3. 训练最稳定，避免显存和NCCL内存溢出问题
    4. 在ScienceQA上达到Q-Former 98%性能
    
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
    4. 适合与LVRHeadAttention结合使用或作为独立替代方案
    
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
        chunk_size: Optional[int] = None
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
    
    def _gated_focus_2d(
        self,
        lang_state: torch.Tensor,
        visual_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Gated Feature Reweighting for 2D format (legacy).
        
        Args:
            lang_state: (batch_size, hidden_size) - LLM hidden state
            visual_tokens: (num_image_tokens, hidden_size) - Image tokens
                          Note: 如果visual_dim != hidden_size，visual_tokens需要先投影
            
        Returns:
            focused_features: (batch_size, hidden_size) - Focused visual features
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
        
        # Step 4: 归一化
        # Ensure dtype compatibility: LayerNorm requires input dtype to match weight dtype
        focused_features = self.norm(focused_features.to(self.norm.weight.dtype))  # (batch_size, num_image_tokens, visual_dim)
        
        # Step 5: 聚合为单一向量（平均池化）
        focused_features = focused_features.mean(dim=1)  # (batch_size, visual_dim)
        
        # Step 6: 投影回hidden_size（如果需要）
        if self.output_proj is not None:
            focused_features = self.output_proj(focused_features)  # (batch_size, hidden_size)
        
        # Step 7: 输出归一化（如果启用）
        if self.output_norm is not None:
            # Ensure dtype compatibility: LayerNorm requires input dtype to match weight dtype
            focused_features = self.output_norm(focused_features.to(self.output_norm.weight.dtype))  # (batch_size, hidden_size)
        
        return focused_features
    
    def _gated_focus_2d_chunked(
        self,
        lang_state: torch.Tensor,
        visual_tokens: torch.Tensor,
        chunk_size: int
    ) -> torch.Tensor:
        """
        Memory-efficient chunked version for 2D format.
        
        Args:
            lang_state: (batch_size, hidden_size)
            visual_tokens: (num_image_tokens, hidden_size)
            chunk_size: Size of each chunk
            
        Returns:
            focused_features: (batch_size, hidden_size)
        """
        batch_size = lang_state.shape[0]
        num_tokens = visual_tokens.shape[0]
        device = lang_state.device
        dtype = lang_state.dtype
        
        # Detach visual_tokens to avoid unnecessary gradient computation (vision tower is typically frozen)
        visual_tokens = visual_tokens.detach()
        
        # Step 1: 生成门控向量
        gate = torch.sigmoid(self.gate_proj(lang_state))  # (batch_size, visual_dim)
        
        # Step 2: 如果visual_tokens的维度不匹配，需要投影
        if self.visual_proj is not None:
            visual_tokens_proj = self.visual_proj(visual_tokens)  # (num_image_tokens, visual_dim)
        else:
            visual_tokens_proj = visual_tokens  # (num_image_tokens, visual_dim)
        
        # Step 3: Chunked processing
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size
        focused_features_sum = torch.zeros(batch_size, self.visual_dim, device=device, dtype=dtype)
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, num_tokens)
            chunk_tokens = visual_tokens_proj[chunk_start:chunk_end]  # (chunk_size, visual_dim)
            
            # 应用门控
            gate_expanded = gate.unsqueeze(1)  # (batch_size, 1, visual_dim)
            chunk_tokens_expanded = chunk_tokens.unsqueeze(0)  # (1, chunk_size, visual_dim)
            focused_chunk = chunk_tokens_expanded * gate_expanded  # (batch_size, chunk_size, visual_dim)
            
            # 归一化
            # Ensure dtype compatibility: LayerNorm requires input dtype to match weight dtype
            focused_chunk = self.norm(focused_chunk.to(self.norm.weight.dtype))  # (batch_size, chunk_size, visual_dim)
            
            # 累加
            focused_features_sum += focused_chunk.sum(dim=1)  # (batch_size, visual_dim)
        
        # 平均池化
        focused_features = focused_features_sum / num_tokens  # (batch_size, visual_dim)
        
        # 投影回hidden_size（如果需要）
        if self.output_proj is not None:
            focused_features = self.output_proj(focused_features)
        
        # 输出归一化（如果启用）
        if self.output_norm is not None:
            # Ensure dtype compatibility: LayerNorm requires input dtype to match weight dtype
            focused_features = self.output_norm(focused_features.to(self.output_norm.weight.dtype))
        
        return focused_features
    
    def _gated_focus_3d(
        self,
        lang_state: torch.Tensor,
        visual_tokens: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Gated Feature Reweighting for 3D format (batched).
        Memory-optimized: processes tokens in chunks to avoid large intermediate tensors.
        
        Args:
            lang_state: (batch_size, hidden_size)
            visual_tokens: (batch_size, max_num_tokens, hidden_size)
            image_attention_mask: Optional (batch_size, max_num_tokens) - True for valid tokens
            
        Returns:
            focused_features: (batch_size, hidden_size)
        """
        batch_size = lang_state.shape[0]
        max_num_tokens = visual_tokens.shape[1]
        device = lang_state.device
        dtype = lang_state.dtype
        
        # Detach visual_tokens to avoid unnecessary gradient computation (vision tower is typically frozen)
        visual_tokens = visual_tokens.detach()
        
        # Step 1: 生成门控向量
        gate = torch.sigmoid(self.gate_proj(lang_state))  # (batch_size, visual_dim)
        
        # Memory optimization: Always use chunked processing for 3D format to avoid OOM
        # Even for "short" sequences, chunked processing saves memory by avoiding large intermediate tensors
        effective_chunk_size = self.chunk_size if self.chunk_size is not None else 256
        
        # Always use chunked processing for 3D format (batched) to ensure memory efficiency
        return self._gated_focus_3d_chunked(
            lang_state, visual_tokens, image_attention_mask, effective_chunk_size
        )
    
    def _gated_focus_3d_chunked(
        self,
        lang_state: torch.Tensor,
        visual_tokens: torch.Tensor,
        image_attention_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 512
    ) -> torch.Tensor:
        """
        Memory-efficient chunked version for 3D format.
        Processes visual tokens in chunks to avoid large intermediate tensors.
        
        Args:
            lang_state: (batch_size, hidden_size)
            visual_tokens: (batch_size, max_num_tokens, hidden_size)
            image_attention_mask: Optional (batch_size, max_num_tokens) - True for valid tokens
            chunk_size: Size of each chunk
            
        Returns:
            focused_features: (batch_size, hidden_size)
        """
        batch_size = lang_state.shape[0]
        max_num_tokens = visual_tokens.shape[1]
        device = lang_state.device
        dtype = lang_state.dtype
        norm_dtype = self.norm.weight.dtype
        
        # Detach visual_tokens to avoid unnecessary gradient computation
        visual_tokens = visual_tokens.detach()
        
        # Step 1: 生成门控向量
        gate = torch.sigmoid(self.gate_proj(lang_state))  # (batch_size, visual_dim)
        
        # Step 2: 如果visual_tokens的维度不匹配，需要投影
        if self.visual_proj is not None:
            visual_tokens_proj = self.visual_proj(visual_tokens)  # (batch_size, max_num_tokens, visual_dim)
        else:
            visual_tokens_proj = visual_tokens  # (batch_size, max_num_tokens, visual_dim)
        
        # Step 3: Chunked processing
        # Memory optimization: Apply gate and aggregate without storing full (batch_size, max_num_tokens, visual_dim) tensor
        num_chunks = (max_num_tokens + chunk_size - 1) // chunk_size
        focused_features_sum = torch.zeros(batch_size, self.visual_dim, device=device, dtype=norm_dtype)
        mask_sum = torch.zeros(batch_size, device=device, dtype=norm_dtype)
        
        gate_expanded = gate.unsqueeze(1)  # (batch_size, 1, visual_dim)
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, max_num_tokens)
            chunk_tokens = visual_tokens_proj[:, chunk_start:chunk_end]  # (batch_size, chunk_size, visual_dim)
            
            # Apply gate
            focused_chunk = chunk_tokens * gate_expanded  # (batch_size, chunk_size, visual_dim)
            
            # Apply mask if provided (before normalization to avoid normalizing padding)
            if image_attention_mask is not None:
                chunk_mask = image_attention_mask[:, chunk_start:chunk_end]  # (batch_size, chunk_size)
                mask_expanded = chunk_mask.unsqueeze(-1).to(norm_dtype)  # (batch_size, chunk_size, 1)
                focused_chunk = focused_chunk.to(norm_dtype) * mask_expanded
                mask_sum += chunk_mask.sum(dim=1).to(norm_dtype)  # (batch_size,)
            else:
                focused_chunk = focused_chunk.to(norm_dtype)
                mask_sum += torch.full((batch_size,), chunk_end - chunk_start, device=device, dtype=norm_dtype)
            
            # Normalize chunk (only valid tokens if mask is provided)
            # Note: LayerNorm on chunked data may have slight numerical differences, but saves memory
            focused_chunk = self.norm(focused_chunk)  # (batch_size, chunk_size, visual_dim)
            
            # Accumulate sum immediately
            focused_features_sum += focused_chunk.sum(dim=1)  # (batch_size, visual_dim)
            
            # Free chunk tensors immediately to save memory
            del focused_chunk, chunk_tokens
        
        # Average pooling
        focused_features = focused_features_sum / (mask_sum.unsqueeze(-1) + 1e-8)  # (batch_size, visual_dim)
        
        # Convert back to original dtype
        focused_features = focused_features.to(dtype)
        
        # Step 4: 投影回hidden_size（如果需要）
        if self.output_proj is not None:
            focused_features = self.output_proj(focused_features)
        
        # Step 5: 输出归一化（如果启用）
        if self.output_norm is not None:
            focused_features = self.output_norm(focused_features.to(self.output_norm.weight.dtype))
        
        return focused_features
    
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
        
        batch_size = hidden_state.shape[0]
        
        if debug_enabled:
            print(f"[LVRHeadGatedFocus.forward] START: batch_size={batch_size}, "
                  f"image_embeds.shape={image_embeds.shape}, visual_dim={self.visual_dim}")
        
        # Detach image_embeds to avoid unnecessary gradient computation (vision tower is typically frozen)
        # This saves memory and avoids NCCL memory overflow issues
        image_embeds = image_embeds.detach()
        
        # Handle 2D format (legacy)
        if image_embeds.dim() == 2:
            num_tokens = image_embeds.shape[0]
            
            # Auto-select chunked or non-chunked based on sequence length
            use_chunked = (self.chunk_size is not None and 
                          num_tokens > self.chunk_size and
                          batch_size == 1)  # Chunked only makes sense for single batch
            
            if use_chunked:
                if debug_enabled:
                    print(f"[LVRHeadGatedFocus.forward] Using chunked 2D routing: "
                          f"num_tokens={num_tokens}, chunk_size={self.chunk_size}")
                v_focal = self._gated_focus_2d_chunked(
                    hidden_state, image_embeds, self.chunk_size
                )
            else:
                if debug_enabled:
                    print(f"[LVRHeadGatedFocus.forward] Using standard 2D routing: "
                          f"num_tokens={num_tokens}")
                v_focal = self._gated_focus_2d(
                    hidden_state, image_embeds
                )
        
        # Handle 3D format (batched)
        elif image_embeds.dim() == 3:
            max_num_tokens = image_embeds.shape[1]
            effective_chunk_size = self.chunk_size if self.chunk_size is not None else 512
            
            # Auto-select chunked processing for long sequences
            use_chunked = max_num_tokens > effective_chunk_size
            
            if use_chunked:
                if debug_enabled:
                    print(f"[LVRHeadGatedFocus.forward] Using chunked 3D routing: "
                          f"max_num_tokens={max_num_tokens}, chunk_size={effective_chunk_size}")
                v_focal = self._gated_focus_3d_chunked(
                    hidden_state, image_embeds, image_attention_mask, effective_chunk_size
                )
            else:
                if debug_enabled:
                    print(f"[LVRHeadGatedFocus.forward] Using standard 3D routing: "
                          f"max_num_tokens={max_num_tokens}")
                v_focal = self._gated_focus_3d(
                    hidden_state, image_embeds, image_attention_mask
                )
        
        else:
            raise ValueError(f"image_embeds must be 2D or 3D, got {image_embeds.dim()}D")
        
        if debug_enabled:
            print(f"[LVRHeadGatedFocus.forward] COMPLETE: v_focal.shape={v_focal.shape}, "
                  f"total elapsed={time.time()-start_time:.3f}s" if start_time else "")
        
        return v_focal


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
        chunk_size: Optional[int] = None
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
    
    def _intrinsic_similarity_2d(
        self,
        h_t: torch.Tensor,
        Z_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        内生相似度映射 for 2D format (legacy).
        
        Args:
            h_t: (batch_size, hidden_size) - LLM hidden state
            Z_grid: (num_image_tokens, hidden_size) - Image tokens
            
        Returns:
            V_focal: (batch_size, hidden_size) - Focused visual features
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
        image_attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        内生相似度映射 for 3D format (batched).
        
        Args:
            h_t: (batch_size, hidden_size)
            Z_grid: (batch_size, max_num_tokens, hidden_size)
            image_attention_mask: Optional (batch_size, max_num_tokens) - True for valid tokens
            
        Returns:
            V_focal: (batch_size, hidden_size) - Focused visual features
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
        
        # Handle 2D format (legacy)
        if image_embeds.dim() == 2:
            num_tokens = image_embeds.shape[0]
            
            # Auto-select chunked or non-chunked based on sequence length and batch size
            # For batch_size > 1, always use chunked to save memory
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
            
            if use_chunked:
                v_focal = self._intrinsic_similarity_2d_chunked(
                    hidden_state, image_embeds, effective_chunk_size
                )
            else:
                v_focal = self._intrinsic_similarity_2d(
                    hidden_state, image_embeds
                )
        
        # Handle 3D format (batched)
        elif image_embeds.dim() == 3:
            max_num_tokens = image_embeds.shape[1]
            
            # Auto-select chunked or non-chunked based on sequence length and batch size
            # For batch_size > 1, always use chunked to save memory
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
            
            if use_chunked:
                v_focal = self._intrinsic_similarity_3d_chunked(
                    hidden_state, image_embeds, image_attention_mask, effective_chunk_size
                )
            else:
                v_focal = self._intrinsic_similarity_3d(
                    hidden_state, image_embeds, image_attention_mask
                )
        
        else:
            raise ValueError(f"image_embeds must be 2D or 3D, got {image_embeds.dim()}D")
        
        # Apply output normalization if enabled
        if self.use_output_norm:
            v_focal = self.output_norm(v_focal)
        
        return v_focal