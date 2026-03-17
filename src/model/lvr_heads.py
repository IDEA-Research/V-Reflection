import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from typing import Optional


class BoxGuidedCompression(nn.Module):
    """
    Box-Guided Compression: Resamples variable-length bbox visual features into fixed num_queries latent tokens
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
        return_attention: bool = False,
    ):
        """
        Args:
            bbox_region_features: (L, N, D) L=num_bboxes, N=variable tokens per bbox, D=vision_dim or hidden_size
            key_padding_mask: (L, N) True for padding positions (ignore), False for valid. Optional.
            return_attention: If True, also return attention weights (L, num_queries, N).
        Returns:
            (L, num_queries, hidden_size) or ((L, num_queries, hidden_size), (L, num_queries, N)) when return_attention
        """
        L, N, D = bbox_region_features.shape
        if self.vision_proj is not None:
            bbox_region_features = self.vision_proj(bbox_region_features)
        q = self.queries.expand(L, -1, -1)
        attn_out, attn_weights = self.cross_attn(
            q,
            bbox_region_features,
            bbox_region_features,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
        )
        out = self.output_norm(attn_out)
        if return_attention:
            return out, attn_weights
        return out

    def get_orthogonality_loss(self) -> torch.Tensor:
        """Orthogonality loss to prevent attention collapse: penalize non-orthogonal queries.
        Computed in float32 to avoid gradient underflow in bf16. Scaled by 1000 so gradient
        magnitude is comparable to other losses (raw MSE ~0.0002 at init would underflow)."""
        q = self.queries.squeeze(0).float()  # [8, D], float32 for stable gradients
        q_norm = F.normalize(q, p=2, dim=-1)
        sim_matrix = torch.matmul(q_norm, q_norm.transpose(0, 1))  # [8, 8]
        identity = torch.eye(sim_matrix.size(0), device=sim_matrix.device, dtype=torch.float32)
        loss = F.mse_loss(sim_matrix, identity)
        return loss * 1000.0  # scale so initial ~0.2, gradient flows properly


class DynamicAutoregressiveCompression(nn.Module):
    """
    Dynamic Autoregressive Compression (Stage 2): Uses LLM's 8 autoregressive hidden states at <lvr> positions as Queries,
    cross-attends to full image features, outputs predicted latent tokens aligned with Teacher target.
    No MLP query projection - queries come directly from LLM hidden states.
    """
    def __init__(
        self,
        hidden_size: int,
        llm_hidden_size: Optional[int] = None,
        vision_dim: Optional[int] = None,
        num_queries: int = 8,
        num_heads: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        llm_hidden_size = llm_hidden_size or hidden_size
        vision_dim = vision_dim or hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.vision_dim = vision_dim
        num_heads = num_heads or min(8, hidden_size // 64)
        self.num_heads = num_heads

        if llm_hidden_size != hidden_size:
            self.q_proj = nn.Linear(llm_hidden_size, hidden_size, bias=False)
        else:
            self.q_proj = None
        if vision_dim != hidden_size:
            self.kv_proj = nn.Linear(vision_dim, hidden_size, bias=False)
        else:
            self.kv_proj = None

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = LayerNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        lvr_hidden_states: torch.Tensor,
        full_image_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        Args:
            lvr_hidden_states: [B, 8, D_llm] - LLM hidden states at 8 consecutive <lvr> positions
            full_image_features: [B, Seq_Len, D_vis] - Full image features from visual encoder
            key_padding_mask: [B, Seq_Len] - True for padding positions (ignore), False for valid. Optional.
            return_attention: If True, also return attention weights (B, 8, Seq_Len).
        Returns:
            [B, 8, hidden_size] or ((B, 8, hidden_size), (B, 8, Seq_Len)) when return_attention
        """
        B, Q, _ = lvr_hidden_states.shape
        if self.q_proj is not None:
            q = self.q_proj(lvr_hidden_states)
        else:
            q = lvr_hidden_states
        if self.kv_proj is not None:
            kv = self.kv_proj(full_image_features)
        else:
            kv = full_image_features
        attn_out, attn_weights = self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=return_attention
        )
        out = self.output_norm(attn_out)
        if return_attention:
            return out, attn_weights
        return out


# Backward compatibility aliases
BoxFeatureResampler = BoxGuidedCompression
DynamicAutoregressiveResampler = DynamicAutoregressiveCompression
