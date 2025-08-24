"""
Nsight annotated version of scaled dot product attention.
"""

import torch, math
from torch import Tensor
from einops import einsum
import torch.cuda.nvtx as nvtx
from jaxtyping import Float, Bool

@nvtx.range("scaled_dot_product_attention")
def annotated_scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Compute scaled dot product attention.

    Args:
        Q: Queries of shape (..., queries, d_k)
        K: Keys of shape (..., keys, d_k)
        V: Values of shape (..., keys, d_v)
        mask: Optional mask of shape (..., queries, keys)

    Returns:
        Attention output of shape (..., queries, d_v)
    """
    # block 1: compute attention scores
    with nvtx.range("computing attention scores", color="blue"):
        d_k = Q.shape[-1]
        # score = Q*K^T / sqrt(d_k)
        scores = einsum(Q, K, "... i d_k, ... j d_k -> ... i j") / math.sqrt(d_k)
        if mask is not None:
            scores = torch.where(mask, scores, float("-inf"))

    # block 2: compute softmax
    with nvtx.range("computing softmax", color="orange"):
        attn = torch.softmax(scores, dim=-1)
    
    # block 3: compute attention output
    with nvtx.range("final matmul", color="yellow"):
        output = einsum(attn, V, "... i j, ... j d_v -> ... i d_v")
    return output
