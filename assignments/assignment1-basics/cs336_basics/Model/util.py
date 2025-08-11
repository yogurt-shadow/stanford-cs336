import torch
from einops import einsum

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the softmax of the input tensor along the last dimension.
    
    Args:
        x (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: Softmax of the input tensor.
    """
    largest = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - largest)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def scaled_dot_product_attention(
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes the scaled dot product attention.
    
    Args:
        keys (torch.Tensor): Keys tensor.
        queries (torch.Tensor): Queries tensor.
        values (torch.Tensor): Values tensor.
        masks (torch.Tensor | None): Optional mask tensor.
        
    Returns:
        torch.Tensor: Result of the scaled dot product attention.
    """
    
    """
    size of inputs:
    1. keys: (batch_size, ..., seq_len, d_k)
    2. queries: (batch_size, ..., seq_len, d_k)
    3. values: (batch_size, ..., seq_len, d_v)
    4. masks: (seq_len, seq_len) or None

    output:
    (batch_size, ..., d_v)
    """
    d_k = queries.size(-1)
    # Q * K^T / \sqrt{d_k}
    scores = einsum(queries, keys, "... i d_k, ... j d_k -> ... i j") / (d_k ** 0.5)
    if masks is not None:
        # Apply mask to scores
        scores = scores.masked_fill(masks == 0, float("-inf"))
    # Softmax to get attention weights
    weights = softmax(scores, dim=-1)
    # Multiply weights with values
    # size of weights: (batch_size, ..., seq_len, seq_len)
    # size of values: (batch_size, ..., seq_len, d_v)
    output = einsum(weights, values, "... i j, ... j d_v -> ... i d_v")
    return output
