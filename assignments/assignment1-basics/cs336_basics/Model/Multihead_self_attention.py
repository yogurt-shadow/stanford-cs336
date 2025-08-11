import torch
from cs336_basics.Model.Linear import Linear
from cs336_basics.Model.RotaryPositionalEmbedding import RotaryPositionalEmbedding
from cs336_basics.Model.util import scaled_dot_product_attention

class Multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model: int, 
                 num_heads: int,
                use_rope: bool = False,
                token_positions: torch.Tensor | None = None,
                theta: float | None = None,
                max_seq_len: int | None = None
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.token_positions = token_positions
        self.theta = theta
        self.max_seq_len = max_seq_len

        """
        size of input:
        Q, K: ... d_k d_in
        V: ... d_v d_in
        out_proj: ... d_model d_v
        in_features: ... seq_len d_in
        token_positions: ... seq_len

        output of forward: ... seq_len d_model
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        # Init weights
        self.Q = Linear(in_features=d_model, out_features=d_model)
        self.K = Linear(in_features=d_model, out_features=d_model)
        self.V = Linear(in_features=d_model, out_features=d_model)
        self.out_proj = Linear(in_features=d_model, out_features=d_model)

    def forward(self, 
                in_features: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            in_features (Float[Tensor, "... d_model"]): Input tensor.
            token_positions (torch.Tensor | None): Token positions for RoPE.

        Returns:
            Float[Tensor, "... d_model"]: Output tensor after multi-head self-attention.
        """
        B, S, D = in_features.size()

        # (B, S, D) -> (B, S, num_heads, head_dim) -> (B, num_heads, S, head_dim)
        q = self.Q(in_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.K(in_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.V(in_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope is True and self.theta is not None and self.max_seq_len is not None:
            rope = RotaryPositionalEmbedding(
                theta=self.theta,
                d_k=self.head_dim,
                max_seq_len=self.max_seq_len
            )
            q = rope(q, self.token_positions)
            k = rope(k, self.token_positions)

        masks = torch.ones((B, self.num_heads, S, S))
        masks = torch.tril(masks, diagonal=0).bool()

        # Compute scaled dot product attention
        attn_output = scaled_dot_product_attention(
            keys=k,
            queries=q,
            values=v,
            masks=masks
        )
        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(attn_output)
        return output
        
