import torch
from cs336_basics.Model.Linear import Linear
from cs336_basics.Model.Embedding import Embedding
from cs336_basics.Model.RMSNorm import RMSNorm
from cs336_basics.Model.Positionwise_Feedforward import PositionwiseFeedforward
from cs336_basics.Model.Multihead_self_attention import Multihead_self_attention

class Transformer_Block(torch.nn.Module):
    def __init__(self, d_model: int, 
                 num_heads: int,
                 d_ff: int,
                 use_rope: bool = False,
                 theta: float | None = None,
                 max_seq_len: int | None = None
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.d_ff = d_ff

        # components
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attention = Multihead_self_attention(
            d_model=d_model, 
            num_heads=num_heads, 
            use_rope=use_rope, 
            theta=theta, 
            max_seq_len=max_seq_len
        )
        self.ffn = PositionwiseFeedforward(
            d_model=d_model, 
            d_ff=d_ff
        )

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer block.

        Args:
            in_features (Float[Tensor, " ... sequence_length d_model"]): Input tensor with shape (..., sequence_length, d_model).

        Returns:
            Float[Tensor, " ... sequence_length d_model"]: Output tensor with the same shape as input.
        """
        x = self.norm1(in_features)
        x = self.attention(x)
        y = in_features + x  # residual connection
        z = self.norm2(y)
        z = self.ffn(z)
        return y + z  # another residual connection
    
class Transformer(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float):
        super().__init__()
        
        # components
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.attention_layers = torch.nn.ModuleList([
            Transformer_Block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                use_rope=True,
                theta=rope_theta,
                max_seq_len=context_length
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output_layer = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, vocab_size).
        """
        x = self.embedding(input_ids)
        for layer in self.attention_layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_layer(x)
        return x