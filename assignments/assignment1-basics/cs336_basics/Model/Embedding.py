import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # size: (num_embeddings, embedding_dim)
        # num_embeddings is the number of tokens in the vocabulary
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        weight_variance = 2.0 / (num_embeddings + embedding_dim)
        weight_std = weight_variance**0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=weight_std, a=-3., b=3.)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.weight[indices]
