import torch
from einops import einsum

# implement own linear class with not nn.Linear
class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # size: (out_features, in_features)
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        weight_variance = 2.0 / (in_features + out_features)
        weight_std = weight_variance**0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=weight_std, a=-3., b=3.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... b i, o i -> ... b o")