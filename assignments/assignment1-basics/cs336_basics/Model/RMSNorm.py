import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # size: (d_model,)
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: (batch_size, seq_len, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Convert to float32 for numerical stability
        # \sqrt(\sum(x^2, dim=-1) / d_model + eps) * self.weight
        y = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / y 
        # size of x: (batch_size, seq_len, d_model)
        x = x * self.weight.to(in_dtype)
        return x.to(in_dtype)  # Convert back to original dtype