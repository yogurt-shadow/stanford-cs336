import torch
from einops import einsum

class PositionwiseFeedforward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = int(d_model * 8 / 3)
        else:
            self.d_ff = d_ff
        # W1, W3: (d_ff, d_model)
        self.weight1 = torch.nn.Parameter(torch.empty(self.d_ff, self.d_model))
        self.weight3 = torch.nn.Parameter(torch.empty(self.d_ff, self.d_model))
        # W2: (d_model, d_ff)
        self.weight2 = torch.nn.Parameter(torch.empty(self.d_model, self.d_ff))
        # init weights
        weight_variance = 2.0 / (self.d_model + self.d_ff)
        weight_std = weight_variance**0.5
        torch.nn.init.trunc_normal_(self.weight1, mean=0.0, std=weight_std, a=-3., b=3.)
        torch.nn.init.trunc_normal_(self.weight2, mean=0.0, std=weight_std, a=-3., b=3.)
        torch.nn.init.trunc_normal_(self.weight3, mean=0.0, std=weight_std, a=-3., b=3.)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).
        
        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        # 1. calculate w1*x, w3*x
        w1_x = einsum(x, self.weight1, "... d_model, d_ff d_model -> ... d_ff")
        w3_x = einsum(x, self.weight3, "... d_model, d_ff d_model -> ... d_ff")
        # 2. apply silu activation
        w1_x_silu = self.silu(w1_x)
        # 3. calculate w1_x_silu \dots w3_x
        w1_x_silu_w3_x = einsum(w1_x_silu, w3_x, "... d_ff, ... d_ff -> ... d_ff")
        # 4. calculate w2 * (w1_x_silu \dots w3_x)
        output = einsum(w1_x_silu_w3_x, self.weight2,
                        "... d_ff, d_model d_ff -> ... d_model")
        return output
        