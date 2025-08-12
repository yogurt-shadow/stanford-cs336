import torch
import math
from collections.abc import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """
    Clips the gradients of the parameters to have a maximum L2 norm.
    
    Args:
        parameters (Iterable[torch.nn.Parameter]): The parameters whose gradients will be clipped.
        max_l2_norm (float): The maximum allowed L2 norm for the gradients.
    """
    eps = 1e-6
    # compute l2 norm of all gradients
    p_grads = [p.grad for p in parameters if p.grad is not None]
    p_grad_norm = [torch.norm(p_grad, p=2) for p_grad in p_grads]
    total_norm = torch.norm(torch.stack(p_grad_norm), p=2).item()
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= scale