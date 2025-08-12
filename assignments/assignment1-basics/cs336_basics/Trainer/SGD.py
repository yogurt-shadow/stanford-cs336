from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("step", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["step"] = t + 1
        return loss
    
if __name__ == "__main__":
    lrs = [1e1, 1e2, 1e3]
    init_weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    datas = {}
    for lr in lrs:
        weights = torch.nn.Parameter(init_weights.clone())
        optimizer = SGD([weights], lr=lr)
        datas[lr] = []
        for t in range(10):
            optimizer.zero_grad()
            loss = weights.sum()
            loss.backward()
            optimizer.step()
            datas[lr].append(weights.clone())
    for lr, data in datas.items():
        print(f"Learning rate: {lr}")
        for t, w in enumerate(data):
            print(f"  Step {t}: {w.mean().item():.4f}, {w.std().item():.4f}")