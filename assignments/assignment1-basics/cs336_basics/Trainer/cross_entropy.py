import torch


def log_sum_exp(inputs: torch.Tensor) -> torch.Tensor:
    """
    Computes log(sum(exp(inputs))) in a numerically stable way.

    log(\sum(e^inputs)) = log(\sum(e^(inputs - max(inputs)))) + max(inputs)
    """
    max_value = inputs.max(dim=-1, keepdim=True).values
    inputs2 = inputs - max_value
    return max_value + torch.log(inputs2.exp().sum(dim=-1, keepdim=True))

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    size:

    inputs: (batch_size, num_classes)
    targets: (batch_size,)
    """
    # logits that match the right output
    target_logits = inputs.gather(dim = -1, index = targets.unsqueeze(-1)).squeeze(-1)
    # compute the cross-entropy loss
    # loss = -z + \sum(log(exp(z_i)))
    loss = -target_logits + log_sum_exp(inputs)
    return loss.mean()
    