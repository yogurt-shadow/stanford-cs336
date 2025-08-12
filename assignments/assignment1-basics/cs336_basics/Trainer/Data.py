import torch
import numpy as np
import numpy.typing as npt
import os
import typing

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a batch of data from the dataset.
    
    Args:
        dataset: The dataset to sample from.
        batch_size: The number of samples in the batch.
        context_length: The length of the context for each sample.
        device: The device to which the tensors should be moved.
    
    Returns:
        A tuple containing the input and target tensors for the batch.
    """
    data_len = len(dataset)
    inputs = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    targets = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)

    for i in range(batch_size):
        # [10, 20, 30, ..., 100, 110, 120]
        # inputs: [10, 20, 30, ..., 100]
        # targets: [20, 30, ..., 110]
        start_idx = np.random.randint(0, data_len - context_length)
        inputs[i] = torch.tensor(dataset[start_idx:start_idx + context_length], dtype=torch.long, device=device)
        targets[i] = torch.tensor(dataset[start_idx + 1:start_idx + context_length + 1], dtype=torch.long, device=device)
    return inputs, targets

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
):
    datas = torch.load(src)
    model.load_state_dict(datas['model_state_dict'])
    optimizer.load_state_dict(datas['optimizer_state_dict'])
    iteration = datas['iteration']
    return iteration

        