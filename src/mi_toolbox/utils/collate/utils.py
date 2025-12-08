from typing import List
import torch

def max_pad_sequence(sequence: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
    """
    Pads a list of tensors to the maximum shape in the batch and stacks them.
    """
    if not all(isinstance(tensor, torch.Tensor) for tensor in sequence):
        raise TypeError(f"Expected all elements to be tensors")

    if not sequence:
        return torch.tensor([])

    num_dim = len(sequence[0].shape)
    max_shape = []

    for tensor in sequence:
        if len(tensor.shape) != num_dim:
            raise ValueError(f"Expected all tensors to have same number of dimension.")
        max_shape.append(tensor.shape)

    max_shape_tensor = torch.tensor(max_shape)
    target_shape = max_shape_tensor.max(dim=0).values.tolist()

    padded_tensors = []
    for tensor in sequence:
        curr_shape = tensor.shape

        tensor_slice = tuple(slice(0, dim) for dim in curr_shape)
        padded_tensor = torch.full(target_shape, padding_value, dtype=tensor.dtype, device=tensor.device)

        padded_tensor[tensor_slice] = tensor
        padded_tensors.append(padded_tensor)

    return torch.stack(padded_tensors, dim=0)