from typing import List
import torch

def max_pad_sequence(sequence: List[torch.Tensor], padding_value: int = 0):
    if not all([isinstance(tensor, torch.Tensor) for tensor in sequence]):
        TypeError(f"Expected all elements to be tensors")

    max_shape = []
    num_dim = len(sequence[0].shape)
    for tensor in sequence:
        if len(tensor.shape) != num_dim:
            raise ValueError(f"Expected all tensors to have same number of dimension.")
        max_shape.append(tensor.shape)
    max_shape = (torch.tensor(max_shape).max(dim=0).values).tolist()

    padded_tensors = []
    for tensor in sequence:
        curr_shape = tensor.shape
        tensor_slice = [slice(0, dim) for dim in curr_shape]
        padded_tensor = torch.full(max_shape, padding_value, dtype=tensor.dtype)
        padded_tensor[*tensor_slice] = tensor
        padded_tensors.append(padded_tensor)

    return torch.stack(padded_tensors, dim=0)