import torch
from typing import Union

from ..utils.tensors import is_broadcastable

def get_dot_prod_contribution(
    parts: torch.Tensor, 
    whole: torch.Tensor, 
    device: Union[str, torch.device] = None
) -> torch.Tensor:
    """
    Calculates the dot product contribution of 'parts' onto 'whole'.
    Typically used to project individual components (parts) onto a specific 
    direction (whole, e.g., a residual stream vector or a weight vector).
    """
    if device:
        parts = parts.to(device)
        whole = whole.to(device)

    whole_sh = whole.shape
    parts_sh = parts.shape
    
    if len(parts_sh) < len(whole_sh):
        raise ValueError(f"The 'parts' {tuple(parts_sh)} must have at least the same number of dimensions as the 'whole' {tuple(whole_sh)}")
    
    if whole_sh[-1] != parts_sh[-1]:
        raise ValueError(f"The reduction dimension of 'parts' {tuple(parts_sh)} and 'whole' {tuple(whole_sh)} must be the same.")
    
    if not is_broadcastable(parts_sh, whole_sh):
        raise ValueError(f"The 'parts' tensor {tuple(parts_sh)} and 'whole' tensor {tuple(whole_sh)} are not broadcastable.")
    
    dot_prod = (parts * whole).sum(-1)
    return dot_prod