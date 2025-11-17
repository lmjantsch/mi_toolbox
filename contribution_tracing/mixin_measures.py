import torch
from typing import Union

def get_dot_prod_contribution(parts: torch.Tensor, whole: torch.Tensor, device: Union[str, torch.device] = None) -> torch.Tensor:
    whole_sh = whole.shape
    parts_sh = parts.shape
    batch_dims = len(parts_sh) - len(whole_sh)
    
    if batch_dims < 0:
        raise ValueError(f"The 'parts' {tuple(parts_sh)} must have at least the same number of dimensions as the 'whole' {tuple(whole_sh)}")
    if parts_sh[batch_dims:] != whole_sh:
        raise ValueError(f"The non-batrch shape of 'parts' {tuple(parts_sh[batch_dims:])} must be the same as the shape of 'whole' {tuple(whole_sh)}")
    
    dot_prod = (parts * whole).sum(-1)
    return dot_prod