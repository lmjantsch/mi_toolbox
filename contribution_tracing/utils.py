import torch
from typing import Union, Tuple

@torch.no_grad()
def get_top_x_contribution_values(contributions: torch.Tensor, top_x: float, min_total_threshold: float =1.0) -> torch.Tensor:

    net_sum = contributions.sum(dim=-1, keepdim=True)
    valid_row_mask = net_sum > min_total_threshold

    pos_contributions = torch.clamp(contributions, min=0)
    sorted_pos_values, sorted_pos_idx = torch.sort(pos_contributions, descending=True, dim=-1)
    
    cummulative_sum = torch.cumsum(sorted_pos_values, dim=-1)

    threshold = net_sum * top_x
    mask = cummulative_sum <= threshold
    
    # include the item that crossed the threshold
    mask = mask.roll(shifts=1, dims=-1)
    mask[..., 0] = 1 

    final_mask = mask & (sorted_pos_values > 0) & valid_row_mask
    filtered_values = torch.where(final_mask, sorted_pos_values, 0.0)

    inversed_idx = sorted_pos_idx.argsort(dim=-1)
    filtered_contributions = filtered_values.gather(-1, inversed_idx)

    return filtered_contributions

@torch.no_grad()
def mean_center_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor - tensor.mean(dim=-1, keepdim=True)

@torch.no_grad()
def trace_through_layer_norm(target: torch.Tensor, norm_W: torch.Tensor, norm_var: torch.Tensor, device: Union[str, torch.device] = 'cuda') -> torch.Tensor:
    input_device = target.device
    if not device:
        device = input_device
    
    var_scaled_target = target.to(device) * norm_var[:, None].to(device)
    norm_w_scaled_target = var_scaled_target / norm_W.to(device)

    return norm_w_scaled_target.to(input_device)

def is_broadcastable(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> bool:
    
    s1 = list(reversed(shape1))
    s2 = list(reversed(shape2))
    
    max_dims = max(len(s1), len(s2))
    
    while len(s1) < max_dims:
        s1.append(1)
    while len(s2) < max_dims:
        s2.append(1)
        
    for dim1, dim2 in zip(s1, s2):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return False
            
    return True