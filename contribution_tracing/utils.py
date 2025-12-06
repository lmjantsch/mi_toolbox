import torch
from typing import Union, Tuple

@torch.no_grad()
def get_top_x_contribution_values(
    contributions: torch.Tensor, 
    top_x: float, 
    min_total_threshold: float = 1.0
) -> torch.Tensor:
    """
    Filters the tensor to keep only the largest positive values that sum up to 
    'top_x' percent of the net sum. Preserves original shape and positions.
    """

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
    """
    Subtracts the mean over the last dimension.
    """
    return tensor - tensor.mean(dim=-1, keepdim=True)