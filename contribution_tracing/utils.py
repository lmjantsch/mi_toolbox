import torch

def get_top_x_contribution_values(contributions: torch.Tensor, top_x: float) -> torch.Tensor:

    sorted_values, sorted_idx = torch.sort(contributions, descending=True, dim=-1)
    cummulative_sum = torch.cumsum(sorted_values.clip(0), dim=-1)

    threshold = sorted_values.sum(dim=-1) * top_x
    mask = cummulative_sum <= threshold[:, None]
    mask = mask.roll(shifts=1, dims=(-1))
    mask[:, 0] = 1
    filtered_values = torch.where(mask, sorted_values, 0)

    inversed_idx = sorted_idx.argsort(dim=-1)
    filtered_contributions = filtered_values.gather(-1, inversed_idx)

    return filtered_contributions

def mean_center_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor - tensor.mean(dim=-1, keepdim=True)