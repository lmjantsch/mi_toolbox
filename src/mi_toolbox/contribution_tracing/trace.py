import torch
from typing import Union

EPSILON = 1e-7

@torch.no_grad()
def trace_through_layer_norm(
    target: torch.Tensor, 
    norm_W: torch.Tensor, 
    norm_std: torch.Tensor, 
    device: Union[str, torch.device] = 'cuda'
) -> torch.Tensor:
    """
    Approximates the contribution of a target tensor through a LayerNorm by 
    scaling it by the standard deviation and the LayerNorm weights.
    """
    input_device = target.device
    if not device:
        device = input_device

    target = target.to(device)
    norm_W = norm_W.to(device)
    norm_std = norm_std.to(device).unsqueeze(-1)
    

    var_scaled_target = target / (norm_std + EPSILON)
    norm_w_scaled_target = var_scaled_target * norm_W

    return norm_w_scaled_target.to(input_device)