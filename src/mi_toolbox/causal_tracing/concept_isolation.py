import torch
import torch.nn.functional as F

def get_mass_mean_vectors(
    constrastive_pairs: torch.Tensor, 
    normalize: bool = True, 
    keepdim: bool = False
) -> torch.Tensor:
    """Returns the mass mean vector of a batch of contrastive pairs.

    Args:
        constrastive_pairs (torch.Tensor): Contrastive sample pairs (batch_dim, 2, num_samples, emb_dim)
        normalize (bool, optional): Returns normalized vector. Defaults to True.
        keepdim (bool, optional): If True, broadcasts the result back to input shape and applies sign flip 
                                  to the second group. Defaults to False.

    Returns:
        torch.Tensor: Mass mean vector.
    """
    num_homographs, _, num_examples, _ = constrastive_pairs.size()
    mean_vec_1, mean_vec_2 = constrastive_pairs.mean(-2).transpose(0, 1)
    mass_mean_vec = mean_vec_1 - mean_vec_2

    if normalize:
        mass_mean_vec = F.normalize(mass_mean_vec, dim=-1)

    if keepdim:
        mass_mean_vec = mass_mean_vec[ :, None, None, :].repeat(1, 2, num_examples, 1)
        signs = torch.tensor([1, -1], device=mass_mean_vec.device, dtype=mass_mean_vec.dtype)
        mass_mean_vec *= signs[None, :, None, None]
        
    return mass_mean_vec