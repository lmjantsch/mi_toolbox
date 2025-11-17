import torch
import torch.nn.functional as F

def get_mass_mean_vectors(constrastive_pairs: torch.Tensor, normalize: bool = True, keepdim: bool = False) -> torch.Tensor:
    """Returns the mass mean vector of a batch of contrastive pairs.

    Args:
        constrastive_pairs (torch.Tensor): Contrastive sample pairs (batch_dim, 2, num_samples, emb_dim)
        normalize (bool, optional): Returns normalized vector. Defaults to True.

    Returns:
        torch.Tensor: Mass mean vector.
    """
    num_homographs, num_examples = constrastive_pairs.size(0), constrastive_pairs.size(2)
    mean_vec_1, mean_vec_2 = constrastive_pairs.mean(-2).transpose(0, 1)
    mass_mean_vec = mean_vec_1 - mean_vec_2

    if normalize:
        mass_mean_vec = F.normalize(mass_mean_vec, dim=-1)

    if keepdim:
        mass_mean_vec = mass_mean_vec[ :, None, None, :].repeat(1, 2, num_examples, 1)
        mass_mean_vec * torch.tensor([1, -1])[:, None, None]
        
    return mass_mean_vec