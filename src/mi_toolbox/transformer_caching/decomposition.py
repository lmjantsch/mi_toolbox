import torch
from typing import Tuple, List

def decompose_attention_to_head(
            attn_weight: torch.Tensor, 
            v_proj: torch.Tensor, 
            o_proj_WT: torch.Tensor,
            num_attention_heads: int,
            num_key_value_heads: int,
            head_dim: int,
            batch_token_index: Tuple[List[int], List[int]] = None
    ) -> torch.Tensor:
    """
    Decomposes attention output into contributions from each head.
    
    Args:
        attn_weight: (bs, num_heads, q_pos, k_pos)
        v_proj: (bs, k_pos, num_kv_heads * head_dim)
        o_proj_WT: (num_heads * head_dim, model_dim) - Assumed to be (In, Out)
    """
    batch_size = v_proj.size(0)
    num_head_groups = num_attention_heads // num_key_value_heads

    # Reshape v_poj to k_v_head view (bs, k_pos, num_k_v_heads, head_dim)
    v_proj = v_proj.view(batch_size, -1, num_key_value_heads, head_dim)
    # Reshape o_proj to head view (num_heads, head_dim, model_dim)
    o_proj_WT = o_proj_WT.view(num_attention_heads, head_dim, -1)
    
    if num_head_groups > 1:
        # Repeat v_poj for number of head groups (bs, k_pos, num_heads, head_dim)
        v_proj = v_proj.repeat_interleave(num_head_groups, -2)
        
    if batch_token_index:
        batch_idx, token_idx = batch_token_index
        # select target (bs*, k_pos, num_heads, head_dim)
        attn_weight = attn_weight[batch_idx, : ,token_idx]
        v_proj = v_proj[batch_idx]
        
        z = torch.einsum('bkhd,bhk -> bhkd', v_proj, attn_weight)
        decomposed_attn = torch.einsum('bhkd,hdm -> bkhm', z, o_proj_WT)

        return decomposed_attn
        
    z = torch.einsum('bkhd,bhqk -> bhqkd', v_proj, attn_weight)
    decomposed_attn = torch.einsum('bhqkd,hdm -> bqkhm',z, o_proj_WT)
    
    return decomposed_attn

def decompose_attention_to_neuron(
            attn_weight: torch.Tensor, 
            v_proj: torch.Tensor, 
            o_proj_WT: torch.Tensor,
            num_attention_heads: int,
            num_key_value_heads: int,
            head_dim: int,
            batch_token_index: Tuple[List[int], List[int]] = None
    ) -> torch.Tensor:
    """
    Decomposes attention output into contributions from each neuron (head dimension features).

    Args:
        attn_weight: (bs, num_heads, q_pos, k_pos)
        v_proj: (bs, k_pos, num_kv_heads * head_dim)
        o_proj_WT: (num_heads * head_dim, model_dim) - Assumed to be (In, Out)
    """
    batch_size = v_proj.size(0)
    num_head_groups = num_attention_heads // num_key_value_heads

    # Reshape v_poj to k_v_head view (bs, k_pos, num_k_v_heads, head_dim)
    v_proj = v_proj.view(batch_size, -1, num_key_value_heads, head_dim)
    # Reshape o_proj to head view (num_heads, head_dim, model_dim)
    o_proj_WT = o_proj_WT.view(num_attention_heads, head_dim, -1)
    
    if num_head_groups > 1:
        # Repeat v_poj for number of head groups (bs, k_pos, num_heads, head_dim)
        v_proj = v_proj.repeat_interleave(num_head_groups, -2)
        
    if batch_token_index:
        batch_idx, token_idx = batch_token_index
        # select target (bs*, k_pos, num_heads, head_dim)
        attn_weight = attn_weight[batch_idx, : ,token_idx]
        v_proj = v_proj[batch_idx]
        
        z = torch.einsum('bkhd,bhk -> bhkd', v_proj, attn_weight)
        decomposed_attn = torch.einsum('bhkd,hdm -> bkhdm', z, o_proj_WT)

        return decomposed_attn
        
    z = torch.einsum('bkhd,bhqk -> bhqkd', v_proj, attn_weight) 
    decomposed_attn = torch.einsum('bhqkd,hdm -> bqkhdm',z, o_proj_WT)

    return decomposed_attn

def decompose_glu_to_neuron(
    down_proj_WT: torch.Tensor, 
    act_prod: torch.Tensor = None,
    batch_token_index: Tuple[List[int], List[int]] = None
    ):
    """
    Decomposes GLU/MLP output into contributions from each intermediate neuron.
    
    Args:
        down_proj_WT: (intermediate_dim, model_dim)
        act_prod: (bs, seq_len, intermediate_dim) - This is (Gate * Up) output
    """
    if batch_token_index:
        batch_idx, token_idx = batch_token_index
        act_prod = act_prod[batch_idx, token_idx]

        decomposed_glu = torch.einsum('bh,hm->bhm', act_prod, down_proj_WT)

        return decomposed_glu
    
    decomposed_glu = torch.einsum('bth,hm->bthm', act_prod, down_proj_WT)
    
    return decomposed_glu
