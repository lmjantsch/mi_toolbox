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
    batch_size = v_proj.size(0)
    num_head_groups = num_attention_heads // num_key_value_heads

    v_proj = v_proj.view(batch_size, -1, num_key_value_heads, head_dim) # bs, k_pos, num_k_v_heads, head_dim
    o_proj_WT = o_proj_WT.reshape(num_attention_heads, head_dim, -1) # num_heads, head_dim, model_dim
    
    if num_head_groups > 1:
        v_proj = v_proj.repeat_interleave(num_head_groups, -2) # bs, k_pos, num_heads, head_dim
        
    if batch_token_index:
        batch_idx, token_idx = batch_token_index
        attn_weight = attn_weight[batch_idx, : ,token_idx] # bs, num_heads, k_pos
        
        z = torch.einsum('bkhd,bhk -> bhkd', v_proj, attn_weight) # bs, num_heads, k_pos, head_dim
        decomposed_attn = torch.einsum('bhkd,hdm -> bkhm', z, o_proj_WT) # bs, k_pos, num_heads, model_dim
        return decomposed_attn
        
    z = torch.einsum('bkhd,bhqk -> bhqkd', v_proj, attn_weight) # bs, num_heads, q_pos, k_pos, head_dim
    decomposed_attn = torch.einsum('bhqkd,hdm -> bqkhm',z, o_proj_WT) # bs, q_pos, k_pos, num_heads, model_dim
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
    batch_size = v_proj.size(0)
    num_head_groups = num_attention_heads // num_key_value_heads

    v_proj = v_proj.view(batch_size, -1, num_key_value_heads, head_dim) # bs, k_pos, num_k_v_heads, head_dim
    o_proj_WT = o_proj_WT.reshape(num_attention_heads, head_dim, -1) # num_heads, head_dim, model_dim
    
    if num_head_groups > 1:
        v_proj = v_proj.repeat_interleave(num_head_groups, -2) # bs, k_pos, num_heads, head_dim
        
    if batch_token_index:
        batch_idx, token_idx = batch_token_index
        attn_weight = attn_weight[batch_idx, : ,token_idx] # bs, num_heads, k_pos
        
        z = torch.einsum('bkhd,bhk -> bhkd', v_proj, attn_weight) # bs, num_heads, k_pos, head_dim
        decomposed_attn = torch.einsum('bhkd,hdm -> bkhdm', z, o_proj_WT) # bs, k_pos, num_heads, head_dim, model_dim
        return decomposed_attn
        
    z = torch.einsum('bkhd,bhqk -> bhqkd', v_proj, attn_weight) # bs, num_heads, q_pos, k_pos, head_dim
    decomposed_attn = torch.einsum('bhqkd,hdm -> bqkhdm',z, o_proj_WT) # bs, q_pos, k_pos, num_heads, head_dim, model_dim
    return decomposed_attn

def decompose_glu_to_neuron(
    down_proj_WT: torch.Tensor, 
    act_prod: torch.Tensor = None,
    batch_token_index: Tuple[List[int], List[int]] = None
    ):
    if batch_token_index:
        act_prod = act_prod[batch_token_index]
        decomposed_glu = torch.einsum('bh,hm->bhm', act_prod, down_proj_WT)
        return decomposed_glu
    
    decomposed_glu = torch.einsum('bth,hm->bthm', act_prod, down_proj_WT)
    return decomposed_glu
