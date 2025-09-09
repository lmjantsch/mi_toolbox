import torch
from torch import Tensor
from ..base_classes import FeatureProcessor, FeatureCache
from typing import Dict, Union, List, Any, Optional, Callable
    
class AttentionProcessor(FeatureProcessor):
    """
    Processes attention weights from a FeatureCache.

    This processor can select specific layers, heads, and tokens, and then
    aggregate the resulting attention maps using a specified method (e.g., mean, max).
    """
    
    def __init__(self, 
                 tokenizer: Any, 
                 target_layers: Optional[Union[int, slice, List[int]]] = None, 
                 target_heads: Optional[Union[int, slice, List[int]]] = None, 
                 target_tokens: Optional[Union[int, slice, List[int]]] = None, 
                 aggregation_method: Optional[str] = 'mean', **kwargs):
        """
        Initializes the AttentionProcessor.

        Args:
            tokenizer: A Hugging Face tokenizer instance.
            target_layers: The layer(s) to select. Can be an int, slice, or list of ints.
                           If None, all layers are used.
            target_heads: The head(s) to select. Can be an int, slice, or list of ints.
                          If None, all heads are used.
            target_tokens: The token(s) to select. Can be an int, slice, or list of ints.
                           If None, all tokens from the 'task_mask' are used.
            aggregation_method: How to aggregate attentions across selected layers/heads.
                                Options: 'mean', 'max', or None (no aggregation).
        """
        super().__init__(tokenizer)
        self.target_layers_spec = target_layers
        self.target_heads_spec = target_heads
        self.target_tokens_spec = target_tokens
        self.aggregation_method = aggregation_method
        
        self.aggregation_mapping: Dict[str, Callable] = {
            'mean': lambda x, dim: torch.mean(x, dim=dim, keepdim=True),
            'max': lambda x, dim: torch.max(x, dim=dim, keepdim=True).values
        }
        self.aggregation_fn = self.aggregation_mapping.get(aggregation_method) if aggregation_method else None
            
    @staticmethod
    def _create_target_tensor(target_spec: Optional[Union[int, slice, List[int]]], max_val: int) -> Union[Tensor, List[Tensor]]:
        """Converts a flexible target specification into a 1D tensor of indices."""
        if target_spec is None:
            return [torch.arange(max_val)]
        if isinstance(target_spec, int):
            return torch.tensor([target_spec])
        if isinstance(target_spec, slice):
            start = target_spec.start or 0
            stop = target_spec.stop or max_val
            step = target_spec.step or 1
            return torch.arange(start, stop, step)
        if isinstance(target_spec, list) and all([isinstance(item, int) for item in target_spec]):
            return torch.tensor(target_spec)
        if isinstance(target_spec, list):
            group_specs = []
            for item in target_spec:
                if isinstance(item, int):
                    group_specs.append(torch.tensor([item]))
                elif isinstance(item, slice):
                    start = item.start or 0
                    stop = item.stop or max_val
                    step = item.step or 1
                    group_specs.append(torch.arange(start, stop, step))
                elif isinstance(item, list) and all([isinstance(el, int) for el in item]):
                    group_specs.append(torch.tensor(item))
                else:
                    raise TypeError(f"Unsupported element type in list: {type(item)}")
            return group_specs
        else:
            raise TypeError(f"Unsupported target type: {type(target_spec)}")
        
    def _select_target_tensor(self, cache_tensor, dim, target_tensor):
        
        if isinstance(target_tensor, Tensor):
            return torch.index_select(cache_tensor, dim, target_tensor)
        
        selcted_cache = []
        if not self.aggregation_fn:
            raise ValueError('The aggregation function must be defined.')
        for target_group in target_tensor:
            selcted_cache.append(
                self.aggregation_fn(
                  torch.index_select(cache_tensor, dim, target_group)   
                , dim)
            )
        return torch.cat(selcted_cache, dim=dim)
       
    def process(self, inputs: Dict[str, Tensor], residual_cache: FeatureCache) -> Dict[str, Any]:
        """
        Selects, aggregates, and formats attention weights for plotting.
        """
        # 1. Prepare the full attention tensor
        all_attentions = torch.stack(list(residual_cache.attentions.values())).squeeze(1) #  (num_layers, num_heads, seq_len, seq_len)
        if len(all_attentions.shape) != 4:
            raise NotImplementedError
        
        # 2. Resolve target indices
        target_layers = self._create_target_tensor(self.target_layers_spec, residual_cache.num_layers)
        target_heads = self._create_target_tensor(self.target_heads_spec, residual_cache.num_heads)
        
        # Determine target tokens from the task mask if not specified
        if self.target_tokens_spec is None:
            # Assuming a single sample for token selection if batch size > 1 -> TODO
            # Use the mask of the first sample to determine the token indices
            task_mask_for_tokens = inputs['task_mask'][0]
            target_tokens = torch.where(task_mask_for_tokens)[0]
        else:
            seq_len = all_attentions.shape[-1]
            target_tokens = self._create_target_tensor(self.target_tokens_spec, seq_len)

        # 3. Perform selection and aggregation with vectorized operations

        selected_attns = self._select_target_tensor(all_attentions, 0, target_layers)
        selected_attns = self._select_target_tensor(selected_attns, 1, target_heads)
        
        # Aggregate across layers and heads if an aggregation function is specified
        final_matrix = selected_attns[:, :, target_tokens, :][:, :, :, target_tokens] # (agg_layer, agg_heads, from_token, to_token)
        attn_mask = m = torch.tril(torch.ones_like(final_matrix))
        final_matrix[attn_mask == 0] = torch.nan
        
        # 4. Prepare output for the plotter
        attn_matrix_np = final_matrix.cpu().float().numpy()

        # Create Axies Labels
        if isinstance(target_heads, Tensor):
            head_labels = target_heads.tolist()
        else:
            head_labels = [f"{self.aggregation_method}({group[0]}~{group[-1]})" for group in target_heads]
            
        if isinstance(target_layers, Tensor):
            layer_labels = target_layers.tolist()
        else:
            layer_labels = [f"{self.aggregation_method}({group[0]}~{group[-1]})" for group in target_layers]
               
            
            
        # Create token labels for plot ticks
        tick_labels = self._create_labels_from_ids(inputs['input_ids'][0], inputs['task_mask'][0])
        # filtered_labels = [tick_labels[i] for i in range(len(tick_labels)) if torch.any(target_tokens == i)]

        # The generic output dictionary, decoupled from any specific plotter
        return {
            "data": {
                "data_matrix": attn_matrix_np
            },
            "config": {
                "title": "Attention Map",
                "x_ticks": tick_labels,
                "y_ticks": tick_labels,
                "x_labels": [f"Head: {h}" for h in head_labels],
                "y_labels": [f"Layer: {l}" for l in layer_labels],
                "rotate_x_labels_degree": 45
            }
        }
