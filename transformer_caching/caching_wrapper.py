from typing import Union, List, Generator, Tuple, Dict, Iterator, Any, Callable, Optional
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..utils.data_types import TransformerCache
from .model_wrapper import load_nnsight_model




@torch.no_grad()
def caching_wrapper(
    model_ids: Union[str, List[str]],
    dl: DataLoader,
    caching_function: Callable,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = 'auto',
    loader: Optional[Callable] = None,
    show_progress: bool = True,
    **loader_kwargs
) -> Dict[str, Any]:
    """
    Wrapper to iterate over models, process data, and cache results.
    
    Args:
        model_ids: Single model ID or list of model IDs.
        dl: DataLoader providing the batches.
        caching_function: Function (model, config, batch) -> dict of data to cache.
        dtype: Model data type.
        device: Model device.
        loader: Context manager for loading models. Defaults to load_nnsight_model.
        show_progress: Whether to show tqdm progress bar.
        **loader_kwargs: Additional args passed to the loader (e.g. causal=False).
    """
    if loader is None:
        loader = load_nnsight_model

    big_cache = {}
    if isinstance(model_ids, str):
        model_ids = [model_ids]

    for model_id in model_ids:
        try:
            print(f"Processing model: {model_id}")
            with loader(model_id, dtype=dtype, device=device, **loader_kwargs) as (llm, config):
                model_cache = TransformerCache(model_config=config)

                iterator = tqdm(
                    dl, 
                    disable=not show_progress, 
                    desc=f"Caching {model_id.split('/')[-1]}"
                )

                for batch in iterator:
                    try:
                        batch_cache = caching_function(llm, config, batch)
                        model_cache.extend(batch_cache)
                    except Exception as e:
                        print(f"Error processing batch for model {model_id}: {e}")
                        raise e

                big_cache[model_id] = model_cache

        except Exception as e:
            print(f"Failed to process model {model_id}. Skipping. Error: {e}")
            continue

    return big_cache