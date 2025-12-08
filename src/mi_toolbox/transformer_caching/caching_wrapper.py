from typing import Union, List, Generator, Tuple, Dict, Any, Callable, Optional
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import gc
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoModel, PretrainedConfig
from nnsight import LanguageModel

from ..utils.data_types import TransformerCache




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

@contextmanager
def load_nnsight_model(
    model_id: Union[str, List[str]],
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = 'auto',
) -> Generator[Tuple['LanguageModel', PretrainedConfig], None, None]:
    """
    Context manager for loading an NNsight LanguageModel with automatic cleanup.
    """
    llm = LanguageModel(
        model_id,
        trust_remote_code=True,
        device_map=device,
        dtype=dtype,
        attn_implementation='eager',
        dispatch=True
    )
    
    try:
        yield llm, llm.config
    except Exception as e:
        print(f"Error in load_nnsight_model: {e}")
        raise e
    finally:
        if hasattr(llm, 'cpu'):
            llm.cpu()
        
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


@contextmanager
def load_hf_model(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = 'auto',
    causal: bool = True,
    **kwargs
) -> Generator[Tuple[Union[AutoModelForCausalLM, AutoModel], PretrainedConfig], None, None]:
    """
    Context manager for loading a standard Hugging Face model with automatic cleanup.
    
    Args:
        model_id: The Hugging Face model ID.
        dtype: Data type for loading.
        device: Device map (e.g., 'auto', 'cuda', 'cpu').
        causal: If True, uses AutoModelForCausalLM. If False, uses AutoModel.
        **kwargs: Additional arguments passed to from_pretrained.
    """
    model_class = AutoModelForCausalLM if causal else AutoModel
    
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation='eager',
        **kwargs
    )
    
    try:
        yield model, model.config
    except Exception as e:
        print(f"Error in load_hf_model: {e}")
        raise e
    finally:
        model.to('cpu')
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")