import torch
import gc
from contextlib import contextmanager
from typing import Union, List, Generator, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoModel, PretrainedConfig
from nnsight import LanguageModel

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