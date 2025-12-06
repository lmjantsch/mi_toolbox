from typing import Union, List, Generator, Tuple, Dict, Iterator
import torch
import time
import gc
from contextlib import contextmanager
from torch.utils.data import DataLoader
from nnsight import LanguageModel
from transformers import PretrainedConfig

from .transformer_cache import TransformerCache


@contextmanager
def load_nnsight_model(
    model_id: Union[str, List[str]],
    dtype: torch.dtype=torch.bfloat16,
    device: Union[str, torch.device]='auto',
) -> Generator[Tuple[LanguageModel, PretrainedConfig], None, None]:
    llm = LanguageModel(
        model_id,
        trust_remote_code=True,
        device_map=device,
        dtype=dtype,
        attn_implementation = 'eager',
        dispatch=True
    )
    try:
        yield llm, llm.config
    except Exception as e:
        print('loader')
        raise e
    finally:
        llm.cpu()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def batch_iterator(dl: DataLoader) -> Iterator[Dict, ]:
    for batch_id, batch in enumerate(dl):
        start_time = time.perf_counter()
        try:
            yield batch
        except Exception as e:
            print('iterator')
            raise e
        finally:
            end_time = time.perf_counter()
            # print(f"Batch {batch_id + 1}/{len(dl)}: {(end_time - start_time):.2f} seconds")


@torch.no_grad()
def caching_wrapper(
        model_ids: Union[str, List[str]],
        dl: DataLoader,
        caching_function: callable,
        dtype: torch.dtype=torch.bfloat16,
        device: Union[str, torch.device]='auto',
):
    big_cache = {}
    if isinstance(model_ids, str):
        model_ids = [model_ids]

    for model_id in model_ids:
        with load_nnsight_model(model_id, dtype, device) as (llm, config):
            model_cache = TransformerCache(config)

            for batch in batch_iterator(dl):
                batch_cache = caching_function(llm, config, batch)
                model_cache.extend(batch_cache)

            big_cache[model_id] = model_cache

    return big_cache