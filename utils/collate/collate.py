from transformers import AutoTokenizer
from typing import List, Dict
import torch

from .utils import max_pad_sequence

class Collator:

    def __init__(self, collate_fn: Dict[str, callable] = None):
        self.collate_fn = collate_fn

    def __call__(self, batch:List[Dict]) -> Dict:
        keys = batch[0].keys()
        batch_out = {}

        for key in keys:
            if self.collate_fn and key in self.collate_fn:
                batch_out |= self.collate_fn[key](key, [row[key] for row in batch])
                continue
            batch_out |= {key: [row[key] for row in batch]}

        return batch_out | {'length': len(batch)}
    

class TensorCollator(Collator):

    def __init__(self, collate_fn: Dict[str, callable] = None, padding: bool = False):
        self.padding = padding
        super().__init__(collate_fn)

    def __call__(self, batch:List[Dict]) -> Dict:
        example_items = batch[0].items()
        batch_out = {}

        for key, example_value in example_items:
            if self.collate_fn and key in self.collate_fn:
                batch_out |= self.collate_fn[key](key, [row[key] for row in batch])
                continue
            if isinstance(example_value, torch.Tensor):
                if self.padding:
                    batch_out |= {key: max_pad_sequence([row[key] for row in batch])}
                    continue
                batch_out |= {key: torch.stack([row[key] for row in batch])}
                continue
            batch_out |= {key: [row[key] for row in batch]}

        return batch_out | {'length': len(batch)}

        

class TokenizeCollator(Collator):

    def __init__(self, tokenizer: AutoTokenizer, collate_fn: Dict[str, callable] = None):
        self.tokenizer = tokenizer
        super().__init__(collate_fn)

    def __call__(self, batch:List[Dict]) -> Dict:
        keys = batch[0].keys()
        batch_out = {}

        for key in keys:
            if self.collate_fn and key in self.collate_fn:
                batch_out |= self.collate_fn[key](key, [row[key] for row in batch])
                continue
            if key == 'prompts':
                batch_out |= self.tokenizer(
                    [row['prompts'] for row in batch], 
                    return_tensors='pt', 
                    padding=True,
                )
            batch_out |= {key: [row[key] for row in batch]}

        return batch_out | {'length': len(batch)}