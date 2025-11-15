from transformers import AutoTokenizer
from typing import List
class Collator:

    def __init__(self, tokenizer: AutoTokenizer, fn:List=[]):
        self.tokenizer = tokenizer
        self.fn = fn

    def __call__(self, batch):
        transposed_batch_values = zip(*[d.values() for d in batch])
        batch_dict = {key: list(value) for key, value in zip(batch[0].keys(), transposed_batch_values)}

        batch_out = self.tokenizer(
            batch_dict['prompts'], 
            return_tensors='pt', 
            padding=True,
        )

        for f in self.fn:
            batch_out = batch_out | f(batch_dict)
        return batch_out