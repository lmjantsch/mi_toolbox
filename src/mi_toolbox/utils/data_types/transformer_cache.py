import torch
from transformers import PretrainedConfig, AutoConfig
from typing import Optional, List

from ..collate.utils import max_pad_sequence
from .data_dict import DataDict

class TransformerCache(DataDict):

    def __init__(self, model_config: Optional[PretrainedConfig] = None, model_id: str = None, columns: Optional[List[str]] = None) -> None:
        self.model_config = model_config
        if model_id and not model_config:
            self.model_config = AutoConfig.from_pretrained(model_id)
            
        super().__init__(columns=columns)

    def stack_tensors(self, padding: bool = False) -> None:
        """
        Iterates over columns. If a column is a list of Tensors, it stacks them.
        If padding is True, it pads them to the max shape before stacking.
        """
        for key in list(self.keys()):
            value = self.data[key]
            
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                if padding:
                    self.data[key] = max_pad_sequence(value)
                else:
                    self.data[key] = torch.stack(value)

    def _save(self, path: str) -> None:
        """
        Overrides DataDict._save to also save the transformer config.
        """
        super()._save(path)
        
        if self.model_config:
            self.model_config.save_pretrained(path)

    @classmethod
    def load(cls, path: str, keys: List[str]) -> 'TransformerCache':
        """
        Loads the TransformerCache from disk, including the config.
        """
        config = None
        try:
            config = AutoConfig.from_pretrained(path)
        except (OSError, ValueError):
            print(f"Warning: No valid transformer config found at {path}")

        obj = cls(model_config=config)
        
        loaded_data = cls._load(path, keys)
        if loaded_data:
            obj._add_data(loaded_data)
            
        return obj