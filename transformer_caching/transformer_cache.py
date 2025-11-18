import torch
from transformers import PretrainedConfig, AutoConfig

from mi_toolbox.utils.collate.utils import max_pad_sequence
from mi_toolbox.utils.data_types import DataDict

class TransformerCache(DataDict):

    def __init__(self, model_config: PretrainedConfig = None, model_id: str = None) -> None:
        self.model_config = model_config
        if model_id and not model_config:
            self.model_config = AutoConfig.from_pretrained(model_id)
        super().__init__()

    def stack_tensors(self, padding=False) -> None:
        for key, value in self.default_entry.items():
            if isinstance(value, torch.Tensor):
                if padding:
                    self[key] = max_pad_sequence(self[key])
                else:
                    self[key] = torch.stack(self[key])
                    