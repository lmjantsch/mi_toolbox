from typing import Optional, Dict, Any
import torch
from torch import Tensor
from dataclasses import dataclass, asdict, field, is_dataclass
from abc import ABC, abstractmethod
import collections.abc

from typing import Optional, Dict, Any
import torch
from torch import Tensor
from dataclasses import dataclass, asdict, field, is_dataclass
from abc import ABC, abstractmethod
import collections.abc

@dataclass
class FeatureCache:
    """
    A dataclass to store cached features from a model's forward pass.

    Attributes:
        hidden_states (Dict[int, Tensor]): A dictionary mapping layer index to hidden state tensors.
        attentions (Dict[int, Tensor]): A dictionary mapping layer index to attention tensors.
        num_layers (Optional[int]): The total number of layers in the model.
        num_heads (Optional[int]): The number of attention heads per layer.
    """
    hidden_states: Dict[int, Tensor] = field(default_factory=dict)
    attentions: Dict[int, Tensor] = field(default_factory=dict)
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None

@dataclass
class PlotData(ABC):
    """An abstract dataclass to hold the data required for plotting."""
    pass

@dataclass
class PlotConfig(ABC):
    """An abstract dataclass to hold the configuration for a plot."""
    title: Optional[str] = None
    
    @staticmethod
    def _deep_merge_dicts(base_dict: dict, override_dict: dict) -> dict:
        """
        Recursively merges two dictionaries.
        Values in override_dict take precedence.
        """
        merged = base_dict.copy()
        for key, value in override_dict.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, collections.abc.Mapping):
                merged[key] = PlotConfig._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def merge(cls, base_config: 'PlotConfig', override_config: 'PlotConfig') -> 'PlotConfig':
        """
        Performs a deep merge of two configuration objects.

        This allows for nested configuration values to be updated without
        overwriting the entire nested structure.

        Args:
            base_config (PlotConfig): The base configuration.
            override_config (PlotConfig): The configuration with overriding values.

        Returns:
            PlotConfig: The new, merged configuration object.
        """
        if not is_dataclass(base_config) or not is_dataclass(override_config):
            raise TypeError("Both configs must be dataclass instances.")
            
        base_dict = asdict(base_config)
        # Only consider non-None values from the override config to allow for partial updates
        override_dict = {k: v for k, v in asdict(override_config).items() if v is not None}
        
        merged_dict = cls._deep_merge_dicts(base_dict, override_dict)
        return cls(**merged_dict)

class FeatureProcessor(ABC):
    """
    Abstract base class for feature processors.
    Operates on PyTorch Tensors to extract and aggregate meaningful features.
    """
    def __init__(self, tokenizer: Optional[Any] = None, **kwargs):
        """
        Initializes the feature processor.

        Args:
            tokenizer (Optional[Any]): A tokenizer instance (e.g., from Hugging Face)
                                       required for any text-based processing.
        """
        self.tokenizer = tokenizer
        self.params = kwargs
        
    @staticmethod     
    def _create_broadcastable_index(*args: Tensor) -> tuple[Tensor, ...]:
        """
        Converts a sequence of 1D tensors into a format suitable for advanced indexing.
        This allows for selecting elements from a high-dimensional tensor using indices 
        from multiple dimensions simultaneously.
        
        Example:
            # Selects elements from a 4D tensor using 1D index tensors
            tensor[FeatureProcessor._create_broadcastable_index(layer_idx, head_idx, token_idx1, token_idx2)]
        """
        out = []
        ndim = len(args)
        for i, v in enumerate(args):
            v = torch.as_tensor(v)
            if v.ndim != 1:
                raise ValueError("All inputs for indexing must be 1D tensors.")
            # Reshape each index tensor to broadcast correctly against the others
            # e.g., for 3 args: (N, 1, 1), (1, M, 1), (1, 1, P)
            shape = [1] * ndim
            shape[i] = -1
            out.append(v.view(*shape))
        return tuple(out)
    
    def _create_labels_from_ids(self, input_ids: Tensor, label_mask: Tensor) -> list[str]:
        """Creates token labels from input IDs, applying a mask."""
        if self.tokenizer is None:
            raise ValueError("A tokenizer must be provided to the processor to create labels.")
        
        # Ensure tensors are on the CPU for decoding and iteration
        input_ids = input_ids.cpu()
        label_mask = label_mask.cpu()
        
        tokens = [self.tokenizer.decode(token_id) 
                  for token_id, include in zip(input_ids, label_mask) if include]
        
        # Replace standard space with a visible space character for better plotting
        return [token.replace(' ', ' ') for token in tokens]

    @abstractmethod
    def process(self, inputs: Dict[str, Tensor], residual_cache: FeatureCache) -> Dict[str, Any]:
        """
        Process high-dimensional features from the residual cache into a generic dictionary.

        This method is decoupled from any specific plotting library. It should return a
        dictionary containing all necessary data (e.g., matrices, labels, metadata)
        that a FeaturePlotter can then use to generate a visualization.

        Args:
            inputs (Dict[str, Tensor]): The input dict from the tokenizer.
            residual_cache (FeatureCache): The cached latent features from the model.

        Returns:
            Dict[str, Any]: A dictionary containing the processed data and metadata.
        """
        pass

class FeaturePlotter(ABC):
    """
    Abstract base class for plotting processed features.
    Subclasses must implement the 'plot' method.
    """
    
    @abstractmethod
    def plot(self, plot_data: PlotData, plot_config: PlotConfig):
        """
        Generates a plot of the features using dataclass inputs.

        Args:
            plot_data (PlotData): The data to be plotted.
            plot_config (PlotConfig): The configuration for the plot.
        """
        pass