import torch
from torch import Tensor
from typing import Optional, Union, List, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_classes import FeaturePlotter, FeatureProcessor, PlotConfig, FeatureCache, PlotData
from .plot_registry import PLOT_REGISTRY

class FeatureExtractor:
    """
    Handles model and tokenizer loading, and the extraction of raw
    features (hidden states, attentions) from a forward pass.
    """
    def __init__(self, model_name: Optional[str] = None, model: Optional[AutoModelForCausalLM] = None, 
                 tokenizer: Optional[AutoTokenizer] = None, batch_size: int = 8):
        """
        Initializes the extractor with a model and tokenizer.

        Args:
            model_name (str, optional): Name of a Hugging Face model to load (e.g., 'gpt2').
            model (AutoModelForCausalLM, optional): An already loaded transformer model.
            tokenizer (AutoTokenizer, optional): An already loaded tokenizer.
            batch_size (int): The batch size for processing inputs during feature extraction.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._initialize_model_and_tokenizer()

    def _initialize_model_and_tokenizer(self):
        """Validates or loads the model and tokenizer."""
        if self.model_name and (self.model is None or self.tokenizer is None):
            self._load_model_and_tokenizer()
        elif self.model and self.tokenizer:
            self._configure_model_and_tokenizer()
        else:
            raise ValueError("Either 'model_name' or both 'model' and 'tokenizer' must be provided.")

    def _configure_model_and_tokenizer(self):
        """Configures the provided model and tokenizer for feature extraction."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ensure model is configured to output necessary features
        self.model.config.output_attentions = True
        self.model.config.output_hidden_states = True
        self.model.config.return_dict_in_generate = True
        
        self.model.to(self.device)
        self.model.eval()
        print("Using provided model and tokenizer.")

    def _load_model_and_tokenizer(self):
        """Loads model and tokenizer from Hugging Face Hub."""
        print(f"Loading model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
                attn_implementation="eager", # Necessary for accessing attention weights
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            self.model.to(self.device)
            self.model.eval()
            print("Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def extract_features(self, inputs: Dict[str, Tensor]) -> FeatureCache:
        """
        Performs a forward pass and extracts hidden states and attentions.
        """
        num_samples = inputs['input_ids'].size(0)
        all_outputs = []
        
        try:
            with torch.no_grad():
                for i in range(0, num_samples, self.batch_size):
                    batch_inputs = {k: v[i:i+self.batch_size].to(self.device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
                    outputs = self.model(**batch_inputs)
                    # Move outputs to CPU immediately to free up GPU memory
                    all_outputs.append({
                        'hidden_states': [h.cpu() for h in outputs.hidden_states],
                        'attentions': [a.cpu() for a in outputs.attentions]
                    })
        except Exception as e:
            print(f"An error occurred during feature extraction: {e}")
            raise

        # Consolidate features from all batches
        feature_cache = FeatureCache(
            hidden_states={i: torch.cat([out['hidden_states'][i] for out in all_outputs], dim=0) for i in range(len(all_outputs[0]['hidden_states']))},
            attentions={i: torch.cat([out['attentions'][i] for out in all_outputs], dim=0) for i in range(len(all_outputs[0]['attentions']))},
            num_layers=self.model.config.num_hidden_layers,
            num_heads=self.model.config.num_attention_heads,
        )
        return feature_cache


class VisualizationPipe:
    """
    Orchestrates the visualization pipeline by separating feature loading
    from plotting for efficient, iterative analysis.
    """
    def __init__(self, 
                 extractor: FeatureExtractor, 
                 inputs: Optional[Dict[str, Tensor]] = None, 
                 feature_cache: Optional[FeatureCache] = None):
        
        self.extractor = extractor
        self.tokenizer = extractor.tokenizer
        self.inputs: Optional[Dict[str, Tensor]] = inputs
        self.feature_cache: Optional[FeatureCache] = feature_cache

    def _encode_prompt(self, tasks: Union[str, List[str]], contexts: Optional[Union[str, List[str]]] = None) -> Dict[str, Tensor]:
        """Encodes prompts and creates a mask for the task-specific tokens."""
        if isinstance(tasks, str):
            tasks = [tasks]
        
        if contexts is None:
            prompts = tasks
            context_lengths = [0] * len(tasks)
        elif isinstance(contexts, str):
            prompts = [contexts + task for task in tasks]
            context_lengths = [len(contexts)] * len(tasks)
        elif isinstance(contexts, list):
            if len(tasks) != len(contexts):
                raise ValueError(f"Mismatched number of tasks ({len(tasks)}) and contexts ({len(contexts)}).")
            prompts = [ctx + task for ctx, task in zip(contexts, tasks)]
            context_lengths = [len(ctx) for ctx in contexts]
        else:
            raise TypeError("Contexts must be a string, a list of strings, or None.")

        inputs = self.tokenizer(
            prompts, 
            return_attention_mask=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Efficiently create task mask
        task_mask = torch.zeros_like(inputs['input_ids'], dtype=torch.bool)
        for i, offset_mapping in enumerate(inputs['offset_mapping']):
            context_len = context_lengths[i]
            # Find the first token that starts at or after the end of the context
            task_start_token_idx = next((idx for idx, (start, end) in enumerate(offset_mapping) if end >= context_len and end > start), None)
            if task_start_token_idx is not None:
                task_mask[i, task_start_token_idx:] = True
        
        # Mask out padding tokens
        task_mask = task_mask & inputs['attention_mask'].bool()
        
        inputs['task_mask'] = task_mask
        del inputs['offset_mapping']
        
        return inputs

    def load(self, tasks: Union[str, List[str]], contexts: Optional[Union[str, List[str]]] = None):
        """
        Encodes prompts and extracts features, caching them for later use.
        This is the computationally expensive step.
        """
        print("Encoding prompts and extracting features...")
        self.inputs = self._encode_prompt(tasks, contexts)
        self.feature_cache = self.extractor.extract_features(self.inputs)
        print("Features loaded and cached successfully.")
        
    def plot(self, plot_key: str, custom_config: Optional['PlotConfig']=None, **kwargs):
        """
        Processes and plots cached features using a predefined plot configuration.

        Args:
            plot_key (str): The string identifier of the plot to generate.
            **kwargs: Arbitrary keyword arguments to customize the plot. These can
                      be consumed by the FeatureProcessor, FeaturePlotter, or used
                      to override the final PlotConfig.
        """
        # 1. Pre-flight checks
        if self.feature_cache is None:
            raise RuntimeError("You must call the 'load' method to extract features before plotting.")
        
        if plot_key not in PLOT_REGISTRY:
            raise ValueError(f"Plot key '{plot_key}' not recognized. Available plots: {list(PLOT_REGISTRY.keys())}")

        # 2. Retrieve plot components from the registry
        ProcessorClass, PlotterClass, DataClass, ConfigClass = PLOT_REGISTRY[plot_key]
        
        pipe_kwargs = {
            'tokenizer': self.tokenizer
        }
        kwargs = pipe_kwargs | kwargs

        # 3. Instantiate components, passing kwargs for their specific configuration
        try:
            feat_processor = ProcessorClass(**kwargs)
            feat_plotter = PlotterClass(**kwargs)
        except TypeError as e:
            print(f"Error initializing plot components for '{plot_key}'. Check your keyword arguments.")
            raise e
        
        self._process_plot(feat_processor, feat_plotter, DataClass, ConfigClass, custom_config)


    def _process_plot(self, feat_processor: 'FeatureProcessor', feat_plotter: 'FeaturePlotter',
             plot_data_class: type[PlotData], plot_config_class: type[PlotConfig],
             custom_config: Optional['PlotConfig']=None):
        """
        Processes and plots the cached features. Can be called multiple times
        after a single call to 'load'.
        """
        if self.inputs is None or self.feature_cache is None:
            raise RuntimeError("You must call the 'load' method to extract features before plotting.")

        # 1. Process features from the cache
        processed_output = feat_processor.process(self.inputs, self.feature_cache)
        
        # 2. Prepare plot-specific data and config
        plot_data = plot_data_class(**processed_output['data'])
        base_plot_config = plot_config_class(**processed_output['config'])
        
        # 3. Merge with any custom configurations
        final_config = PlotConfig.merge(base_plot_config, custom_config) if custom_config else base_plot_config
        
        print(final_config)
        
        # 4. Generate the plot
        feat_plotter.plot(plot_data, final_config)