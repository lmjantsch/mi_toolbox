from typing import Tuple, Type, Dict
from .base_classes import FeatureProcessor, FeaturePlotter, PlotData, PlotConfig

from visualization_tool.processors import *
from visualization_tool.plotters import *

# Type alias for the components required for a plot
PlotComponentBundle = Tuple[Type[FeatureProcessor], Type[FeaturePlotter], Type[PlotData], Type[PlotConfig]]

# Global registry to hold predefined plot configurations
PLOT_REGISTRY: Dict[str, PlotComponentBundle] = {
    'attention_heatmap': (
        AttentionProcessor,
        HeatmapPlotter,
        HeatmapData,
        HeatmapConfig
    ),
    'pref_attn_heatmap': (
        PrefAttnProcessor,
        HeatmapPlotter,
        HeatmapData,
        HeatmapConfig
    ),
    'attn_from_heatmap': (
        AttnFromProcessor,
        HeatmapPlotter,
        HeatmapData,
        HeatmapConfig
    ),
}