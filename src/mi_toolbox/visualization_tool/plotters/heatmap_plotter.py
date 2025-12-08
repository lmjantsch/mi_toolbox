import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field

from ..base_classes import FeaturePlotter, PlotData, PlotConfig


@dataclass
class HeatmapData(PlotData):
    """Data required for a heatmap plot."""
    data_matrix: np.ndarray

@dataclass 
class HeatmapConfig(PlotConfig):
    """Configuration for a heatmap plot."""
    show_cbar: bool = False
    normalize_cmap: bool = False
    x_labels: List[str] = field(default_factory=list)
    y_labels: List[str] = field(default_factory=list)
    x_ticks: List[str] = field(default_factory=list)
    y_ticks: List[str] = field(default_factory=list)
    rotate_x_labels_degree: Optional[int] = None

class HeatmapPlotter(FeaturePlotter):
    """
    Generates single or grid-based heatmaps from a data matrix.
    - A 2D matrix creates a single heatmap.
    - A 3D matrix creates a row of heatmaps.
    - A 4D matrix creates a grid of heatmaps.
    """
    def __init__(self, **kwargs):
        super().__init__()
    
    def _prepare_matrix(self, data_matrix: np.ndarray) -> np.ndarray:
        """
        Ensures the data matrix is 4D for consistent processing,
        and calculates subplot dimensions.
        """
        dims = data_matrix.ndim
        if dims == 2:
            self.n_rows, self.n_cols = 1, 1
            return data_matrix[np.newaxis, np.newaxis, :, :]
        elif dims == 3:
            self.n_rows, self.n_cols = 1, data_matrix.shape[0]
            return data_matrix[np.newaxis, :, :, :]
        elif dims == 4:
            self.n_rows, self.n_cols = data_matrix.shape[0], data_matrix.shape[1]
            return data_matrix
        else: 
            raise ValueError(f"Input data must be 2D, 3D, or 4D, but got {dims} dimensions.")

    def _set_ticks_and_labels(self, ax: plt.Axes, config: HeatmapConfig, row_idx: int, col_idx: int):
        """Configures the ticks and labels for a single subplot."""
        
        # Set Y-axis labels (row labels)
        if config.y_labels and col_idx == 0:
            ax.set_ylabel(config.y_labels[row_idx])

        # Set X-axis labels (column labels)
        if config.x_labels and row_idx == self.n_rows - 1:
            ax.set_xlabel(config.x_labels[col_idx])
            
        # Set Y-axis ticks
        if config.y_ticks and col_idx == 0:
            ax.set_yticks(np.arange(len(config.y_ticks)) + 0.5)
            ax.set_yticklabels(config.y_ticks, rotation=0, fontsize=8)

        # Set X-axis ticks
        if config.x_ticks and row_idx == self.n_rows - 1:
            ax.set_xticks(np.arange(len(config.x_ticks)) + 0.5)
            ax.set_xticklabels(config.x_ticks, fontsize=8)
            if config.rotate_x_labels_degree:
                plt.setp(ax.get_xticklabels(), rotation=config.rotate_x_labels_degree, ha="right", rotation_mode="anchor")
                

    def plot(self, plot_data: HeatmapData, plot_config: HeatmapConfig):
        """
        Generates and displays the heatmap plot.
        """
        matrix = self._prepare_matrix(plot_data.data_matrix)
        
        # Dynamically adjust figure size based on the matrix dimensions
        # to ensure labels and ticks are not cramped.
        n_y_ticks = matrix.shape[-2]
        n_x_ticks = matrix.shape[-1]
        dynamic_height = max(1, n_y_ticks * 0.2)
        dynamic_width = max(1, n_x_ticks * 0.2)
        
        if plot_config.show_cbar and not plot_config.normalize_cmap:
            dynamic_width += 1
        
        fig_height = self.n_rows * dynamic_height + 1.5 # Add space for title/labels
        fig_width = self.n_cols * dynamic_width + 2.5
        
        vmin = matrix.min()
        vmax = matrix.max()
        
        cmap = plt.cm.coolwarm
        cmap.set_bad('lightgrey')
        
        fig, axes = plt.subplots(
            nrows=self.n_rows, 
            ncols=self.n_cols, 
            figsize=(fig_width, fig_height),
            squeeze=False
        )
        fig.suptitle(plot_config.title, fontsize=16, y=0.98)
        
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                ax = axes[r, c]
                subplot_matrix = matrix[r, c]
                
                sns.heatmap(
                    subplot_matrix,
                    ax=ax,
                    cmap=cmap,
                    linewidths=0.5,
                    yticklabels=(c == 0),
                    xticklabels=(r == self.n_rows - 1),
                    annot=False,
                    cbar=plot_config.show_cbar and not plot_config.normalize_cmap,
                    vmin=vmin if plot_config.normalize_cmap else None,
                    vmax=vmax if plot_config.normalize_cmap else None,
                )
                self._set_ticks_and_labels(ax, plot_config, r, c)
                
        # add ONE shared colorbar for the entire figure
        if plot_config.normalize_cmap:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax))
            fig.colorbar(sm, cax=cbar_ax)

        plt.subplots_adjust(hspace = 0.1, wspace=0.1)
        # Adjust subplot layout to make room for the global color bar
        # fig.tight_layout(rect=[0, 0, 0.9, 0.96])            
        plt.show()