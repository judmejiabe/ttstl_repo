"""
Unified Experiment Analysis Framework

This module provides a modular, reusable framework for analyzing TTSTL experiments,
supporting both synthetic study and market data experiments through abstract interfaces
and concrete implementations.

Author: Juan Diego Mejia Becerra
Email: juanmejia.7565@gmail.com
Date: November 4, 2025
"""

# Standard library imports
import argparse
import json
import logging
import os
import re
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from scipy import stats
from statsmodels.tsa.stattools import ccf

# Local imports
from ttstl.architectures.cnn_lstm import CNNLSTM
from ttstl.backbone import Backbone
from ttstl.utils.data_downloader import import_to_pandas
from ttstl.utils.data_preprocessor import create_pretrained_scaler

# =============================================================================
# 1. CORE DATA STRUCTURES
# =============================================================================


@dataclass
class ExperimentMetadata:
    """Reconstruct Orchestrator from run directory using config.yaml file."""

    run_dir: Path

    # GCS data paths (optional - for data loading)
    gcs_bucket: Optional[str] = None
    gcs_credentials_path: str = "credentials.json"
    gcs_ed_global_path: Optional[
        str
    ] = None  # e.g., 'synthetic_data/08042025_long_event_driven_recorded_lob_0.parquet'
    gcs_ptd_global_path: Optional[
        str
    ] = None  # e.g., 'synthetic_data/08042025_long_physical_time_driven_recorded_data_0.parquet'
    gcs_ed_local_path: Optional[
        str
    ] = None  # e.g., 'synthetic_data/08042025_long_event_driven_recorded_lob_1.parquet'
    gcs_ptd_local_path: Optional[
        str
    ] = None  # e.g., 'synthetic_data/08042025_long_physical_time_driven_recorded_data_1.parquet'

    # Cached configuration (loaded on demand)
    _config: Optional[Dict[str, Any]] = field(default=None, repr=False, init=False)

    def __post_init__(self):
        """Initialize GCS bucket if not provided."""
        if self.gcs_bucket is None:
            # Strict loading from credentials.json
            bucket_name = None

            creds_path = Path(self.gcs_credentials_path)
            if not creds_path.exists():
                # Try finding it relative to project root if simple path fails
                # Just a best-effort check if we are deep in directories
                project_root = Path(__file__).resolve().parent.parent.parent
                alt_path = project_root / "credentials.json"
                if alt_path.exists():
                    creds_path = alt_path
                    self.gcs_credentials_path = str(alt_path)

            if creds_path.exists():
                try:
                    with open(creds_path, "r") as f:
                        creds = json.load(f)
                        bucket_name = creds.get("bucket_name")
                except Exception as e:
                    logging.error(f"Failed to read credentials.json: {e}")
                    raise ValueError(
                        f"Failed to read credentials.json: {e}. Please ensure it is a valid JSON file."
                    )

            if not bucket_name:
                error_msg = (
                    f"Could not find 'bucket_name' in {self.gcs_credentials_path}. "
                    "The credentials.json file must contain a 'bucket_name' key. "
                    "Please refer to the 'Credentials Configuration' section in README.md."
                )
                logging.error(error_msg)
                raise ValueError(error_msg)

            self.gcs_bucket = bucket_name

    def get_config(self) -> Dict[str, Any]:
        """
        Load and return the Orchestrator configuration from config.yaml.
        
        This method reads the config.yaml file in the run directory and returns
        the complete Orchestrator configuration as a dictionary. The configuration
        is cached after the first load.
        
        Returns:
            Dictionary containing the full Orchestrator configuration
            
        Raises:
            FileNotFoundError: If config.yaml doesn't exist in run_dir
            ValueError: If config.yaml cannot be parsed
        """
        if self._config is None:
            config_path = self.run_dir / "config.yaml"

            if not config_path.exists():
                raise FileNotFoundError(f"config.yaml not found in {self.run_dir}")

            try:
                with open(config_path, "r") as f:
                    # Use unsafe_load to handle Python objects (StandardScaler, etc.)
                    # This is safe because we're loading our own generated config files
                    self._config = yaml.unsafe_load(f)
            except Exception as e:
                raise ValueError(f"Failed to parse config.yaml in {self.run_dir}: {e}")

        return self._config

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific value from the Orchestrator configuration.
        
        Args:
            key: Configuration parameter name (e.g., 'run_id', 'epochs', 'seed')
            default: Value to return if key doesn't exist
            
        Returns:
            Configuration value or default if key not found
            
        Example:
            >>> metadata = ExperimentMetadata(run_dir=Path('outputs/runs/exp1'))
            >>> run_id = metadata.get_config_value('run_id')
            >>> epochs = metadata.get_config_value('epochs', default=0)
        """
        config = self.get_config()
        return config.get(key, default)


@dataclass
class PerformanceData:
    """Unified performance data structure within a backbone and mode."""

    data: pd.DataFrame  # Raw performance metrics
    metadata: ExperimentMetadata
    mode: str  # Mode of operation (e.g., 'global_update', 'local_adapt')
    backbone_bar: str  # Backbone bar identifier (e.g., 'bb_000', 'bb_001')

    # Convergence statistics (optional, populated from convergence_performance.json)
    convergence_failed: Optional[
        int
    ] = None  # Count of adapter bars with 'failed' status
    convergence_saddle_point: Optional[
        int
    ] = None  # Count of adapter bars with 'saddle_point_detected' status

    # Required columns for performance data
    # Core metrics that must be present in the DataFrame
    required_columns: ClassVar[List[str]] = [
        "garch_mae",
        "garch_mse",
        "garch_relmae",
        "garch_relmse",
        "garch_r2",
        "nn_mae",
        "nn_mse",
        "nn_relmae",
        "nn_relmse",
        "nn_r2",
    ]

    def validate(self) -> bool:
        """
        Check if DataFrame has required columns.
        
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If required columns are missing or data is empty
        """
        if self.data is None or self.data.empty:
            raise ValueError("Performance data is empty")

        # Check for required columns
        missing_cols = set(self.required_columns) - set(self.data.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in performance data: {sorted(missing_cols)}\n"
                f"Available columns: {sorted(self.data.columns.tolist())}"
            )

        # Check for adapter_bar in index or columns
        has_adapter_bar = (
            "adapter_bar" in self.data.columns
            or self.data.index.name == "adapter_bar"
            or "adapter_bar" in self.data.index.names
        )
        if not has_adapter_bar:
            raise ValueError(
                "Performance data must have 'adapter_bar' in index or columns"
            )

        return True

    def compute_relative_performance_nn_baseline(self) -> pd.DataFrame:
        """
        Compute relative performance of GARCH model against NN baseline.
        
        Creates a new column 'garch_relmae_nn_baseline' which is the ratio of
        garch_relmae to nn_relmae. Values < 1 indicate GARCH outperforms NN baseline.
        
        Returns:
            DataFrame with additional 'garch_relmae_nn_baseline' column
            
        Raises:
            ValueError: If required columns are missing
        """
        required = ["garch_relmae", "nn_relmae"]
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Compute relative performance: GARCH RelMAE / NN RelMAE
        self.data["garch_relmae_nn_baseline"] = (
            self.data["garch_relmae"] / self.data["nn_relmae"]
        )

        return self.data


# =============================================================================
# 2. DATA LOADING LAYER
# =============================================================================


class ExperimentLoader:
    """Class for loading experiment data."""

    def __init__(self, metadata: ExperimentMetadata):
        """
        Initialize loader with experiment metadata.
        
        Args:
            metadata: ExperimentMetadata object containing run directory and GCS paths
        """
        self.metadata = metadata

    def load_data(self, mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess data from GCS.
        
        Args:
            mode: Which data to load - 'global_update' or 'local_adapt'
            
        Returns:
            Tuple of (ed_df, ptd_df) for the requested mode
            
        Raises:
            ValueError: If GCS paths are not set or mode is invalid
            ImportError: If data_downloader module is not available
        """
        if mode not in ["global_update", "local_adapt"]:
            raise ValueError(
                f"mode must be 'global_update' or 'local_adapt', got: {mode}"
            )

        def load_and_preprocess(
            ed_path: str, ptd_path: str
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """Load event-driven and physical time-driven data, compute returns."""
            if ed_path is None or ptd_path is None:
                raise ValueError(f"GCS paths not set for this mode")

            # Load event driven data
            ed_df = import_to_pandas(
                blob_name=ed_path,
                credentials_path=self.metadata.gcs_credentials_path,
                bucket_name=self.metadata.gcs_bucket,
            )

            # Load physical time driven data
            ptd_df = import_to_pandas(
                blob_name=ptd_path,
                credentials_path=self.metadata.gcs_credentials_path,
                bucket_name=self.metadata.gcs_bucket,
            )

            # Handle timestamp index/column conversion
            if ptd_df.index.name == "timestamp" or "timestamp" in ptd_df.index.names:
                ptd_df = ptd_df.reset_index()
            elif "timestamp" not in ptd_df.columns:
                raise ValueError(
                    "No timestamp found in either index or columns for physical time-driven data"
                )

            # Compute micro returns (future returns by default)
            ptd_df["micro_return"] = ptd_df["micro"].pct_change() * 1e3
            # ptd_df['micro_return'] = ptd_df['micro_return'].shift(-1)
            ptd_df = ptd_df.iloc[:-1]

            return ed_df, ptd_df

        if mode == "global_update":
            return load_and_preprocess(
                self.metadata.gcs_ed_global_path, self.metadata.gcs_ptd_global_path
            )
        else:  # local_adapt
            return load_and_preprocess(
                self.metadata.gcs_ed_local_path, self.metadata.gcs_ptd_local_path
            )

    def load_performance_data(
        self, backbone_bar: str, mode: str
    ) -> Optional[PerformanceData]:
        """
        Load performance metrics from performance.parquet file in adapter subdirectory.
        
        Args:
            backbone_bar: Backbone bar identifier (e.g., 'bb_000', 'bb_001')
            mode: Which mode to load - 'global_update' or 'local_adapt'
            
        Returns:
            PerformanceData object with performance metrics, or None if file doesn't exist
            
        Raises:
            ValueError: If mode is invalid
            FileNotFoundError: If performance.parquet doesn't exist
            
        Example:
            >>> loader = ExperimentLoader(metadata)
            >>> perf_global = loader.load_performance_data('bb_000', mode='global_update')
            >>> perf_local = loader.load_performance_data('bb_000', mode='local_adapt')
        """
        if mode not in ["global_update", "local_adapt"]:
            raise ValueError(
                f"mode must be 'global_update' or 'local_adapt', got: {mode}"
            )

        perf_path = (
            self.metadata.run_dir
            / backbone_bar
            / mode
            / "adapter"
            / "performance.parquet"
        )

        if not perf_path.exists():
            return None

        try:
            df = pd.read_parquet(perf_path)
            # Add metadata columns if not present
            if "backbone_bar" not in df.columns:
                df["backbone_bar"] = backbone_bar
            if "mode" not in df.columns:
                df["mode"] = mode
        except Exception as e:
            raise ValueError(f"Failed to load performance data from {perf_path}: {e}")

        # Create and validate PerformanceData object
        perf_data = PerformanceData(
            data=df, metadata=self.metadata, mode=mode, backbone_bar=backbone_bar
        )
        perf_data.validate()

        return perf_data


# =============================================================================
# 3. DATA PROCESSING LAYER
# =============================================================================

# 3.1 Performance Aggregation
# -----------------------------------------------------------------------------


def compare_experiments(
    perf_data_list: List[PerformanceData], metric: str = "garch_relmae"
) -> pd.DataFrame:
    """
    Compare multiple experiments side-by-side with average rankings.
    
    Args:
        perf_data_list: List of PerformanceData objects to compare
        metric: Performance metric to compare (default: 'garch_relmae')
    
    Returns:
        DataFrame with comparison statistics including:
        - experiment_id: Experiment identifier (from metadata.get_config_value('run_id'))
        - avg_rank: Average rank across all adapter bars (ties handled with 'average' method)
        - median_rank: Median rank
        - std_rank: Standard deviation of ranks
        - best_rank: Best (minimum) rank achieved
        - worst_rank: Worst (maximum) rank achieved
        - rank_range: Difference between worst and best rank
        - times_ranked_1st: Number of times ranked first
        - times_in_top3: Number of times ranked in top 3
        - avg_metric: Average value of the metric
        - median_metric: Median value of the metric
    
    Raises:
        ValueError: If perf_data_list is empty or contains mixed modes
    
    Notes:
        - Ties are handled using pandas rank(method='average'), which assigns the
          average of the ranks that would have been assigned to tied values
        - Lower metric values are assumed to be better (ascending=True for ranking)
        - experiment_id is extracted from metadata.get_config_value('run_id')
        - IMPORTANT: All PerformanceData objects must have the same mode
          (global_update and local_adapt are not comparable)
    
    Example:
        >>> # Compare local adaptation performance
        >>> perf_list = [perf_data_m01, perf_data_m02, perf_data_sa01]
        >>> comparison = compare_experiments(perf_list, metric='garch_relmae')
        >>> 
        >>> # Compare global update performance separately
        >>> perf_list_global = [perf_data_global_01, perf_data_global_02]
        >>> comparison_global = compare_experiments(perf_list_global, metric='garch_relmae')
    """
    if not perf_data_list:
        raise ValueError("perf_data_list cannot be empty")

    # Infer mode from first PerformanceData object
    mode = perf_data_list[0].mode

    # Validate that all PerformanceData objects have the same mode
    for perf_data in perf_data_list:
        if perf_data.mode != mode:
            raise ValueError(
                f"All PerformanceData objects must have the same mode. "
                f"Found mixed modes: '{mode}' and '{perf_data.mode}'. "
                f"global_update and local_adapt are not comparable and must be analyzed separately."
            )

    comparison_data = []

    # Collect all performance data
    all_data = []
    for perf_data in perf_data_list:

        # Compute garch_relmae_nn_baseline if the metric requires it
        if (
            metric == "garch_relmae_nn_baseline"
            and "garch_relmae_nn_baseline" not in perf_data.data.columns
        ):
            perf_data.compute_relative_performance_nn_baseline()

        df = perf_data.data.copy(deep=True)

        # Get experiment_id from metadata
        exp_id = perf_data.metadata.get_config_value("run_id", default="unknown")

        # Add experiment_id if not present
        if "experiment_id" not in df.columns:
            df["experiment_id"] = exp_id

        all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=False)

    # Reset index to get adapter_bar as a column if it's in the index
    if combined_df.index.name == "adapter_bar":
        combined_df = combined_df.reset_index()

    # Get unique adapter bars
    adapter_bars = combined_df["adapter_bar"].unique()

    # Calculate ranks at each adapter bar
    ranking_data = []
    for adapter_bar in adapter_bars:
        bar_data = combined_df[combined_df["adapter_bar"] == adapter_bar].copy()

        if bar_data.empty or metric not in bar_data.columns:
            continue

        # Rank by metric (ascending - lower is better) with average tie handling
        bar_data["rank"] = bar_data[metric].rank(method="average", ascending=True)

        # Store ranking information
        for _, row in bar_data.iterrows():
            ranking_data.append(
                {
                    "experiment_id": row["experiment_id"],
                    "adapter_bar": adapter_bar,
                    "rank": row["rank"],
                    metric: row[metric],
                }
            )

    if not ranking_data:
        return pd.DataFrame()

    # Create ranking DataFrame
    ranking_df = pd.DataFrame(ranking_data)

    # Calculate comparison statistics for each experiment
    for exp_id in ranking_df["experiment_id"].unique():
        exp_ranks = ranking_df[ranking_df["experiment_id"] == exp_id]

        ranks = exp_ranks["rank"]
        metric_values = exp_ranks[metric]

        comparison_entry = {
            "experiment_id": exp_id,
            "avg_rank": ranks.mean(),
            "median_rank": ranks.median(),
            "std_rank": ranks.std(),
            "best_rank": ranks.min(),
            "worst_rank": ranks.max(),
            "rank_range": ranks.max() - ranks.min(),
            "times_ranked_1st": (ranks == 1.0).sum(),
            "times_in_top3": (ranks <= 3.0).sum(),
            "num_observations": len(ranks),
            f"avg_{metric}": metric_values.mean(),
            f"median_{metric}": metric_values.median(),
            f"std_{metric}": metric_values.std(),
            f"min_{metric}": metric_values.min(),
            f"max_{metric}": metric_values.max(),
        }
        comparison_data.append(comparison_entry)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Add overall rank based on average rank
    comparison_df["overall_rank"] = comparison_df["avg_rank"].rank(
        method="average", ascending=True
    )

    return comparison_df


def plot_performance_time_series(
    perf_data_list: List[PerformanceData],
    metric: str = "garch_relmae",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot performance metric time series for multiple experiments.
    
    Creates a line plot showing how the performance metric evolves across adapter bars
    for all experiments in the list. Missing adapter bars (due to numerical failures)
    are handled gracefully with gaps in the lines.
    
    Args:
        perf_data_list: List of PerformanceData objects to plot
        metric: Performance metric to plot (default: 'garch_relmae')
        save_path: Optional path to save the figure (e.g., 'outputs/analysis/plot.png')
        figsize: Figure size as (width, height) in inches
        title: Optional custom title for the plot
    
    Returns:
        Matplotlib Figure object
    
    Raises:
        ValueError: If perf_data_list is empty or contains mixed modes
    
    Notes:
        - All PerformanceData objects must have the same mode
          (global_update and local_adapt should be plotted separately)
    
    Example:
        >>> perf_list = [perf_data_m01, perf_data_m02, perf_data_sa01]
        >>> fig = plot_performance_time_series(
        ...     perf_list, 
        ...     metric='garch_relmae',
        ...     save_path='outputs/analysis/performance_timeseries.png'
        ... )
    """
    if not perf_data_list:
        raise ValueError("perf_data_list cannot be empty")

    # Infer mode from first PerformanceData object
    mode = perf_data_list[0].mode

    # Validate that all PerformanceData objects have the same mode
    for perf_data in perf_data_list:
        if perf_data.mode != mode:
            raise ValueError(
                f"All PerformanceData objects must have the same mode. "
                f"Found mixed modes: '{mode}' and '{perf_data.mode}'. "
                f"global_update and local_adapt should be plotted separately."
            )

    # Collect data for plotting
    plot_data = []

    for perf_data in perf_data_list:
        # Compute garch_relmae_nn_baseline if the metric requires it
        if (
            metric == "garch_relmae_nn_baseline"
            and "garch_relmae_nn_baseline" not in perf_data.data.columns
        ):
            perf_data.compute_relative_performance_nn_baseline()

        df = perf_data.data.copy(deep=True)

        # Get experiment_id from metadata
        exp_id = perf_data.metadata.get_config_value("run_id", default="unknown")

        # Reset index to get adapter_bar as column if needed
        if df.index.name == "adapter_bar":
            df = df.reset_index()

        # Check if metric exists
        if metric not in df.columns:
            continue

        # Extract adapter_bar and metric values
        for _, row in df.iterrows():
            plot_data.append(
                {
                    "experiment_id": exp_id,
                    "adapter_bar": row["adapter_bar"],
                    metric: row[metric],
                }
            )

    if not plot_data:
        raise ValueError("No data available to plot")

    # Create long-form DataFrame
    plot_df_long = pd.DataFrame(plot_data)

    # Convert to wide format: rows=adapter_bar, columns=experiment_id
    plot_df = plot_df_long.pivot(
        index="adapter_bar", columns="experiment_id", values=metric
    ).sort_index()

    # Create complete index with all adapter bars (to introduce NaNs for missing bars)
    min_bar = plot_df.index.min()
    max_bar = plot_df.index.max()
    complete_index = pd.RangeIndex(start=min_bar, stop=max_bar + 1)

    # Reindex to include all adapter bars, filling missing ones with NaN
    plot_df = plot_df.reindex(complete_index)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot line for each experiment (column in wide DataFrame)
    for exp_id in plot_df.columns:
        ax.plot(
            plot_df.index,  # adapter_bar values
            plot_df[exp_id],
            marker="o",
            markersize=4,
            label=exp_id,
            linewidth=1.5,
            alpha=0.8,
        )

    # Customize plot
    ax.set_xlabel("Adapter Bar", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(
            f'{metric.replace("_", " ").title()} Across Adapter Bars ({mode.replace("_", " ").title()})',
            fontsize=14,
            fontweight="bold",
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved plot to {save_path}")

    return fig


def plot_performance_boxplot(
    perf_data_dict: Dict[str, List[PerformanceData]],
    metric: str = "garch_relmae",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10),
    title: Optional[str] = None,
    xtick_labels: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    color_dict: Optional[Dict[str, List[str]]] = None,
    legend_config: Optional[Dict[str, str]] = None,
    hline: Optional[float] = 1.0,
    ylim: Optional[Tuple[float, float]] = (0.9, 1.1),
) -> plt.Figure:
    """
    Create grouped boxplots comparing the metric distribution across experiments.
    
    Each key in perf_data_dict represents a group, and each group contains a list of
    PerformanceData objects. Boxplots for experiments within each group are plotted
    side-by-side for comparison.
    
    Args:
        perf_data_dict: Dictionary mapping group names to lists of PerformanceData objects
                       e.g., {'Config_01': [perf_M_01, perf_SA_01, perf_PL3_01],
                              'Config_02': [perf_M_02, perf_SA_02, perf_PL3_02]}
        metric: Performance metric to plot (default: 'garch_relmae')
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches
        title: Optional custom title for the plot
        xtick_labels: Optional custom labels for x-axis groups (defaults to dict keys)
        xlabel: Optional label for the x-axis (defaults to 'Groups')
        color_dict: Dictionary mapping experiment IDs to colors (e.g., {'M_01': '#FF0000', 'SA_02': '#00FF00'})
                   If an experiment ID is not in the dict, default gray color is used.
                   If None, all experiments use gray.
        legend_config: Dictionary with complete legend configuration passed directly to ax.legend().
                      Must include 'handles' and 'labels' keys, plus any other matplotlib legend kwargs.
                      If None, no legend is created. Example:
                      {'handles': [patch1, patch2], 'labels': ['A', 'B'], 'loc': 'upper right'}
        hline: Y-value at which to draw a horizontal dashed reference line (default: 1.0)
        ylim: Tuple specifying y-axis limits (ymin, ymax). 
              Default: (0.875, 1.075) for garch_relmae, (0.925, 1.1) for garch_relmae_nn_baseline
              Set to None to use automatic limits.
    
    Returns:
        Matplotlib Figure object
    
    Example:
        >>> perf_dict = {
        ...     'M': [perf_M_01, perf_M_02, perf_M_03],
        ...     'SA': [perf_SA_01, perf_SA_02, perf_SA_03],
        ...     'PL3': [perf_PL3_01, perf_PL3_02, perf_PL3_03]
        ... }
        >>> color_dict = {'M_01': '#e74c3c', 'M_02': '#3498db', 'M_03': '#2ecc71',
        ...               'SA_01': '#e74c3c', 'SA_02': '#3498db', 'SA_03': '#2ecc71',
        ...               'PL3_01': '#e74c3c', 'PL3_02': '#3498db', 'PL3_03': '#2ecc71'}
        >>> fig = plot_performance_boxplot(
        ...     perf_dict,
        ...     metric='garch_relmae',
        ...     mode='local_adapt',
        ...     color_dict=color_dict,
        ...     hline=1.0,
        ...     save_path='outputs/analysis/boxplot.png'
        ... )
    """
    # Collect data for each group
    plot_data = []
    group_names = list(perf_data_dict.keys())

    for group_name, perf_data_list in perf_data_dict.items():
        for perf_data in perf_data_list:

            # Compute garch_relmae_nn_baseline if the metric requires it
            if (
                metric == "garch_relmae_nn_baseline"
                and "garch_relmae_nn_baseline" not in perf_data.data.columns
            ):
                perf_data.compute_relative_performance_nn_baseline()

            df = perf_data.data.copy(deep=True)

            # Get experiment_id from metadata
            exp_id = perf_data.metadata.get_config_value("run_id", default="unknown")

            # Check if metric exists
            if metric not in df.columns:
                continue

            # Extract all metric values for this experiment
            for _, row in df.iterrows():
                plot_data.append(
                    {
                        "group": group_name,
                        "experiment_id": exp_id,
                        "metric_value": row[metric],
                    }
                )

    if not plot_data:
        raise ValueError("No data available to plot")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for grouped boxplot
    # Group by (group, experiment_id) and collect metric values
    grouped_data = []
    positions = []
    labels = []
    colors = []

    # Default gray color for experiments not in color_dict
    default_color = "gray"

    # Use color_dict directly as mapping from experiment_id to color
    # If color_dict is None, all experiments get default gray
    if color_dict is None:
        color_dict = {}

    position = 0
    group_positions = []
    group_centers = []

    for group_idx, group_name in enumerate(group_names):
        group_df = plot_df[plot_df["group"] == group_name]
        experiments_in_group = group_df["experiment_id"].unique()

        group_start = position

        for exp_id in experiments_in_group:
            exp_data = group_df[group_df["experiment_id"] == exp_id][
                "metric_value"
            ].values
            grouped_data.append(exp_data)
            positions.append(position)
            labels.append(exp_id)
            colors.append(color_dict.get(exp_id, default_color))
            position += 1

        # Add spacing between groups
        group_end = position - 1
        group_centers.append((group_start + group_end) / 2)
        position += 1  # Add gap between groups
        group_positions.append(group_end + 0.5)

    # Create boxplots
    bp = ax.boxplot(
        grouped_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="red"),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5),
    )

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add horizontal reference line at hline
    if hline is not None:
        ax.axhline(
            y=hline, color="black", linestyle="--", linewidth=2, alpha=0.7, zorder=0
        )

    # Customize plot
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=14)
    ax.set_xlabel(xlabel if xlabel else "Groups", fontsize=14)

    # Set x-tick labels at group centers
    if xtick_labels:
        ax.set_xticks(group_centers)
        ax.set_xticklabels(xtick_labels, fontsize=12)
    else:
        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_names, fontsize=12)

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold")
    else:
        ax.set_title(
            f'Grouped {metric.replace("_", " ").title()} Distribution',
            fontsize=16,
            fontweight="bold",
        )

    # Create legend only if legend_config is provided
    if legend_config:
        ax.legend(**legend_config)

    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Set y-axis limits
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved boxplot to {save_path}")

    return fig


# 3.3 Covariate Analysis
# -----------------------------------------------------------------------------


def load_covariate_counts(
    experiments: Dict[str, ExperimentMetadata], mode: str = "global_update"
) -> pd.DataFrame:
    """
    Extract number of selected covariates for each experiment.
    Returns DataFrame with experiment_id, backbone_bar, covariate_count.
    """
    pass


def compare_covariate_selection(
    experiments: List[ExperimentMetadata], backbone_bar: str = "bb_000"
) -> pd.DataFrame:
    """
    Compare which covariates were selected across experiments.
    Returns intersection/union analysis.
    """
    pass


def analyze_covariate_trends(
    experiment: ExperimentMetadata,
    covariate_list: List[str],
    backbone_bar: str = "bb_000",
    mode: str = "local_adapt",
    pretrained_scaler: Optional[Any] = None,
    lower_bounds: Optional[Dict[str, float]] = None,
    upper_bounds: Optional[Dict[str, float]] = None,
    device: str = "cuda",
    batch_size: int = 512,
    compute_ccf: bool = False,
    max_lags: Optional[int] = None,
) -> pd.DataFrame:
    """
    Analyze trends in covariate statistics over time.
    
    This function loads physical time-driven data, optionally extracts neural covariates,
    filters to a specific backbone_bar, and computes aggregate statistics per adapter_bar.
    
    Implementation Plan (Multi-Stage):
    
    Stage 1 - Data Loading and Preparation:
    - Load physical time-driven data using ExperimentLoader
    - Add backbone_bar and adapter_bar columns based on session timing
    - Filter to session bounds
    
    Stage 2 (Implemented) - Generative Covariates:
    - Automatically detect neural covariates from covariate_list:
      * "hidden_activations_*" prefix → requires non-parsimonious mode (64 features)
      * "forward_pass" prefix → requires parsimonious mode (1 feature)
      * Both prefixes → extract both types and merge
    - Load trained model weights for the specified mode
    - Extract neural network features as needed
    - Merge neural covariates with physical time-driven data using merge_asof
    
    Stage 3 (Implemented) - Statistical Aggregation:
    - Filter data to specified backbone_bar
    - Compute aggregate statistics per adapter_bar using efficient pandas groupby:
      * Mean
      * Standard deviation
      * Skewness
      * Kurtosis
      * Coefficient of variation (std/mean)
    - Return DataFrame with statistics
    
    Args:
        experiment: ExperimentMetadata object for a single experiment (with GCS paths set)
        covariate_list: List of covariate names to analyze. Can include:
                       - Raw covariates (e.g., 'OFI0', 'itv')
                       - Hidden activations (e.g., 'hidden_activations_0', 'hidden_activations_10')
                       - Forward pass output (e.g., 'forward_pass')
                       Neural covariates are automatically detected and extracted as needed.
        backbone_bar: Backbone bar identifier (e.g., 'bb_000', 'bb_001')
        mode: Which mode data to use - 'global_update' or 'local_adapt'
        save_path: Deprecated - no longer used (function returns DataFrame, not plot)
        figsize: Deprecated - no longer used (function returns DataFrame, not plot)
        pretrained_scaler: Fitted StandardScaler for preprocessing (required if neural covariates detected).
                          Should be created externally using create_pretrained_scaler().
        lower_bounds: Dictionary mapping feature names to lower quantile bounds (required if neural covariates detected)
        upper_bounds: Dictionary mapping feature names to upper quantile bounds (required if neural covariates detected)
        device: Device for model inference ('cuda' or 'cpu')
        batch_size: Batch size for neural covariate extraction
        compute_ccf: If True, compute cross-correlation function (CCF) between each covariate and micro_return
                    at both positive and negative lags. Default is False (backward compatible).
        max_lags: Maximum absolute lag for CCF computation (e.g., max_lags=50 computes lags from -50 to +50).
                 Required if compute_ccf=True. Must be a positive integer.
    
    Returns:
        pandas DataFrame with columns:
            - backbone_bar: Backbone bar index (only present if multiple adapters share same backbone)
            - adapter_bar: Adapter bar index
            - {covariate}_mean: Mean value per adapter_bar
            - {covariate}_std: Standard deviation per adapter_bar  
            - {covariate}_skewness: Skewness per adapter_bar
            - {covariate}_kurtosis: Kurtosis per adapter_bar
            - {covariate}_cv: Coefficient of variation per adapter_bar
            - {covariate}_ccf_lag_{lag}: Cross-correlation with micro_return at specified lag 
                                         (only present if compute_ccf=True)
                                         Positive lags: covariate leads micro_return
                                         Negative lags: micro_return leads covariate
        
    Example:
        >>> # Basic usage with raw covariates only
        >>> stats_df = analyze_covariate_trends(
        ...     experiment=metadata,
        ...     covariate_list=['OFI0', 'itv'],
        ...     backbone_bar='bb_000',
        ...     mode='local_adapt'
        ... )
        >>> 
        >>> # With neural covariates (auto-detected)
        >>> from ttstl.utils.data_preprocessor import create_pretrained_scaler
        >>> loader = ExperimentLoader(metadata)
        >>> ed_df, _ = loader.load_data(mode='local_adapt')
        >>> scaler, lower_bounds, upper_bounds = create_pretrained_scaler(ed_df, quantile=0.005)
        >>> 
        >>> stats_df = analyze_covariate_trends(
        ...     experiment=metadata,
        ...     covariate_list=['OFI0', 'hidden_activations_0', 'forward_pass'],
        ...     backbone_bar='bb_000',
        ...     mode='local_adapt',
        ...     pretrained_scaler=scaler,
        ...     lower_bounds=lower_bounds,
        ...     upper_bounds=upper_bounds,
        ...     device='cuda',
        ...     batch_size=512
        ... )
        >>> 
        >>> # Inspect the resulting statistics
        >>> print(stats_df.columns)
        >>> # ['adapter_bar', 'OFI0_mean', 'OFI0_std', 'OFI0_skewness', 'OFI0_kurtosis', 'OFI0_cv', ...]
        >>> print(stats_df.head())
        >>> 
        >>> # With CCF analysis (requires micro_return in data)
        >>> ccf_stats_df = analyze_covariate_trends(
        ...     experiment=metadata,
        ...     covariate_list=['OFI0', 'itv'],
        ...     backbone_bar='bb_000',
        ...     mode='local_adapt',
        ...     compute_ccf=True,
        ...     max_lags=50  # Computes CCF from lag -50 to +50
        ... )
        >>> 
        >>> # CCF columns will be added: OFI0_ccf_lag_-50, ..., OFI0_ccf_lag_0, ..., OFI0_ccf_lag_50
        >>> print([col for col in ccf_stats_df.columns if 'ccf' in col])
    """
    # ==========================================================================
    # AUTO-DETECT: Parse covariate_list to determine neural covariate needs
    # ==========================================================================

    # Detect which types of neural covariates are requested
    needs_hidden_activations = any(
        "hidden_activations_" in cov for cov in covariate_list
    )
    needs_forward_pass = any("forward_pass" in cov for cov in covariate_list)
    needs_neural_covariates = needs_hidden_activations or needs_forward_pass

    if needs_neural_covariates:
        print(f"🔍 Auto-detected neural covariate requirements:")
        if needs_hidden_activations:
            print(f"   ✓ Hidden activations (64D) - non-parsimonious mode")
        if needs_forward_pass:
            print(f"   ✓ Forward pass output (1D) - parsimonious mode")

    # ==========================================================================
    # STAGE 1: Basic Time Series Visualization
    # ==========================================================================

    # 1. Load physical time-driven data using ExperimentLoader
    loader = ExperimentLoader(experiment)

    # Determine which data to load based on mode
    if mode == "global_update":
        ed_df, ptd_df = loader.load_data(mode="global_update")
    elif mode == "local_adapt":
        ed_df, ptd_df = loader.load_data(mode="local_adapt")
    else:
        raise ValueError(f"mode must be 'global_update' or 'local_adapt', got: {mode}")

    # 2. Get metadata for bar calculations
    session_start = experiment.get_config_value("session_start")
    session_finish = experiment.get_config_value("session_finish")
    bdelta = experiment.get_config_value("bdelta")
    adelta = experiment.get_config_value("adelta")

    if session_start is None or session_finish is None:
        raise ValueError(
            "session_start and session_finish must be defined in experiment metadata"
        )
    if bdelta is None or adelta is None:
        raise ValueError("bdelta and adelta must be defined in experiment metadata")

    # Ensure timestamp is a column (not index)
    if ptd_df.index.name == "timestamp" or "timestamp" in ptd_df.index.names:
        ptd_df = ptd_df.reset_index()

    # Add backbone_bar and adapter_bar columns based on timestamp
    # Backbone bars: timestamps from session_start to session_start + bdelta, etc.
    # Adapter bars: within each backbone bar, subdivided by adelta

    # Calculate backbone bar index
    ptd_df["backbone_bar"] = ((ptd_df["timestamp"] - session_start) // bdelta).astype(
        int
    )

    # Calculate global adapter bar index (direct calculation)
    ptd_df["adapter_bar"] = ((ptd_df["timestamp"] - session_start) // adelta).astype(
        int
    )

    # Filter to only include data within session bounds
    ptd_df = ptd_df[
        (ptd_df["timestamp"] >= session_start) & (ptd_df["timestamp"] <= session_finish)
    ]

    print(f"Stage 1: Loaded {len(ptd_df):,} physical time-driven data points")

    # ==========================================================================
    # STAGE 2: Generative Covariates Extraction
    # ==========================================================================

    if needs_neural_covariates:
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info("STAGE 2: Extracting Neural Covariates")
        logger.info("=" * 80)

        # Validate that preprocessing components are provided
        if pretrained_scaler is None or lower_bounds is None or upper_bounds is None:
            raise ValueError(
                "Neural covariates detected in covariate_list but preprocessing components are missing. "
                "Please provide pretrained_scaler, lower_bounds, and upper_bounds. "
                "Create these externally using create_pretrained_scaler() before calling this function."
            )

        # Set up device properly
        if device == "cpu":
            device_obj = torch.device("cpu")
        else:
            # Handle cuda, cuda:0, cuda:1, etc.
            device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
            if not torch.cuda.is_available() and device != "cpu":
                logger.warning(
                    f"CUDA not available. Falling back to CPU instead of {device}"
                )

        logger.info(f"Using device: {device_obj}")

        try:
            # Step 1: Log preprocessing components
            logger.info("Using externally provided preprocessing components...")
            logger.info(f"Scaler available: {pretrained_scaler is not None}")
            logger.info(f"Bounds available for {len(lower_bounds)} features")

            # Step 2: Instantiate model architecture
            logger.info("Instantiating CNNLSTM architecture...")
            model = CNNLSTM()

            # Step 3: Load trained weights
            # Note: best.pt weights are always stored under global_update directory
            weights_path = (
                experiment.run_dir
                / backbone_bar
                / "global_update"
                / "backbone"
                / "architecture"
                / "best.pt"
            )
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Trained weights not found at {weights_path}. "
                    f"Make sure the experiment has been run and weights were saved."
                )

            logger.info(f"Loading trained weights from {weights_path.name}...")

            # Load weights with security best practices
            state_dict = torch.load(
                weights_path,
                map_location=device_obj,
                weights_only=True,  # Security: prevent arbitrary code execution
            )

            # Load weights into model with error checking
            try:
                missing_keys, unexpected_keys = model.load_state_dict(
                    state_dict, strict=False
                )
                if missing_keys:
                    logger.warning(f"Missing keys when loading weights: {missing_keys}")
                if unexpected_keys:
                    logger.warning(
                        f"Unexpected keys when loading weights: {unexpected_keys}"
                    )
            except Exception as e:
                raise ValueError(f"Failed to load model weights: {str(e)}")

            # Move model to device and set to evaluation mode
            model.to(device_obj)
            model.eval()

            logger.info("Successfully loaded trained weights")

            # Step 4: Get configuration parameters
            delta_sample = experiment.get_config_value("sdelta")
            delta_adapter = experiment.get_config_value("adelta")
            past_returns = experiment.get_config_value("past_returns", default=False)

            if delta_sample is None or delta_adapter is None:
                raise ValueError(
                    "sdelta and adelta must be defined in experiment config"
                )

            logger.info(
                f"Configuration: sdelta={delta_sample}, adelta={delta_adapter}, past_returns={past_returns}"
            )

            # Step 5: Extract neural covariates based on detected needs
            # We may need to extract both types: hidden activations (64D) and forward pass (1D)

            if needs_hidden_activations:
                logger.info(
                    "Creating Backbone instance for hidden activations (non-parsimonious mode)..."
                )
                backbone_hidden = Backbone(
                    model=model,
                    delta_adapter=delta_adapter,
                    delta_sample=delta_sample,
                    parsimonious_mode=False,  # 64D hidden features
                    neural_covariates_only=True,
                    manual_covariates_only=False,
                    pretrained_scaler=pretrained_scaler,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    past_returns=past_returns,
                )

                logger.info(f"Extracting hidden activations (64D)...")
                hidden_covariates_df = backbone_hidden.extract_neural_covariates(
                    event_driven_data=ed_df,
                    physical_time_data=ptd_df,
                    batch_size=batch_size,
                    get_output=False,
                )
                logger.info(
                    f"✅ Extracted {hidden_covariates_df.shape[1] - 1} hidden activation features"
                )

                # Ensure timestamp dtype compatibility before merge
                # Convert both to float64 for consistent merging
                ptd_df["timestamp"] = ptd_df["timestamp"].astype(float)
                hidden_covariates_df["timestamp"] = hidden_covariates_df[
                    "timestamp"
                ].astype(float)

                # Merge hidden activations with physical time data using merge_asof
                # Sort both DataFrames by timestamp before merge_asof
                ptd_df = ptd_df.sort_values("timestamp")
                hidden_covariates_df = hidden_covariates_df.sort_values("timestamp")
                ptd_df = pd.merge_asof(
                    ptd_df,
                    hidden_covariates_df,
                    on="timestamp",
                    direction="nearest",
                    tolerance=1e-6,
                )

            if needs_forward_pass:
                logger.info(
                    "Creating Backbone instance for forward pass (parsimonious mode)..."
                )
                backbone_forward = Backbone(
                    model=model,
                    delta_adapter=delta_adapter,
                    delta_sample=delta_sample,
                    parsimonious_mode=True,  # 1D forward pass output
                    neural_covariates_only=True,
                    manual_covariates_only=False,
                    pretrained_scaler=pretrained_scaler,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    past_returns=past_returns,
                )

                logger.info(f"Extracting forward pass output (1D)...")
                forward_covariates_df = backbone_forward.extract_neural_covariates(
                    event_driven_data=ed_df,
                    physical_time_data=ptd_df,
                    batch_size=batch_size,
                    get_output=False,
                )
                logger.info(
                    f"✅ Extracted {forward_covariates_df.shape[1] - 1} forward pass feature(s)"
                )

                # Ensure timestamp dtype compatibility before merge
                # Convert both to float64 for consistent merging
                ptd_df["timestamp"] = ptd_df["timestamp"].astype(float)
                forward_covariates_df["timestamp"] = forward_covariates_df[
                    "timestamp"
                ].astype(float)

                # Merge forward pass with physical time data using merge_asof
                # Sort both DataFrames by timestamp before merge_asof
                ptd_df = ptd_df.sort_values("timestamp")
                forward_covariates_df = forward_covariates_df.sort_values("timestamp")
                ptd_df = pd.merge_asof(
                    ptd_df,
                    forward_covariates_df,
                    on="timestamp",
                    direction="nearest",
                    tolerance=1e-6,
                )

            logger.info(f"✅ Stage 2 complete. Total columns: {len(ptd_df.columns)}")

        except Exception as e:
            logger.error(f"Failed to extract neural covariates: {str(e)}")
            logger.warning("Continuing with raw covariates only (Stage 1)")
            traceback.print_exc()

    # ==========================================================================
    # STAGE 3: Statistical Aggregation
    # ==========================================================================

    print(f"Stage 3: Computing statistics per adapter_bar for backbone {backbone_bar}")

    # Filter data to specified backbone_bar
    bb_index = int(backbone_bar.split("_")[-1])
    ptd_df_filtered = ptd_df[ptd_df["backbone_bar"] == bb_index].copy()

    if ptd_df_filtered.empty:
        raise ValueError(
            f"No data found for backbone_bar {backbone_bar} (index {bb_index})"
        )

    print(
        f"Filtered to {len(ptd_df_filtered):,} data points in backbone {backbone_bar}"
    )

    # Filter to only include requested covariates
    available_covariates = [
        cov for cov in covariate_list if cov in ptd_df_filtered.columns
    ]
    if not available_covariates:
        raise ValueError(
            f"None of the requested covariates {covariate_list} found in data"
        )

    missing_covariates = set(covariate_list) - set(available_covariates)
    if missing_covariates:
        print(f"⚠️  Covariates not found in data: {missing_covariates}")

    # Select only necessary columns: adapter_bar + covariates
    cols_to_aggregate = ["adapter_bar"] + available_covariates
    df_to_aggregate = ptd_df_filtered[cols_to_aggregate]

    # Compute all statistics using groupby aggregation
    # This is much more efficient than iterating through covariates
    stats_df = df_to_aggregate.groupby("adapter_bar").agg(
        [
            ("mean", "mean"),
            ("std", "std"),
            ("skewness", lambda x: stats.skew(x, nan_policy="omit")),
            ("kurtosis", lambda x: stats.kurtosis(x, nan_policy="omit")),
            ("cv", lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan),
        ]
    )

    # Flatten multi-level columns: (covariate, statistic) -> covariate_statistic
    stats_df.columns = ["_".join(col).strip() for col in stats_df.columns.values]

    # Reset index to make adapter_bar a column
    stats_df = stats_df.reset_index()

    print(
        f"✅ Computed statistics for {len(available_covariates)} covariates across {len(stats_df)} adapter bars"
    )
    print(f"✅ Resulting DataFrame shape: {stats_df.shape}")

    # ==========================================================================
    # STAGE 4: Cross-Correlation Function (CCF) Analysis (optional)
    # ==========================================================================
    if compute_ccf:
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info("STAGE 4: Computing Cross-Correlation Functions (CCF)")
        logger.info("=" * 80)

        # Validate max_lags
        if max_lags is None or not isinstance(max_lags, int) or max_lags <= 0:
            raise ValueError("compute_ccf=True requires a positive integer 'max_lags'")

        # Ensure micro_return is present
        if "micro_return" not in ptd_df_filtered.columns:
            raise ValueError(
                "CCF analysis requires 'micro_return' column in filtered physical time data"
            )

        # Only consider covariates that were requested (covariate_list) and are available
        ccf_covariates = [cov for cov in covariate_list if cov in available_covariates]
        if not ccf_covariates:
            logger.warning(
                "No requested covariates from covariate_list found in filtered data for CCF computation"
            )
        else:
            logger.info(
                f"Computing CCF for {len(ccf_covariates)} covariates: {ccf_covariates}"
            )
            logger.info(f"Max lags: ±{max_lags}")

        # Prepare storage
        ccf_results = {}  # {adapter_bar: {covariate: {lag: value}}}

        # Adapter bars to iterate (use those present in stats_df to keep alignment)
        adapter_bars_to_process = list(stats_df["adapter_bar"].unique())

        # Minimum required observations heuristic
        min_required_obs = max(2 * max_lags + 10, 30)

        for adapter_bar in adapter_bars_to_process:
            # Extract adapter_bar-level data (already filtered to backbone)
            adapter_data = ptd_df_filtered[
                ptd_df_filtered["adapter_bar"] == adapter_bar
            ].sort_values("timestamp")

            # Convert to numpy arrays for speed
            micro_arr = adapter_data["micro_return"].to_numpy()

            # Initialize per-adapter store
            ccf_results[adapter_bar] = {}

            # Quick check
            if adapter_data.shape[0] < min_required_obs:
                logger.warning(
                    f"Adapter bar {adapter_bar}: only {adapter_data.shape[0]} observations; "
                    f"need at least {min_required_obs} for reliable CCF (max_lags={max_lags}); "
                    "CCF will be skipped for this adapter_bar (NaNs recorded)."
                )
                # Fill NaNs for each requested covariate
                for cov in ccf_covariates:
                    ccf_results[adapter_bar][cov] = {
                        lag: np.nan for lag in range(-max_lags, max_lags + 1)
                    }
                continue

            # For each requested covariate compute CCF for lags -max_lags..+max_lags
            for cov in ccf_covariates:
                if cov not in adapter_data.columns:
                    # Shouldn't usually happen because we filtered available_covariates, but guard anyway
                    logger.warning(
                        f"Adapter bar {adapter_bar}: covariate '{cov}' not found in data; filling NaNs"
                    )
                    ccf_results[adapter_bar][cov] = {
                        lag: np.nan for lag in range(-max_lags, max_lags + 1)
                    }
                    continue

                cov_arr = adapter_data[cov].to_numpy()

                # Check for NaNs - raise error if any are present
                # (removing NaNs destroys temporal dependence structure)
                if np.any(np.isnan(micro_arr)):
                    raise ValueError(
                        f"Adapter bar {adapter_bar}: micro_return contains NaN values. "
                        f"CCF computation requires complete time series without NaNs to preserve temporal structure."
                    )

                if np.any(np.isnan(cov_arr)):
                    raise ValueError(
                        f"Adapter bar {adapter_bar}, covariate '{cov}': contains NaN values. "
                        f"CCF computation requires complete time series without NaNs to preserve temporal structure."
                    )

                # Check minimum required observations
                if micro_arr.size < min_required_obs:
                    logger.warning(
                        f"Adapter bar {adapter_bar}, covariate '{cov}': "
                        f"only {micro_arr.size} observations; need at least {min_required_obs}; skipping CCF."
                    )
                    ccf_results[adapter_bar][cov] = {
                        lag: np.nan for lag in range(-max_lags, max_lags + 1)
                    }
                    continue

                try:
                    # Positive/non-negative lags: correlation between cov[t] and micro[t+lag]
                    # statsmodels.ccf(x, y, nlags) returns correlations for lags 0..nlags where
                    # element k corresponds to corr(x[t + k], y[t]).
                    pos_ccf = ccf(cov_arr, micro_arr, nlags=max_lags + 1)
                    # Negative lags: correlation for lag -k is corr(cov[t-k], micro[t])
                    # We obtain these by computing ccf(y, x) and mapping index k -> lag -k
                    neg_ccf = ccf(micro_arr, cov_arr, nlags=max_lags + 1)

                    # Combine into full mapping -max_lags..+max_lags.
                    # pos_ccf[0] and neg_ccf[0] represent lag 0; use pos_ccf[0].
                    lag_map = {}
                    # negative lags: -max_lags .. -1 mapped from neg_ccf[ max_lags .. 1 ]
                    for k in range(1, max_lags + 1):
                        lag = -k
                        # neg_ccf[k] corresponds to corr(x[t - k], y[t]) (lag -k)
                        lag_map[lag] = (
                            float(neg_ccf[k]) if not np.isnan(neg_ccf[k]) else np.nan
                        )

                    # lag 0
                    lag_map[0] = (
                        float(pos_ccf[0]) if not np.isnan(pos_ccf[0]) else np.nan
                    )

                    # positive lags: 1 .. max_lags from pos_ccf[1 ..]
                    for k in range(1, max_lags + 1):
                        lag = k
                        lag_map[lag] = (
                            float(pos_ccf[k]) if not np.isnan(pos_ccf[k]) else np.nan
                        )

                    ccf_results[adapter_bar][cov] = lag_map

                except Exception as e:
                    logger.error(
                        f"Adapter bar {adapter_bar}, covariate '{cov}': CCF computation failed with error: {e}"
                    )
                    # On failure put NaNs for all lags
                    ccf_results[adapter_bar][cov] = {
                        lag: np.nan for lag in range(-max_lags, max_lags + 1)
                    }

        # Merge CCF results into stats_df (Option A: wide format columns with inner join)
        if ccf_covariates:
            logger.info(
                "Merging CCF results into statistics DataFrame (wide format columns, inner join)"
            )
            # Build ordered list of columns (covariate_lag_-max .. covariate_lag_max)
            ccf_columns = []
            for cov in ccf_covariates:
                for lag in range(-max_lags, max_lags + 1):
                    ccf_columns.append(f"{cov}_ccf_lag_{lag}")

            # Prepare data rows aligned with stats_df adapter_bar ordering
            ccf_rows_data = []

            for adapter_bar in stats_df["adapter_bar"]:
                row_dict = {"adapter_bar": adapter_bar, "backbone_bar": bb_index}

                if adapter_bar in ccf_results:
                    for cov in ccf_covariates:
                        cov_lags = ccf_results[adapter_bar].get(cov, None)
                        if cov_lags is None:
                            # fill NaNs for that covariate
                            for lag in range(-max_lags, max_lags + 1):
                                row_dict[f"{cov}_ccf_lag_{lag}"] = np.nan
                        else:
                            for lag in range(-max_lags, max_lags + 1):
                                row_dict[f"{cov}_ccf_lag_{lag}"] = cov_lags.get(
                                    lag, np.nan
                                )
                else:
                    # No CCF computed for this adapter bar => fill NaNs for all covariates/lags
                    for cov in ccf_covariates:
                        for lag in range(-max_lags, max_lags + 1):
                            row_dict[f"{cov}_ccf_lag_{lag}"] = np.nan

                ccf_rows_data.append(row_dict)

            # Create ccf_df
            ccf_df = pd.DataFrame(ccf_rows_data)

            # Add backbone_bar to stats_df if not present (for merge)
            if "backbone_bar" not in stats_df.columns:
                stats_df["backbone_bar"] = bb_index

            # Inner merge on backbone_bar and adapter_bar
            stats_df = pd.merge(
                stats_df, ccf_df, on=["backbone_bar", "adapter_bar"], how="inner"
            )

            logger.info(
                f"Added {len(ccf_columns)} CCF columns to stats_df (lags -{max_lags}..+{max_lags})"
            )
            logger.info(f"Merged result shape: {stats_df.shape}")

        logger.info("✅ Stage 4 (CCF) complete")

    return stats_df


def extract_generative_covariates(
    experiment: ExperimentMetadata,
    backbone_bar: str = "bb_000",
    mode: str = "local_adapt",
    device: str = "cuda",
    batch_size: int = 512,
    parsimonious_mode: bool = False,
) -> pd.DataFrame:
    """
    Extract generative (neural network) covariates from a trained experiment.
    
    This function reconstructs the trained neural network from an experiment's saved
    weights and uses it to extract neural covariates for the physical time-driven data.
    It follows the exact preprocessing pipeline used during training to ensure consistency.
    
    NOTE: This function is also integrated into analyze_covariate_trends() as Stage 2.
    Use this standalone function when you need neural covariates without plotting,
    or use analyze_covariate_trends(include_neural_covariates=True) for integrated analysis.
    
    Implementation Steps:
    1. Load event-driven and physical time-driven data using ExperimentLoader
    2. Create preprocessing components (scaler, bounds) from event-driven data
    3. Instantiate the model architecture (CNNLSTM)
    4. Load trained weights from best.pt file
    5. Create Backbone instance with proper configuration
    6. Extract neural covariates using backbone.extract_neural_covariates()
    
    Args:
        experiment: ExperimentMetadata object with run_dir and GCS paths configured
        backbone_bar: Backbone bar identifier (e.g., 'bb_000', 'bb_001')
        mode: Which mode to use - 'global_update' or 'local_adapt'
        device: Device for model inference ('cuda' or 'cpu')
        batch_size: Batch size for neural covariate extraction
        parsimonious_mode: If True, use forward pass output (1D); if False, use hidden activations (64D)
    
    Returns:
        DataFrame with columns:
        - timestamp: Physical time timestamps
        - hidden_activations_0, hidden_activations_1, ... (if parsimonious_mode=False)
        - forward_pass_0 (if parsimonious_mode=True)
    
    Raises:
        FileNotFoundError: If best.pt weights file doesn't exist
        ValueError: If mode is invalid or data loading fails
    
    Example:
        >>> metadata = ExperimentMetadata(
        ...     run_dir=Path('outputs/runs_synthetic_study_covariance/2025-11-05_...'),
        ...     gcs_ed_global_path='synthetic_data/...',
        ...     gcs_ptd_global_path='synthetic_data/...',
        ...     gcs_ed_local_path='synthetic_data/...',
        ...     gcs_ptd_local_path='synthetic_data/...'
        ... )
        >>> neural_covs = extract_generative_covariates(
        ...     metadata, 
        ...     backbone_bar='bb_000', 
        ...     mode='local_adapt',
        ...     parsimonious_mode=False
        ... )
        >>> print(neural_covs.shape)  # (n_physical_timestamps, 65) # timestamp + 64 hidden features
    
    Notes:
        - Uses the same preprocessing pipeline as during training (via create_pretrained_scaler)
        - Requires GCS paths to be set in the ExperimentMetadata
        - The returned covariates are aligned with physical time-driven timestamps
        - NaN values indicate physical timestamps before the first event-driven observation
    """
    logger = logging.getLogger(__name__)

    # Validate inputs
    if mode not in ["global_update", "local_adapt"]:
        raise ValueError(f"mode must be 'global_update' or 'local_adapt', got: {mode}")

    # Set up device properly
    if device == "cpu":
        device_obj = torch.device("cpu")
    else:
        # Handle cuda, cuda:0, cuda:1, etc.
        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available() and device != "cpu":
            logger.warning(
                f"CUDA not available. Falling back to CPU instead of {device}"
            )

    logger.info(f"Using device: {device_obj}")

    # Step 1: Load data using ExperimentLoader
    logger.info(f"Loading data for {mode} mode...")
    loader = ExperimentLoader(experiment)
    ed_df, ptd_df = loader.load_data(mode=mode)
    logger.info(
        f"Loaded {len(ed_df):,} event-driven rows and {len(ptd_df):,} physical time-driven rows"
    )

    # Step 2: Create preprocessing components from event-driven data
    # This replicates the exact preprocessing used during training
    logger.info("Creating preprocessing components from event-driven data...")
    try:
        scaler, lower_bounds, upper_bounds = create_pretrained_scaler(
            ed_df, quantile=0.005
        )
        logger.info(
            f"Created scaler for {len([col for col in ed_df.columns if 'of' in col])} order flow features"
        )
    except Exception as e:
        raise ValueError(f"Failed to create preprocessing components: {str(e)}")

    # Step 3: Instantiate model architecture
    logger.info("Instantiating CNNLSTM architecture...")
    model = CNNLSTM()

    # Step 4: Load trained weights from best.pt
    # Note: best.pt weights are always stored under global_update directory
    weights_path = (
        experiment.run_dir
        / backbone_bar
        / "global_update"
        / "backbone"
        / "architecture"
        / "best.pt"
    )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {weights_path}. "
            f"Make sure the experiment has been run and weights were saved."
        )

    logger.info(f"Loading trained weights from {weights_path}...")

    # Load weights with security best practices
    state_dict = torch.load(
        weights_path,
        map_location=device_obj,
        weights_only=True,  # Security: prevent arbitrary code execution
    )

    # Load weights into model with error checking
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading weights: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading weights: {unexpected_keys}")
    except Exception as e:
        raise ValueError(f"Failed to load model weights: {str(e)}")

    # Move model to device and set to evaluation mode
    model.to(device_obj)
    model.eval()

    logger.info("Successfully loaded trained weights")

    # Step 5: Get configuration parameters from metadata
    delta_sample = experiment.get_config_value("sdelta")
    delta_adapter = experiment.get_config_value("adelta")
    past_returns = experiment.get_config_value("past_returns", default=False)

    if delta_sample is None or delta_adapter is None:
        raise ValueError("sdelta and adelta must be defined in experiment config")

    logger.info(
        f"Using delta_sample={delta_sample}, delta_adapter={delta_adapter}, past_returns={past_returns}"
    )

    # Step 6: Create Backbone instance with preprocessing components
    logger.info("Creating Backbone instance...")
    backbone = Backbone(
        model=model,
        delta_adapter=delta_adapter,
        delta_sample=delta_sample,
        parsimonious_mode=parsimonious_mode,
        neural_covariates_only=True,  # We only want neural covariates
        manual_covariates_only=False,
        pretrained_scaler=scaler,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        past_returns=past_returns,
    )

    # Step 7: Extract neural covariates
    logger.info(
        f"Extracting neural covariates (parsimonious_mode={parsimonious_mode})..."
    )
    try:
        neural_covariates_df = backbone.extract_neural_covariates(
            event_driven_data=ed_df,
            physical_time_data=ptd_df,
            batch_size=batch_size,
            get_output=False,  # Use parsimonious_mode parameter instead
        )
        logger.info(
            f"✅ Successfully extracted {neural_covariates_df.shape[1] - 1} neural covariates"
        )
        return neural_covariates_df
    except Exception as e:
        raise ValueError(f"Failed to extract neural covariates: {str(e)}")


# =============================================================================
# 4. VISUALIZATION LAYER
# =============================================================================

# 4.1 Generic Visualization Utilities
# -----------------------------------------------------------------------------


@dataclass
class PlotConfig:
    """Configuration for plot aesthetics."""

    style: str = "default"
    palette: str = "husl"
    figsize: Tuple[int, int] = (16, 10)
    dpi: int = 300
    colors: Dict[int, str] = field(
        default_factory=lambda: {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    )
    markers: Dict[int, str] = field(default_factory=lambda: {0: "o", 1: "s", 2: "^"})


def setup_plot_style(config: PlotConfig = PlotConfig()) -> None:
    """Apply consistent plot styling."""
    pass


# 4.2 Performance Visualization - Time Series Plots
# -----------------------------------------------------------------------------

# 4.2 Performance Visualization - Box Plots
# -----------------------------------------------------------------------------


# 4.2 Performance Visualization - Heatmaps
# -----------------------------------------------------------------------------


def plot_ranking_heatmap(
    ranking_df: pd.DataFrame,
    title: str = "Experiment Rankings",
    cmap: str = "RdYlGn_r",
    save_path: Optional[Path] = None,
    config: PlotConfig = PlotConfig(),
) -> plt.Figure:
    """
    Visualize rankings as heatmap.
    Rows: adapter_bars/time_points
    Columns: experiments
    Values: ranks (lower=better, colored green)
    """
    pass


# 4.3 Covariate Visualization
# -----------------------------------------------------------------------------


def plot_covariate_counts_time_series(
    covariate_df: pd.DataFrame,
    x_col: str = "adapter_bar",  # 'adapter_bar' or 'pair_number'
    x_labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    config: PlotConfig = PlotConfig(),
) -> plt.Figure:
    """
    Plot number of selected covariates over time.
    Lines: one per backbone_bar
    """
    pass


def plot_covariate_selection_comparison(
    experiments: List[ExperimentMetadata],
    backbone_bar: str = "bb_000",
    save_path: Optional[Path] = None,
    config: PlotConfig = PlotConfig(),
) -> plt.Figure:
    """
    Compare covariate selection across experiments.
    Visualization: Upset plot or intersection matrix
    """
    pass


# =============================================================================
# 5. STATISTICAL ANALYSIS LAYER
# =============================================================================

# 5.1 Summary Statistics
# -----------------------------------------------------------------------------


def create_summary_statistics(
    perf_data: PerformanceData,
    metrics: List[str] = ["garch_relmae", "nn_relmae"],
    groupby: List[str] = ["experiment_id", "mode", "backbone_bar"],
) -> pd.DataFrame:
    """
    Calculate descriptive statistics.
    Returns: mean, median, std, min, max, count for each group.
    """
    pass


# 5.2 Hypothesis Testing
# -----------------------------------------------------------------------------


def test_performance_difference(
    perf_data1: PerformanceData,
    perf_data2: PerformanceData,
    metric: str = "garch_relmae",
    test_type: str = "wilcoxon",  # or 't-test', 'mann-whitney'
) -> Dict[str, Any]:
    """
    Statistical test for performance difference.
    Returns: test_statistic, p_value, effect_size
    """
    pass


def bootstrap_confidence_intervals(
    perf_data: PerformanceData,
    metric: str = "garch_relmae",
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for performance metrics.
    Returns: lower_bound, upper_bound for each experiment.
    """
    pass


# =============================================================================
# 6. EXPORT UTILITIES
# =============================================================================

# 6.1 Table Export
# -----------------------------------------------------------------------------


def export_summary_table(
    summary_df: pd.DataFrame,
    output_path: Path,
    format: str = "csv",  # 'csv', 'excel', 'latex'
) -> None:
    """Export summary statistics to file."""
    pass


def export_latex_table(
    df: pd.DataFrame,
    output_path: Path,
    caption: str = "",
    label: str = "",
    float_format: str = "%.3f",
) -> None:
    """Export formatted LaTeX table for publications."""
    pass


# 6.2 Report Generation
# -----------------------------------------------------------------------------


def generate_analysis_report(
    perf_data: PerformanceData,
    output_dir: Path,
    include_plots: bool = True,
    include_rankings: bool = True,
    include_statistics: bool = True,
) -> None:
    """
    Generate comprehensive analysis report.
    Creates: summary tables, plots, rankings, markdown report
    """
    pass


# =============================================================================
# 7. MAIN ORCHESTRATION
# =============================================================================

# 7.1 Analysis Pipeline
# -----------------------------------------------------------------------------


class AnalysisPipeline:
    """Orchestrate complete analysis workflow."""

    def __init__(self, experiment_type: str, runs_dir: Path, output_dir: Path):
        """Initialize with appropriate loader."""
        pass

    def run_full_analysis(
        self,
        backbone_bars: List[str] = ["bb_000", "bb_001"],
        modes: List[str] = ["global_update", "local_adapt"],
    ) -> None:
        """Execute complete analysis pipeline."""
        pass


# 7.2 Unified Main Function
# -----------------------------------------------------------------------------


def setup_logging():
    """Setup logging for analysis operations."""
    pass


def main():
    """Main entry point with CLI argument parsing."""
    pass


if __name__ == "__main__":
    main()
