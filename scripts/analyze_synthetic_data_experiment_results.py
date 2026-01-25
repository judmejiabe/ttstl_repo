"""
Test script to demonstrate performance comparison for synthetic study experiments.

This script shows how to:
1. Load performance metrics from M, SA, and PL3 experiments
2. Compare experiments across all covariate configurations (01-07)
3. Rank experiments using average ranking with tie handling
4. Analyze results for both global_update and local_adapt modes
"""

# Standard library imports
import json
import logging
import sys
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils_experiment_analysis import (
    ExperimentLoader,
    ExperimentMetadata,
    analyze_covariate_trends,
    compare_experiments,
    plot_performance_boxplot,
    plot_performance_time_series,
)

# Local imports
from ttstl.utils.data_preprocessor import create_pretrained_scaler


def setup_logging():
    """Setup logging for the test script."""
    logging.basicConfig(
        level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def get_convergence_stats(run_dir: Path, backbone_bar: str, mode: str = "local_adapt"):
    """
    Extract convergence statistics from convergence_performance.json file.

    Args:
        run_dir: Path to the experiment run directory
        backbone_bar: Backbone bar identifier (e.g., 'bb_000')
        mode: Mode to check ('local_adapt' or 'global_update')

    Returns:
        Tuple of (failed_count, saddle_point_count) or (0, 0) if file not found
    """
    convergence_file = (
        run_dir / backbone_bar / mode / "adapter" / "convergence_performance.json"
    )

    if not convergence_file.exists():
        return 0, 0

    try:
        with open(convergence_file, "r") as f:
            convergence_data = json.load(f)

        processing_stats = convergence_data.get("processing_stats", {})

        failed_count = 0
        saddle_point_count = 0

        for adapter_bar, stats in processing_stats.items():
            status = stats.get("status", "")
            if status == "failed":
                failed_count += 1
            elif status == "saddle_point_detected":
                saddle_point_count += 1

        return failed_count, saddle_point_count

    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Failed to read convergence stats from {convergence_file}: {e}"
        )
        return 0, 0


def load_all_experiments(
    update_methods=["M", "SA", "PL3", "ES010"],
    backbone_bar="bb_000",
    config_numbers=["01", "02", "03", "04", "05", "06", "07"],
):
    """Load performance data for all experiments matching the specified update methods and config numbers (local_adapt mode only)."""
    logger = logging.getLogger(__name__)
    runs_dir = Path("outputs/runs_synthetic_study")

    if not runs_dir.exists():
        logger.error(f"Runs directory not found: {runs_dir}")
        return []

    # Hardcoded experiment directories to ensure correct loading
    hardcoded_experiments = {
        "M_01": "2025-11-14_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idM_01",
        "M_02": "2025-11-14_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idM_02",
        "M_03": "2025-11-14_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idM_03",
        "M_04": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idM_04",
        "M_05": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idM_05",
        "M_06": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idM_06",
        "M_07": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idM_07",
        "SA_01": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idSA_01",
        "SA_02": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idSA_02",
        "SA_03": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idSA_03",
        "SA_04": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idSA_04",
        "SA_05": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idSA_05",
        "SA_06": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idSA_06",
        "SA_07": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idSA_07",
        "PL3_01": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idPL3_01",
        "PL3_02": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idPL3_02",
        "PL3_03": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idPL3_03",
        "PL3_04": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idPL3_04",
        "PL3_05": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idPL3_05",
        "PL3_06": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idPL3_06",
        "PL3_07": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idPL3_07",
        "ES010_01": "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idES010_01",
        "ES010_02": "2025-11-16_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idES010_02",
        "ES010_03": "2025-11-16_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idES010_03",
        "ES010_04": "2025-11-16_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idES010_04",
        "ES010_05": "2025-11-16_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idES010_05",
        "ES010_06": "2025-11-16_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idES010_06",
        "ES010_07": "2025-11-16_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idES010_07",
    }

    # Build list of experiments based on requested update methods and config numbers
    all_experiments = []
    for method in update_methods:
        for config_num in config_numbers:
            exp_key = f"{method}_{config_num}"
            if exp_key in hardcoded_experiments:
                exp_dir = runs_dir / hardcoded_experiments[exp_key]
                if exp_dir.exists() and (exp_dir / "config.yaml").exists():
                    all_experiments.append(exp_dir)
                else:
                    raise FileNotFoundError(
                        f"Hardcoded experiment not found: {exp_key} -> {exp_dir}"
                    )

    logger.info(
        f"Found {len(all_experiments)} experiments across update methods: {update_methods}"
    )

    # Load performance data for each experiment (local_adapt mode only)
    perf_data_local = []

    for run_dir in all_experiments:
        try:
            metadata = ExperimentMetadata(run_dir=run_dir)
            loader = ExperimentLoader(metadata)

            run_id = metadata.get_config_value("run_id")
            update_method = metadata.get_config_value("update_method")

            # Load performance for local_adapt mode only
            perf_local = loader.load_performance_data(backbone_bar, mode="local_adapt")
            if perf_local:
                # Add convergence statistics to the performance data object
                failed_count, saddle_point_count = get_convergence_stats(
                    run_dir, backbone_bar, mode="local_adapt"
                )
                perf_local.convergence_failed = failed_count
                perf_local.convergence_saddle_point = saddle_point_count

                perf_data_local.append(perf_local)
                logger.info(
                    f"Loaded local_adapt: {run_id} ({update_method}) - Failed: {failed_count}, Saddle: {saddle_point_count}"
                )

        except Exception as e:
            logger.error(f"Failed to load {run_dir.name}: {e}")

    return perf_data_local


def group_by_update_method(perf_data_list):
    """Group PerformanceData objects by their update method."""
    grouped = dict()

    for perf_data in perf_data_list:
        update_method = perf_data.metadata.get_config_value(
            "update_method", default="unknown"
        )

        # Map update methods to cleaner names
        method_map = {
            "myopic": "M (Myopic)",
            "simple_average": "SA (Simple Average)",
            "power_law": "PL3 (Power Law)",
            "exponential_smoothing": "ES010 (Exp Smoothing)",
        }

        method_name = method_map.get(update_method, update_method)

        if method_name not in grouped:
            grouped[method_name] = []

        grouped[method_name].append(perf_data)

    return grouped


def performance_comparison():
    """Compare local adapt performance for bb_000 and bb_001 across all experiments."""
    logger = logging.getLogger(__name__)

    print("Comparing model configurations performance")

    # Create output directory if it doesn't exist
    save_path = Path("outputs/analysis_synthetic_data_experiment_results")
    save_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOCAL ADAPT COMPARISON - bb_000
    # =========================================================================
    perf_data_local_bb000 = load_all_experiments(
        update_methods=["M", "SA", "PL3", "ES010"], backbone_bar="bb_000"
    )

    if perf_data_local_bb000:
        comparison_local_bb000 = compare_experiments(
            perf_data_list=perf_data_local_bb000, metric="garch_relmae"
        )

        if not comparison_local_bb000.empty:
            # Add convergence statistics columns
            comparison_local_bb000["convergence_failed"] = [
                perf.convergence_failed for perf in perf_data_local_bb000
            ]
            comparison_local_bb000["convergence_saddle_point"] = [
                perf.convergence_saddle_point for perf in perf_data_local_bb000
            ]

            # Save comparison results to CSV
            csv_path = save_path / "comparison_bb000_local_adapt.csv"
            comparison_local_bb000.to_csv(csv_path, index=False)

            plot_performance_time_series(
                perf_data_list=perf_data_local_bb000,
                metric="garch_relmae",
                save_path=save_path / "performance_timeseries_bb000_local_adapt.png",
                title="GARCH RelMAE Performance - Local Adapt (bb_000)",
            )

            # Define colors for each configuration
            config_colors = {
                "01": "#e74c3c",  # Red
                "02": "#3498db",  # Blue
                "03": "#2ecc71",  # Green
                "04": "#f39c12",  # Orange
                "05": "#9b59b6",  # Purple
                "06": "#1abc9c",  # Turquoise
                "07": "#e67e22",  # Dark Orange
            }

            # Build color_dict mapping experiment IDs to colors based on config number
            color_dict = {}
            for perf_data in perf_data_local_bb000:
                exp_id = perf_data.metadata.get_config_value(
                    "run_id", default="unknown"
                )
                config_num = exp_id.split("_")[-1] if "_" in exp_id else None
                if config_num and config_num in config_colors:
                    color_dict[exp_id] = config_colors[config_num]

            # Create legend for configuration colors
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", alpha=0.7),
                plt.Rectangle((0, 0), 1, 1, fc="#3498db", alpha=0.7),
                plt.Rectangle((0, 0), 1, 1, fc="#2ecc71", alpha=0.7),
                plt.Rectangle((0, 0), 1, 1, fc="#f39c12", alpha=0.7),
                plt.Rectangle((0, 0), 1, 1, fc="#9b59b6", alpha=0.7),
                plt.Rectangle((0, 0), 1, 1, fc="#1abc9c", alpha=0.7),
                plt.Rectangle((0, 0), 1, 1, fc="#e67e22", alpha=0.7),
            ]
            legend_labels = [
                "Config 01",
                "Config 02",
                "Config 03",
                "Config 04",
                "Config 05",
                "Config 06",
                "Config 07",
            ]
            legend_config = {
                "handles": legend_handles,
                "labels": legend_labels,
                "loc": "upper left",
                "bbox_to_anchor": (1.01, 1),
                "fontsize": 10,
                "title": "Configurations",
            }

            grouped_data_bb000 = group_by_update_method(perf_data_local_bb000)
            plot_performance_boxplot(
                perf_data_dict=grouped_data_bb000,
                metric="garch_relmae",
                color_dict=color_dict,
                legend_config=legend_config,
                xlabel="Update Method",
                ylim=(0.875, 1.075),
                save_path=save_path / "performance_boxplot_bb000.png",
                title="GARCH RelMAE by Update Method (Local Adapt Mode Backbone Bar 1)",
            )

            plot_performance_boxplot(
                perf_data_dict=grouped_data_bb000,
                metric="garch_relmae_nn_baseline",
                color_dict=color_dict,
                legend_config=legend_config,
                xlabel="Update Method",
                ylim=(0.925, 1.1),
                save_path=save_path / "performance_boxplot_bb000_nn_baseline.png",
                title="GARCH RelMAE with CNN-LSTM Baseline by Update Method (Local Adapt Mode Backbone Bar 1)",
            )

    # =========================================================================
    # LOCAL ADAPT COMPARISON - bb_001
    # =========================================================================
    perf_data_local_bb001 = load_all_experiments(
        update_methods=["M", "SA", "PL3", "ES010"], backbone_bar="bb_001"
    )

    if perf_data_local_bb001:
        comparison_local_bb001 = compare_experiments(
            perf_data_list=perf_data_local_bb001, metric="garch_relmae"
        )

        # Add convergence statistics columns
        comparison_local_bb001["convergence_failed"] = [
            perf.convergence_failed for perf in perf_data_local_bb001
        ]
        comparison_local_bb001["convergence_saddle_point"] = [
            perf.convergence_saddle_point for perf in perf_data_local_bb001
        ]

        csv_path = save_path / "comparison_bb001_local_adapt.csv"
        comparison_local_bb001.to_csv(csv_path, index=False)

        plot_performance_time_series(
            perf_data_list=perf_data_local_bb001,
            metric="garch_relmae",
            save_path=save_path / "performance_timeseries_bb001_local_adapt.png",
            title="GARCH RelMAE Performance - Local Adapt (bb_001)",
        )

        # Use same color definitions from bb_000
        color_dict_bb001 = {}
        for perf_data in perf_data_local_bb001:
            exp_id = perf_data.metadata.get_config_value("run_id", default="unknown")
            config_num = exp_id.split("_")[-1] if "_" in exp_id else None
            if config_num and config_num in config_colors:
                color_dict_bb001[exp_id] = config_colors[config_num]

        grouped_data_bb001 = group_by_update_method(perf_data_local_bb001)
        plot_performance_boxplot(
            perf_data_dict=grouped_data_bb001,
            metric="garch_relmae",
            color_dict=color_dict_bb001,
            legend_config=legend_config,
            xlabel="Update Method",
            ylim=(0.875, 1.075),
            save_path=save_path / "performance_boxplot_bb001.png",
            title="GARCH RelMAE by Update Method (Local Adapt Mode Backbone Bar 2)",
        )

        plot_performance_boxplot(
            perf_data_dict=grouped_data_bb001,
            metric="garch_relmae_nn_baseline",
            color_dict=color_dict_bb001,
            legend_config=legend_config,
            xlabel="Update Method",
            ylim=(0.925, 1.1),
            save_path=save_path / "performance_boxplot_bb001_nn_baseline.png",
            title="GARCH RelMAE with CNN-LSTM Baseline by Update Method (Local Adapt Mode Backbone Bar 2)",
        )

    # =========================================================================
    # MERGE COMPARISONS BY EXPERIMENT CONFIGURATION
    # =========================================================================
    if perf_data_local_bb000 and perf_data_local_bb001:
        if not comparison_local_bb000.empty and not comparison_local_bb001.empty:
            # Select columns to keep (including convergence stats, excluding num_observations)
            cols_to_keep = [
                "experiment_id",
                "overall_rank",
                "avg_garch_relmae",
                "best_rank",
                "worst_rank",
                "times_in_top3",
                "convergence_failed",
                "convergence_saddle_point",
            ]

            # Prepare bb_000 data with suffix
            bb000_subset = comparison_local_bb000[cols_to_keep].copy()
            bb000_subset.columns = ["experiment_id"] + [
                f"{col}_bb000" for col in cols_to_keep[1:]
            ]

            # Prepare bb_001 data with suffix
            bb001_subset = comparison_local_bb001[cols_to_keep].copy()
            bb001_subset.columns = ["experiment_id"] + [
                f"{col}_bb001" for col in cols_to_keep[1:]
            ]

            # Merge on experiment_id
            merged_comparison = pd.merge(
                bb000_subset, bb001_subset, on="experiment_id", how="outer"
            )

            # Save merged comparison
            csv_path = save_path / "merged_comparison.csv"
            merged_comparison.to_csv(csv_path, index=False)

    print(f"Comparison complete. Produced files are available at {save_path}")


def covariates_aggregate_measures_trends():
    """Analyze covariate trends across both backbone bars for multiple variables."""
    logger = logging.getLogger(__name__)
    runs_dir = Path("outputs/runs_synthetic_study")
    save_path = Path("outputs/analysis_synthetic_data_experiment_results")

    print("Plotting aggregate measures sequential boxplots")

    # Load experiment
    exp_dir = (
        runs_dir
        / "2025-11-15_SIM_bdelta5000000_adelta100000_sdelta100_seed19991810_idES010_01"
    )

    if not exp_dir.exists():
        logger.warning(f"Experiment directory not found: {exp_dir}")
        return

    metadata = ExperimentMetadata(run_dir=exp_dir)

    # Set GCS paths for synthetic study data
    metadata.gcs_ed_global_path = (
        "synthetic_data/08042025_long_event_driven_recorded_lob_0.parquet"
    )
    metadata.gcs_ptd_global_path = (
        "synthetic_data/08042025_long_physical_time_driven_recorded_data_0.parquet"
    )
    metadata.gcs_ed_local_path = (
        "synthetic_data/08042025_long_event_driven_recorded_lob_1.parquet"
    )
    metadata.gcs_ptd_local_path = (
        "synthetic_data/08042025_long_physical_time_driven_recorded_data_1.parquet"
    )

    # Create preprocessing components
    loader = ExperimentLoader(metadata)
    ed_df, _ = loader.load_data(mode="global_update")
    scaler, lower_bounds, upper_bounds = create_pretrained_scaler(ed_df, quantile=0.005)

    # Define variables to analyze
    variables = [
        "micro_return",
        "hidden_activations_36",
        "hidden_activations_2",
        "hidden_activations_60",
        "hidden_activations_0",
        "hidden_activations_4",
        "hidden_activations_58",
        "forward_pass_0",
        "ASKof0",
        "BIDof0",
        "OFI0",
        "itv",
    ]

    # Extract statistics for both backbone bars
    stats_bb000 = analyze_covariate_trends(
        experiment=metadata,
        covariate_list=variables,
        backbone_bar="bb_000",
        mode="local_adapt",
        pretrained_scaler=scaler,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        device="cuda:2",
        batch_size=512,
    )
    stats_bb000["backbone_bar"] = "bb_000"

    stats_bb001 = analyze_covariate_trends(
        experiment=metadata,
        covariate_list=variables,
        backbone_bar="bb_001",
        mode="local_adapt",
        pretrained_scaler=scaler,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        device="cuda:2",
        batch_size=512,
    )
    stats_bb001["backbone_bar"] = "bb_001"

    # Combine both backbone bars
    all_stats = pd.concat([stats_bb000, stats_bb001], ignore_index=True)

    # Save combined statistics to CSV
    csv_path = save_path / "covariate_statistics_both_backbones.csv"
    all_stats.to_csv(csv_path, index=False)

    # Create plots for each aggregate measure
    aggregate_measures = ["mean", "skewness", "kurtosis"]

    # Standard deviation will be split into two plots
    std_high_scale_vars = ["ASKof0", "BIDof0", "OFI0"]  # Different scale
    std_normal_scale_vars = [v for v in variables if v not in std_high_scale_vars]

    for measure in aggregate_measures:
        # Prepare data for this measure
        plot_data = []
        for var in variables:
            col_name = f"{var}_{measure}"
            if col_name in all_stats.columns:
                for _, row in all_stats.iterrows():
                    plot_data.append(
                        {
                            "variable": var,
                            "backbone_bar": row["backbone_bar"],
                            "adapter_bar": row["adapter_bar"],
                            "value": row[col_name],
                        }
                    )

        plot_df = pd.DataFrame(plot_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))

        # Create boxplot with variables on x-axis, grouped by backbone bar
        positions = []
        labels = []
        boxplot_data = []
        colors = []

        # Color scheme: bb_000 = blue, bb_001 = orange
        color_bb000 = "#3498db"
        color_bb001 = "#e74c3c"

        for i, var in enumerate(variables):
            # Position for bb_000
            pos_bb000 = i * 3
            data_bb000 = plot_df[
                (plot_df["variable"] == var) & (plot_df["backbone_bar"] == "bb_000")
            ]["value"].values

            # Position for bb_001
            pos_bb001 = i * 3 + 1
            data_bb001 = plot_df[
                (plot_df["variable"] == var) & (plot_df["backbone_bar"] == "bb_001")
            ]["value"].values

            if len(data_bb000) > 0:
                positions.append(pos_bb000)
                boxplot_data.append(data_bb000)
                colors.append(color_bb000)

            if len(data_bb001) > 0:
                positions.append(pos_bb001)
                boxplot_data.append(data_bb001)
                colors.append(color_bb001)

            # Add label at center of the two boxes
            labels.append((i * 3 + 0.5, var))

        # Create boxplot
        bp = ax.boxplot(
            boxplot_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=2, color="black"),
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Set x-axis labels
        ax.set_xticks([pos for pos, _ in labels])
        ax.set_xticklabels([label for _, label in labels], rotation=45, ha="right")

        # Add grid
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Labels and title
        measure_titles = {
            "mean": "Mean",
            "std": "Standard Deviation",
            "skewness": "Skewness",
            "kurtosis": "Kurtosis",
            "cv": "Coefficient of Variation",
        }

        ax.set_ylabel(measure_titles[measure], fontsize=12, fontweight="bold")
        ax.set_xlabel("Variables", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{measure_titles[measure]} Comparison Across Backbone Bars - Local Adapt Mode",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Create legend
        legend_elements = [
            Patch(facecolor=color_bb000, alpha=0.7, label="Backbone Bar 1"),
            Patch(facecolor=color_bb001, alpha=0.7, label="Backbone Bar 2"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=11,
        )

        plt.tight_layout()

        # Save plot
        save_path_file = save_path / f"covariate_{measure}_both_backbones.png"
        plt.savefig(save_path_file, dpi=300, bbox_inches="tight")
        plt.close()

    # Create two separate plots for standard deviation
    _create_std_plot(all_stats, std_high_scale_vars, "std_high_scale", save_path)
    _create_std_plot(all_stats, std_normal_scale_vars, "std_normal_scale", save_path)

    print(f"Produced plots are available at {save_path}")


def _create_std_plot(all_stats, variables_subset, plot_suffix, save_path):
    """Helper function to create standard deviation plots."""
    measure = "std"

    # Prepare data for this measure
    plot_data = []
    for var in variables_subset:
        col_name = f"{var}_{measure}"
        if col_name in all_stats.columns:
            for _, row in all_stats.iterrows():
                plot_data.append(
                    {
                        "variable": var,
                        "backbone_bar": row["backbone_bar"],
                        "adapter_bar": row["adapter_bar"],
                        "value": row[col_name],
                    }
                )

    plot_df = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Create boxplot with variables on x-axis, grouped by backbone bar
    positions = []
    labels = []
    boxplot_data = []
    colors = []

    # Color scheme: bb_000 = blue, bb_001 = orange
    color_bb000 = "#3498db"
    color_bb001 = "#e74c3c"

    for i, var in enumerate(variables_subset):
        # Position for bb_000
        pos_bb000 = i * 3
        data_bb000 = plot_df[
            (plot_df["variable"] == var) & (plot_df["backbone_bar"] == "bb_000")
        ]["value"].values

        # Position for bb_001
        pos_bb001 = i * 3 + 1
        data_bb001 = plot_df[
            (plot_df["variable"] == var) & (plot_df["backbone_bar"] == "bb_001")
        ]["value"].values

        if len(data_bb000) > 0:
            positions.append(pos_bb000)
            boxplot_data.append(data_bb000)
            colors.append(color_bb000)

        if len(data_bb001) > 0:
            positions.append(pos_bb001)
            boxplot_data.append(data_bb001)
            colors.append(color_bb001)

        # Add label at center of the two boxes
        labels.append((i * 3 + 0.5, var))

    # Create boxplot
    bp = ax.boxplot(
        boxplot_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="black"),
    )

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Set x-axis labels
    ax.set_xticks([pos for pos, _ in labels])
    ax.set_xticklabels([label for _, label in labels], rotation=45, ha="right")

    # Add grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Labels and title
    var_list = ", ".join(variables_subset)

    ax.set_ylabel("Standard Deviation", fontsize=12, fontweight="bold")
    ax.set_xlabel("Variables", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Standard Deviation Comparison Across Backbone Bars - Local Adapt Mode",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Create legend
    legend_elements = [
        Patch(facecolor=color_bb000, alpha=0.7, label="Backbone Bar 1"),
        Patch(facecolor=color_bb001, alpha=0.7, label="Backbone Bar 2"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=11
    )

    plt.tight_layout()

    # Save plot
    save_path_file = save_path / f"covariate_{plot_suffix}_both_backbones.png"
    plt.savefig(save_path_file, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main function - comprehensive analysis."""
    setup_logging()

    print("Starting analysis of experimental study with synthetic data")

    # Part 1: Performance comparison
    performance_comparison()

    # Part 2: Covariate trends analysis
    covariates_aggregate_measures_trends()

    print("Concluding analysis of experimental study with synthetic data")


if __name__ == "__main__":
    main()
