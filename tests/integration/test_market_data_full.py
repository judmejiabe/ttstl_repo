import logging
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
import torch

from ttstl import orchestrator
from ttstl.orchestrator import Orchestrator, OrchestratorConfig
from ttstl.utils.data_downloader import import_to_pandas

# Import necessary methods from market_data_experiments script
from ttstl.utils.experiment_utils import (
    get_base_config,
    load_date_data,
    load_pretrained_model_for_experiments,
    log_orchestrator_config,
    set_trained_model_to_orchestrator,
)


def setup_logging():
    """Setup logging to see orchestrator operations."""

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"market_data_experiments_test_{timestamp}.log"

    # Configure logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Write to file
            logging.StreamHandler(),  # Also show in terminal
        ],
    )

    print(f"📝 Logs will be saved to: {log_file}")


def load_market_data():
    """
    Loads market data for INTC for the test date pair: June 10, 2024 and June 11, 2024 (PAIR05).
    
    Returns:
        Tuple of (global_update_ed_df, global_update_ptd_df, local_update_ed_df, local_update_ptd_df)
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading market data for INTC test pair PAIR05: 06102024 → 06112024")

    # Load global update data (June 10, 2024)
    logger.info("Loading global update data (June 10, 2024)")
    global_ed_df, global_ptd_df = load_date_data("06102024")

    if global_ed_df is None or global_ptd_df is None:
        logger.error("Failed to load global update data for June 10, 2024")
        raise ValueError("Could not load global update data for 06102024")

    # Load local update data (June 11, 2024)
    logger.info("Loading local update data (June 11, 2024)")
    local_ed_df, local_ptd_df = load_date_data("06112024")

    if local_ed_df is None or local_ptd_df is None:
        logger.error("Failed to load local update data for June 11, 2024")
        raise ValueError("Could not load local update data for 06112024")

    return global_ed_df, global_ptd_df, local_ed_df, local_ptd_df


def plot_microreturns_analysis(
    ptd_df: pd.DataFrame,
    session_start: float,
    session_finish: float,
    save_dir: Path,
    data_label: str = "",
):
    """
    Create time series and histogram plots for microreturns within session bounds.
    
    Args:
        ptd_df: Physical time-driven DataFrame with timestamp and micro_return columns
        session_start: Session start timestamp (nanoseconds)
        session_finish: Session finish timestamp (nanoseconds)
        save_dir: Directory to save plots
        data_label: Label for the data (e.g., "June_27" or "June_28")
    """
    logger = logging.getLogger(__name__)

    # Filter data to session bounds
    session_data = ptd_df[
        (ptd_df["timestamp"] >= session_start) & (ptd_df["timestamp"] <= session_finish)
    ].copy()

    if session_data.empty:
        logger.warning(f"No data found within session bounds for {data_label}")
        return

    # Remove NaN values
    session_data = session_data.dropna(subset=["micro_return"])

    if session_data.empty:
        logger.warning(f"No valid microreturns found for {data_label}")
        return

    logger.info(f"Plotting microreturns analysis for {data_label}")
    logger.info(f"Data points in session: {len(session_data)}")
    logger.info(
        f"Microreturn stats - Mean: {session_data['micro_return'].mean():.6f}, "
        f"Std: {session_data['micro_return'].std():.6f}, "
        f"Min: {session_data['micro_return'].min():.6f}, "
        f"Max: {session_data['micro_return'].max():.6f}"
    )

    # Convert timestamp to seconds for better readability
    session_data["timestamp_seconds"] = (
        session_data["timestamp"] - session_start
    ) / 1e9

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Time series plot
    ax1.plot(
        session_data["timestamp_seconds"],
        session_data["micro_return"],
        linewidth=0.5,
        alpha=0.7,
        color="blue",
    )
    ax1.set_xlabel("Time (seconds from session start)")
    ax1.set_ylabel("Microreturn (‰)")
    ax1.set_title(f"Microreturn Time Series - {data_label}")
    ax1.grid(True, alpha=0.3)

    # Add summary statistics as text
    stats_text = (
        f"N = {len(session_data):,}\n"
        f'Mean = {session_data["micro_return"].mean():.4f}‰\n'
        f'Std = {session_data["micro_return"].std():.4f}‰\n'
        f'Min = {session_data["micro_return"].min():.4f}‰\n'
        f'Max = {session_data["micro_return"].max():.4f}‰'
    )

    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Histogram
    ax2.hist(
        session_data["micro_return"],
        bins=50,
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    ax2.set_xlabel("Microreturn (‰)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Microreturn Distribution - {data_label}")
    ax2.grid(True, alpha=0.3)

    # Add normal distribution overlay for comparison
    mean_ret = session_data["micro_return"].mean()
    std_ret = session_data["micro_return"].std()
    x_range = np.linspace(
        session_data["micro_return"].min(), session_data["micro_return"].max(), 100
    )
    normal_dist = (1 / (std_ret * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_range - mean_ret) / std_ret) ** 2
    )

    # Scale normal distribution to match histogram
    hist_counts, _ = np.histogram(session_data["micro_return"], bins=50)
    scale_factor = max(hist_counts) / max(normal_dist)
    normal_dist_scaled = normal_dist * scale_factor

    ax2.plot(
        x_range,
        normal_dist_scaled,
        "r--",
        linewidth=2,
        label="Normal Distribution",
        alpha=0.8,
    )
    ax2.legend()

    plt.tight_layout()

    # Save plot
    save_path = (
        save_dir / f'microreturns_analysis_{data_label.lower().replace(" ", "_")}.png'
    )
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Microreturns analysis plot saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save microreturns plot: {e}")
    finally:
        plt.close(fig)


@pytest.mark.comprehensive
def test_market_data_full():
    """
    Main function to test the market data TTSTL system with a single date pair.
    
    This test follows the same logic as market_data_experiments.py but runs only
    configuration ES010_05 on the date pair 06102024 → 06112024 (PAIR05).
    """

    print("🚀 Starting market data TTSTL test for ES010_05 on PAIR05...")
    print("=" * 80)

    # Set global random seeds for reproducibility
    SEED = 19991810  # Same seed as in config
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # PyTorch reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting market data TTSTL test for INTC - Configuration ES010_05 on PAIR05"
    )
    logger.info(f"Global random seed set to: {SEED}")
    logger.info("Test date pair: June 10, 2024 → June 11, 2024 (PAIR05)")

    # Specify the pretrained model directory
    pretraining_dir = Path("outputs/cnn_lstm_pretraining_market_data")

    # Load pretrained CNN-LSTM model and preprocessing components
    logger.info(
        f"Loading pretrained CNN-LSTM model and preprocessing components from {pretraining_dir}..."
    )
    model_data = load_pretrained_model_for_experiments(pretraining_dir)

    if model_data is None:
        logger.error(
            "Failed to load pretrained model and preprocessing components. Please run cnn_lstm_pretraining.py first."
        )
        print("❌ Failed to load pretrained model and preprocessing components.")
        print(
            "💡 Please run cnn_lstm_pretraining.py first to create a pretrained model."
        )
        return

    trained_model, scaler, lower_bounds, upper_bounds = model_data
    logger.info(
        "Pretrained CNN-LSTM model and preprocessing components loaded successfully"
    )

    # Load market data for the test date pair
    global_ed_df, global_ptd_df, local_ed_df, local_ptd_df = load_market_data()

    logger.info("Setting up orchestrator configuration for ES010_05")

    # Get base configuration from experiments script
    cfg = get_base_config()

    # Update configuration with preprocessing components
    cfg.pretrained_scaler = scaler
    cfg.lower_bounds = lower_bounds
    cfg.upper_bounds = upper_bounds

    # Apply ES010_05 configuration (from get_experiment_configs)
    cfg.run_id = "TEST_ES010_05_PAIR05_06102024_06112024"
    cfg.parsimonious_mode = True
    cfg.neural_covariates_only = True
    cfg.manual_covariates_only = False
    cfg.epochs = 2
    cfg.user_defined_variables = ["forward_pass_0"]
    cfg.arma_order = (0, 1)
    cfg.kpss_verification_threshold = 0.8  # Relaxed threshold for forward_pass
    cfg.past_returns = False  # Explicitly use future returns mode

    # Log configuration
    log_orchestrator_config(cfg)

    # Create runs directory
    runs_dir = Path("outputs/runs_market_data_test")
    runs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating orchestrator with base configuration")

    # Create orchestrator
    orchestrator_instance = Orchestrator(runs_dir, cfg)

    # Set the pretrained model to the orchestrator
    logger.info("Setting pretrained CNN-LSTM model to orchestrator")
    set_trained_model_to_orchestrator(orchestrator_instance, trained_model)

    # Generate microreturns analysis plots
    logger.info("Generating microreturns analysis plots")
    plot_microreturns_analysis(
        global_ptd_df,
        cfg.session_start,
        cfg.session_finish,
        runs_dir,
        "June_10_Global_Update_PAIR05",
    )
    plot_microreturns_analysis(
        local_ptd_df,
        cfg.session_start,
        cfg.session_finish,
        runs_dir,
        "June_11_Local_Update_PAIR05",
    )

    logger.info("Starting global update phase with June 10, 2024 data (PAIR05)")
    logger.info(f"Configuration: ES010_05 (forward_pass_0 only, parsimonious mode)")
    logger.info(
        f"Neural network training: {'ENABLED' if cfg.epochs > 0 else 'DISABLED'} ({cfg.epochs} epochs)"
    )
    logger.info(
        f"AdaLASSO hyperparameter selection: {'MSE mode' if cfg.second_stage_mse else 'BIC mode'}"
    )

    try:
        orchestrator_instance.global_update(global_ed_df, global_ptd_df)
        logger.info("Global update completed successfully")
    except Exception as e:
        logger.error(f"Global update failed: {str(e)}")
        logger.exception("Full traceback:")
        print("❌ Global update failed. Check logs for details.")
        return

    logger.info("Starting local adaptation phase with June 11, 2024 data (PAIR05)")

    try:
        orchestrator_instance.local_adapt(local_ed_df, local_ptd_df)
        logger.info("Local adaptation completed successfully")
    except Exception as e:
        logger.error(f"Local adaptation failed: {str(e)}")
        logger.exception("Full traceback:")
        print("❌ Local adaptation failed. Check logs for details.")
        return

    print("🛬 Market data TTSTL test completed successfully!")
    print("=" * 80)
    print(f"📁 Results saved to: {runs_dir}")
    print(f"📝 Check logs for detailed information")
    print(f"🎯 Test configuration: {cfg.run_id}")
    print(f"🔬 Configuration: ES010_05 (forward_pass_0 only, parsimonious mode)")
    print(f"🔬 AdaLASSO mode: {'MSE-based' if cfg.second_stage_mse else 'BIC-based'}")

    logger.info("=" * 80)
    logger.info("MARKET DATA TTSTL TEST COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Test date pair: 06102024 → 06112024 (PAIR05)")
    logger.info(f"Configuration: ES010_05 (forward_pass_0 only, parsimonious mode)")
    logger.info(f"User defined variables: {cfg.user_defined_variables}")
    logger.info(
        f"AdaLASSO hyperparameter selection: {'MSE mode' if cfg.second_stage_mse else 'BIC mode'}"
    )
    logger.info(f"KPSS verification threshold: {cfg.kpss_verification_threshold}")
    logger.info(f"Results directory: {runs_dir}")
    logger.info(f"Run ID: {cfg.run_id}")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_market_data_full()
