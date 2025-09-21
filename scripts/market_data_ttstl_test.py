from pathlib import Path
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ttstl import orchestrator
from ttstl.orchestrator import Orchestrator, OrchestratorConfig
from ttstl.utils.data_downloader import import_to_pandas


def setup_logging():
    """Setup logging to see orchestrator operations."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"market_data_test_{timestamp}.log"
    
    # Configure logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Write to file
            logging.StreamHandler()         # Also show in terminal
        ]
    )
    
    print(f"📝 Logs will be saved to: {log_file}")

def load_market_data():
    """
    Loads market data for INTC from the 'evolving-microstructural-tk-data' bucket.
    
    Returns:
        Tuple of (global_update_ed_df, global_update_ptd_df, local_update_ed_df, local_update_ptd_df)
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading market data for INTC")
    
    credentials_path = 'credentials.json'
    bucket_name = 'evolving-microstructural-tk-data'
    gcs_market_data_folder = 'market_data/'

    # Load global update data (June 27, 2024)
    logger.info("Loading global update data (June 27, 2024)")
    global_ed_df = import_to_pandas(
        blob_name=gcs_market_data_folder + "06272024_INTC_event_driven_recorded_lob.parquet",
        credentials_path=credentials_path,
        bucket_name=bucket_name
    )
    
    logger.info(
        "Loaded global update event driven data with shape %s from %s",
        global_ed_df.shape,
        f"{gcs_market_data_folder}06272024_INTC_event_driven_recorded_lob.parquet"
    )

    global_ptd_df = import_to_pandas(
        blob_name=gcs_market_data_folder + "06272024_INTC_physical_time_driven_recorded_data.parquet",
        credentials_path=credentials_path,
        bucket_name=bucket_name
    )

    # Check if timestamp is an index and convert to column if needed
    if global_ptd_df.index.name == 'timestamp' or 'timestamp' in global_ptd_df.index.names:
        logger.info("Converting timestamp from index to column for global update physical time-driven data")
        global_ptd_df = global_ptd_df.reset_index()
    elif 'timestamp' not in global_ptd_df.columns:
        logger.error("No timestamp found in either index or columns for global update physical time-driven data")
        raise ValueError("No timestamp found in either index or columns for global update physical time-driven data")

    # Calculate micro returns for global update data
    global_ptd_df['micro_return'] = global_ptd_df['micro'].pct_change() * 1e3
    global_ptd_df['micro_return'] = global_ptd_df['micro_return'].shift(-1)
    global_ptd_df = global_ptd_df.iloc[:-1]

    logger.info(
        "Loaded global update physical time-driven data with shape %s from %s",
        global_ptd_df.shape,
        f"{gcs_market_data_folder}06272024_INTC_physical_time_driven_recorded_data.parquet"
    )

    # Load local update data (June 28, 2024)
    logger.info("Loading local update data (June 28, 2024)")
    local_ed_df = import_to_pandas(
        blob_name=gcs_market_data_folder + "06282024_INTC_event_driven_recorded_lob.parquet",
        credentials_path=credentials_path,
        bucket_name=bucket_name
    )
    
    logger.info(
        "Loaded local update event driven data with shape %s from %s",
        local_ed_df.shape,
        f"{gcs_market_data_folder}06282024_INTC_event_driven_recorded_lob.parquet"
    )

    local_ptd_df = import_to_pandas(
        blob_name=gcs_market_data_folder + "06282024_INTC_physical_time_driven_recorded_data.parquet",
        credentials_path=credentials_path,
        bucket_name=bucket_name
    )

    # Check if timestamp is an index and convert to column if needed
    if local_ptd_df.index.name == 'timestamp' or 'timestamp' in local_ptd_df.index.names:
        logger.info("Converting timestamp from index to column for local update physical time-driven data")
        local_ptd_df = local_ptd_df.reset_index()
    elif 'timestamp' not in local_ptd_df.columns:
        logger.error("No timestamp found in either index or columns for local update physical time-driven data")
        raise ValueError("No timestamp found in either index or columns for local update physical time-driven data")

    # Calculate micro returns for local update data
    local_ptd_df['micro_return'] = local_ptd_df['micro'].pct_change() * 1e3
    local_ptd_df['micro_return'] = local_ptd_df['micro_return'].shift(-1)
    local_ptd_df = local_ptd_df.iloc[:-1]

    logger.info(
        "Loaded local update physical time-driven data with shape %s from %s",
        local_ptd_df.shape,
        f"{gcs_market_data_folder}06282024_INTC_physical_time_driven_recorded_data.parquet"
    )

    return global_ed_df, global_ptd_df, local_ed_df, local_ptd_df

def log_orchestrator_config(cfg: OrchestratorConfig):
    """Log the orchestrator configuration in a readable format."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Orchestrator configuration")
    logger.info("=" * 80)
    
    # Convert config to dictionary for easy logging
    config_dict = cfg.__dict__ if hasattr(cfg, '__dict__') else vars(cfg)
    
    for key, value in config_dict.items():
        if isinstance(value, (list, tuple)) and len(value) > 10:
            # For long lists/tuples, show only first few and last few elements
            logger.info(f"{key}: [{value[0]}, {value[1]}, ..., {value[-2]}, {value[-1]}] (length: {len(value)})")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 80)

def plot_microreturns_analysis(ptd_df: pd.DataFrame, session_start: float, session_finish: float, 
                              save_dir: Path, data_label: str = ""):
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
        (ptd_df['timestamp'] >= session_start) & 
        (ptd_df['timestamp'] <= session_finish)
    ].copy()
    
    if session_data.empty:
        logger.warning(f"No data found within session bounds for {data_label}")
        return
    
    # Remove NaN values
    session_data = session_data.dropna(subset=['micro_return'])
    
    if session_data.empty:
        logger.warning(f"No valid microreturns found for {data_label}")
        return
    
    logger.info(f"Plotting microreturns analysis for {data_label}")
    logger.info(f"Data points in session: {len(session_data)}")
    logger.info(f"Microreturn stats - Mean: {session_data['micro_return'].mean():.6f}, "
                f"Std: {session_data['micro_return'].std():.6f}, "
                f"Min: {session_data['micro_return'].min():.6f}, "
                f"Max: {session_data['micro_return'].max():.6f}")
    
    # Convert timestamp to seconds for better readability
    session_data['timestamp_seconds'] = (session_data['timestamp'] - session_start) / 1e9
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series plot
    ax1.plot(session_data['timestamp_seconds'], session_data['micro_return'], 
             linewidth=0.5, alpha=0.7, color='blue')
    ax1.set_xlabel('Time (seconds from session start)')
    ax1.set_ylabel('Microreturn (‰)')
    ax1.set_title(f'Microreturn Time Series - {data_label}')
    ax1.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    stats_text = (f'N = {len(session_data):,}\n'
                  f'Mean = {session_data["micro_return"].mean():.4f}‰\n'
                  f'Std = {session_data["micro_return"].std():.4f}‰\n'
                  f'Min = {session_data["micro_return"].min():.4f}‰\n'
                  f'Max = {session_data["micro_return"].max():.4f}‰')
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Histogram
    ax2.hist(session_data['micro_return'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Microreturn (‰)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Microreturn Distribution - {data_label}')
    ax2.grid(True, alpha=0.3)
    
    # Add normal distribution overlay for comparison
    mean_ret = session_data['micro_return'].mean()
    std_ret = session_data['micro_return'].std()
    x_range = np.linspace(session_data['micro_return'].min(), 
                          session_data['micro_return'].max(), 100)
    normal_dist = ((1 / (std_ret * np.sqrt(2 * np.pi))) * 
                   np.exp(-0.5 * ((x_range - mean_ret) / std_ret) ** 2))
    
    # Scale normal distribution to match histogram
    hist_counts, _ = np.histogram(session_data['micro_return'], bins=50)
    scale_factor = max(hist_counts) / max(normal_dist)
    normal_dist_scaled = normal_dist * scale_factor
    
    ax2.plot(x_range, normal_dist_scaled, 'r--', linewidth=2, 
             label='Normal Distribution', alpha=0.8)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    save_path = save_dir / f'microreturns_analysis_{data_label.lower().replace(" ", "_")}.png'
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Microreturns analysis plot saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save microreturns plot: {e}")
    finally:
        plt.close(fig)

def main():
    """
    Main function to orchestrate the market data test.
    """

    print("🚀 Starting market data TTSTL test...")
    print("="*80)

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting market data TTSTL test for INTC")

    # Load market data
    global_ed_df, global_ptd_df, local_ed_df, local_ptd_df = load_market_data()

    logger.info("Setting up orchestrator configuration")

    # Single base configuration with power law update rule in full mode
    cfg = OrchestratorConfig(
        # General settings
        ticker='INTC',
        bdelta=7_200 * 1_000_000_000.0,  # Two hours
        #bdelta=3_000 * 1_000_000_000.0,  # 50 minutes - For testing purposes
        adelta=600 * 1_000_000_000.0,    # Five minutes
        sdelta=0.5 * 1_000_000_000.0,      # 0.5 seconds
        seed=19991810,
        session_start=35_100 * 1_000_000_000.0,  # 9:45 AM in nanoseconds
        session_finish=56_700 * 1_000_000_000.0,  # 15:45 PM in nanoseconds
        #session_finish=38_100 * 1_000_000_000.0,  # 10:35 AM in nanoseconds - For testing purposes
        target_col_name='micro_return',
        
        # Backbone knobs - full mode
        parsimonious_mode=False,  # Full mode (not parsimonious)
        neural_covariates_only=False,
        manual_covariates_only=False,
        epochs=5,           # For testing purposes
        batch_size=32,
        validation_proportion_backbone=0.1,
        buffer_proportion=0.05,
        quantile=0.005,      # Windsorization quantile
        adf_rejection_rate_threshold=0.8,
        kpss_verification_threshold=0.2,
        na_verification_threshold=0.2,
        learning_rate=0.0001,
        correlation_threshold=0.95,

        # Interface knobs
        #user_defined_variables=['forward_pass_0'],
        alpha_enet_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        lambda_enet_grid=np.logspace(-4, 1, 25).tolist(),
        tau_adalasso_grid=[0.25, 0.5, 1.0, 1.5, 2.0],
        lambda_adalasso_grid=np.logspace(0, 1, 25).tolist(),
        validation_proportion_interface=0.2,
        selection_threshold=0.11,  # L = 24, alpha = 0.00005
        tolerance_threshold=1e-6,
        ols_significance_level=0.05,
        wald_significance_level=0.05,
        n_lags=5,

        # Adapter knobs - power law update rule
        update_method='power_law',
        arma_order=(0, 1),
        garch_order=(1, 1),
        dist='nig',
        model='eGARCH',
        archm=False,
        update_parameters={'alpha': 3.0}
    )

    # Create runs directory
    runs_dir = Path('runs_market_data_test')
    runs_dir.mkdir(exist_ok=True)

    # Create orchestrator
    orchestrator = Orchestrator(runs_dir, cfg)

    # Log configuration
    log_orchestrator_config(cfg)

    # Generate microreturns analysis plots
    logger.info("Generating microreturns analysis plots")
    plot_microreturns_analysis(global_ptd_df, cfg.session_start, cfg.session_finish, 
                              runs_dir, "June_27_Global_Update")
    plot_microreturns_analysis(local_ptd_df, cfg.session_start, cfg.session_finish, 
                              runs_dir, "June_28_Local_Update")

    logger.info("Starting global update phase with June 27, 2024 data")
    orchestrator.global_update(global_ed_df, global_ptd_df)
    
    logger.info("Global update completed successfully")

    logger.info("Starting local adaptation phase with June 28, 2024 data")
    orchestrator.local_adapt(local_ed_df, local_ptd_df)
    
    logger.info("Local adaptation completed successfully")

    print("🛬 Market data TTSTL test completed successfully!")
    print("="*80)
    print(f"📁 Results saved to: {runs_dir}")
    print(f"📝 Check logs for detailed information")

if __name__ == "__main__":
    main()