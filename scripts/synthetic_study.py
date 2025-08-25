from pathlib import Path
import logging
from datetime import datetime

import numpy as np
import pandas as pd

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
    log_file = log_dir / f"synthetic_study_{timestamp}.log"
    
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

def load_synthetic_data(id : int = 0):
    """
    Loads synthetic data for the study from the 'evolving-microstructural-tk-data' bucket.
    The method downloads the 08042025_long_event_driven_recorded_lob_[id].parquet file

    Arguments:
    id : int
        The ID of the synthetic data file to load.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading synthetic data")
    
    credentials_path = 'credentials.json'
    bucket_name = 'evolving-microstructural-tk-data'

    gcs_synthetic_data_folder = 'synthetic_data/'

    ed_df = import_to_pandas(blob_name = gcs_synthetic_data_folder + \
                                 f"08042025_long_event_driven_recorded_lob_{id}.parquet",
                                 credentials_path=credentials_path,
                                 bucket_name=bucket_name)
    
    # Filter for testing purposes
    ed_df = ed_df[ed_df['timestamp'] < 1_000_000]

    logger.info(
        "Loaded event driven synthetic data with shape %s from %s in bucket %s",
        ed_df.shape,
        f"{gcs_synthetic_data_folder}08042025_long_event_driven_recorded_lob_{id}.parquet",
        bucket_name
    )

    ptd_df = import_to_pandas(blob_name = gcs_synthetic_data_folder + \
                                 f"08042025_long_physical_time_driven_recorded_data_{id}.parquet",
                                 credentials_path=credentials_path,
                                 bucket_name=bucket_name)

    # Check if timestamp is an index and convert to column if needed
    if ptd_df.index.name == 'timestamp' or 'timestamp' in ptd_df.index.names:
        logger.info("Converting timestamp from index to column for physical time-driven data")
        ptd_df = ptd_df.reset_index()
    elif 'timestamp' not in ptd_df.columns:
        logger.error("No timestamp found in either index or columns for physical time-driven data")
        raise ValueError("No timestamp found in either index or columns for physical time-driven data")

    ptd_df['micro_return'] = ptd_df['micro'].pct_change() * 1e3
    ptd_df['micro_return'] = ptd_df['micro_return'].shift(-1)
    ptd_df = ptd_df.iloc[:-1]

    ptd_df = ptd_df[ptd_df['timestamp'] < 1_000_000]  # Filter for testing purposes
    

    logger.info(
        "Loaded physical time-driven synthetic data with shape %s from %s in bucket %s",
        ptd_df.shape,
        f"{gcs_synthetic_data_folder}08042025_long_physical_time_driven_recorded_data_{id}.parquet",
        bucket_name
    )

    return ed_df, ptd_df

def log_orchestrator_config(cfg: OrchestratorConfig):
    """Log the orchestrator configuration in a readable format."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("ORCHESTRATOR CONFIGURATION")
    logger.info("=" * 80)
    
    # Convert config to dictionary for easy logging
    config_dict = cfg.__dict__ if hasattr(cfg, '__dict__') else vars(cfg)
    
    for key, value in config_dict.items():
        if isinstance(value, (list, tuple)) and len(value) > 10:
            logger.info(f"{key}: {value}")
            # For long lists/tuples, show only first few and last few elements
            logger.info(f"{key}: [{value[0]}, {value[1]}, ..., {value[-2]}, {value[-1]}] (length: {len(value)})")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 80)

def main():
    """
    Main function to orchestrate the synthetic study.
    """

    print("🚀 Starting synthetic study...")
    print("="*80)

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting synthetic study")

    ed_df_0, ptd_df_0 = load_synthetic_data(0)

    logger.info("Setting up orchestrator class")

    cfg = OrchestratorConfig(
        # General settings
        ticker='SIM',
        #bdelta=5_000_000.0,
        bdelta=500_000.0,  # For testing purposes
        adelta=100_000.0,
        sdelta=100.0,
        seed=19991810,
        session_start=0.0,
        #session_finish=10_000_000.0,
        session_finish=1_000_000.0,  # For testing purposes
        target_col_name='micro_return',
        
        # Backbone knobs
        parsimonious_mode=False,  # Use parsimonious mode for testing
        #epochs=5,
        epochs=2,  # For testing purposes 
        batch_size=32,
        validation_proportion_backbone=0.1,
        buffer_proportion=0.05,
        quantile=0.005, # Windsorization quantile
        adf_rejection_rate_threshold=0.8,
        kpss_verification_threshold=0.2,
        na_verification_threshold=0.1,

        # Interface knobs
        alpha_enet_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        lambda_enet_grid=np.logspace(-4, 2, 25).tolist(),
        tau_adalasso_grid=[0.25, 0.5, 1.0, 1.5, 2.0],
        lambda_adalasso_grid=np.logspace(-4, 2, 25).tolist(),
        validation_proportion_interface=0.2,
        selection_threshold=0.40,
        tolerance_threshold=1e-6,
        ols_significance_level=0.05,
        n_lags=5,

        # Adapter kobs
        update_method='power_law',
        arma_order=(1, 1),
        garch_order=(1, 1),
        dist='sstd',
        model='sGARCH',
        archm=False,
        update_parameters={'alpha' : 3}
    )

    
    runs_dir = Path('runs')
    runs_dir.mkdir(exist_ok=True)

    orchestrator = Orchestrator(runs_dir, cfg)

    log_orchestrator_config(cfg)

    all_global_results = orchestrator.global_update(ed_df_0, ptd_df_0)

    # Access results for each backbone bar
    for result in all_global_results:
        bb_num = result['backbone_bar']
        status = result['backbone_results']['status']
        print(f"Backbone bar {bb_num}: {status}")

    print("🛬 Finishing synthetic study...")

if __name__ == "__main__":
    main()