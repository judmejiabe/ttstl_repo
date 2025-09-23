from pathlib import Path
import logging
from datetime import datetime
import itertools

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
    logger.info("Orchestrator configuration")
    logger.info("=" * 80)
    
    # Convert config to dictionary for easy logging
    config_dict = cfg.__dict__ if hasattr(cfg, '__dict__') else vars(cfg)
    
    for key, value in config_dict.items():
        if isinstance(value, (list, tuple)) and len(value) > 10:
            logger.info(f"{key}: [{value[0]}, {value[1]}, ..., {value[-2]}, {value[-1]}] (length: {len(value)})")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("=" * 80)

def get_base_config():
    """Get the base configuration that will be modified for each experiment."""
    return OrchestratorConfig(
        # General settings
        ticker='SIM',
        bdelta=5_000_000.0,  
        adelta=100_000.0,
        sdelta=100.0,
        seed=19991810,
        session_start=0.0,
        session_finish=10_000_000.0,  
        target_col_name='micro_return',
        
        # Backbone knobs (will be modified per experiment)
        parsimonious_mode=True,  # Default, will be overridden
        neural_covariates_only=False,  # Default, will be overridden
        manual_covariates_only=False,  # Default, will be overridden
        epochs=10,  # Default, will be overridden
        batch_size=32,
        validation_proportion_backbone=0.1,
        buffer_proportion=0.05,
        quantile=0.005,
        adf_rejection_rate_threshold=0.8,
        kpss_verification_threshold=0.2,
        na_verification_threshold=0.1,
        learning_rate=0.0001,
        correlation_threshold=0.95,

        # Interface knobs (will be modified per experiment)
        user_defined_variables=None,  # Default, will be overridden
        alpha_enet_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        lambda_enet_grid=np.logspace(-4, 1, 25).tolist(),
        tau_adalasso_grid=[0.25, 0.5, 1.0, 1.5, 2.0],
        lambda_adalasso_grid=np.logspace(0, 1, 25).tolist(),
        validation_proportion_interface=0.2,
        selection_threshold=0.17, # alpha = 0.05 and gamma = 0.00005
        tolerance_threshold=1e-6,
        ols_significance_level=0.05,
        wald_significance_level=0.05,
        n_lags=5,

        # Adapter knobs (will be modified per experiment)
        update_method='power_law',  # Default, will be overridden
        arma_order=(1, 1),
        garch_order=(1, 1),
        dist='sstd',
        model='sGARCH',
        archm=False,
        update_parameters={'alpha': 3.0}  # Will be overridden for different methods
    )

def get_update_method_configs():
    """Get configurations for different update methods."""
    return {
        'M': {
            'update_method': 'myopic',
            'update_parameters': {}
        },
        'SA': {
            'update_method': 'simple_average',
            'update_parameters': {}
        },
        'PL3': {
            'update_method': 'power_law',
            'update_parameters': {'alpha': 3.0}
        },
        'ES010': {
            'update_method': 'exponential_smoothing',
            'update_parameters': {'alpha': 0.1}
        }
    }

def get_covariate_configs():
    """Get configurations for different covariate construction methods."""
    return {
        '01': {
            'description': 'Full mode (all covariates, parsimonious_mode=False)',
            'parsimonious_mode': False,
            'neural_covariates_only': False,
            'manual_covariates_only': False,
            'epochs': 10,
            'user_defined_variables': None
        },
        '02': {
            'description': 'Parsimonious mode (all covariates, parsimonious_mode=True)',
            'parsimonious_mode': True,
            'neural_covariates_only': False,
            'manual_covariates_only': False,
            'epochs': 0,
            'user_defined_variables': None
        },
        '03': {
            'description': 'Manual covariates only (no neural network training)',
            'parsimonious_mode': False,
            'neural_covariates_only': False,
            'manual_covariates_only': True,
            'epochs': 0,
            'user_defined_variables': None
        },
        '04': {
            'description': 'Neural covariates only (no manual features)',
            'parsimonious_mode': False,
            'neural_covariates_only': True,
            'manual_covariates_only': False,
            'epochs': 0,
            'user_defined_variables': None
        },
        '05': {
            'description': 'Only forward_pass_0 as an user defined covariate',
            'parsimonious_mode': True,
            'neural_covariates_only': True,
            'manual_covariates_only': False,
            'epochs': 0,
            'user_defined_variables': ['forward_pass_0']
        },
        '06': {
            'description': 'No covariates',
            'parsimonious_mode': True,
            'neural_covariates_only': False,
            'manual_covariates_only': False,
            'epochs': 0,
            'user_defined_variables': []
        },
        '07': {
            'description': 'Only itv as an user defined covariate',
            'parsimonious_mode': True,
            'neural_covariates_only': False,
            'manual_covariates_only': False,
            'epochs': 0,
            'user_defined_variables': ['itv']
        }
    }

def create_experiment_config(base_config, update_method_config, covariate_config, run_id):
    """Create a configuration for a specific experiment."""
    config = OrchestratorConfig(**vars(base_config))
    
    # Apply update method configuration
    config.update_method = update_method_config['update_method']
    config.update_parameters = update_method_config['update_parameters']
    
    # Apply covariate configuration
    config.parsimonious_mode = covariate_config['parsimonious_mode']
    config.neural_covariates_only = covariate_config['neural_covariates_only']
    config.manual_covariates_only = covariate_config['manual_covariates_only']
    config.epochs = covariate_config['epochs']
    config.user_defined_variables = covariate_config['user_defined_variables']
    
    # Set run_id
    config.run_id = run_id
    
    return config

# Culprit:
def run_experiment(config, base_orchestrator, ed_df_0, ptd_df_0, ed_df_1, ptd_df_1, runs_dir):
    """Run a single experiment with the given configuration, importing backbone from base_orchestrator."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 100)
    logger.info(f"Starting experiment: {config.run_id}")
    logger.info(f"Update method: {config.update_method}")
    logger.info(f"Update parameters: {config.update_parameters}")
    logger.info("=" * 100)
    
    try:
        # Create orchestrator
        orchestrator = Orchestrator(runs_dir, config)
        
        # Import backbone models from the base orchestrator
        if base_orchestrator is not None:
            # Import the backbone models
            orchestrator.import_backbone_models_from_orchestrator(base_orchestrator)
        
        # Log configuration
        log_orchestrator_config(config)
        
        # Run global update (this will skip backbone training since it's imported)
        logger.info(f"Running global update for {config.run_id}")
        all_global_results = orchestrator.global_update(ed_df_0, ptd_df_0)
        
        # Run local adaptation
        logger.info(f"Running local adaptation for {config.run_id}")
        all_local_results = orchestrator.local_adapt(ed_df_1, ptd_df_1)
        
        logger.info(f"Experiment {config.run_id} completed successfully")
        return orchestrator
        
    except Exception as e:
        logger.error(f"Experiment {config.run_id} failed with error: {str(e)}")
        logger.exception("Full traceback:")
        return None

def main():
    """
    Main function to orchestrate the comprehensive synthetic study.
    """
    print("🚀 Starting comprehensive synthetic study...")
    print("="*100)

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive synthetic study")

    # Load data
    logger.info("Loading synthetic data")
    ed_df_0, ptd_df_0 = load_synthetic_data(0)
    ed_df_1, ptd_df_1 = load_synthetic_data(1)

    # Create runs directory
    runs_dir = Path('runs_synthetic_study')
    runs_dir.mkdir(exist_ok=True)

    # Get base configuration and experiment configurations
    base_config = get_base_config()
    update_methods = get_update_method_configs()
    covariate_configs = get_covariate_configs()

    # Generate all experiments
    experiments = []
    for update_key, update_config in update_methods.items():
        for cov_key, cov_config in covariate_configs.items():
            run_id = f"{update_key}_{cov_key}"
            exp_config = create_experiment_config(base_config, update_config, cov_config, run_id)
            experiments.append((run_id, exp_config, cov_config['description'], cov_key))

    logger.info(f"Generated {len(experiments)} experiments to run")
    
    # Print experiment summary
    logger.info("=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    for run_id, config, description, cov_key in experiments:
        # Extract update method info
        update_method = config.update_method
        update_params = config.update_parameters
        
        # Format update method with parameters if they exist
        if update_params:
            param_str = ", ".join([f"{k}={v}" for k, v in update_params.items()])
            update_info = f"{update_method} ({param_str})"
        else:
            update_info = update_method
            
        logger.info(f"{run_id}: Update: {update_info} | {description}")
    logger.info("=" * 80)

    # PHASE 1: Run M_01 experiment first (this will train the backbone)
    logger.info("Running M_01 experiment to train shared backbone neural network")
    
    # Find M_01 experiment
    m01_experiment = None
    remaining_experiments = []
    
    for exp_data in experiments:
        run_id, config, description, cov_key = exp_data
        if run_id == "M_01":
            m01_experiment = exp_data
        else:
            remaining_experiments.append(exp_data)
    
    if m01_experiment is None:
        logger.error("M_01 experiment not found! Cannot proceed.")
        return
    
    # Run M_01 to train the backbone
    run_id, config, description, cov_key = m01_experiment
    logger.info(f"Training backbone with experiment: {run_id}")
    logger.info(f"Description: {description}")
    
    m01_orchestrator = run_experiment(config, None, ed_df_0, ptd_df_0, ed_df_1, ptd_df_1, runs_dir)
    
    if m01_orchestrator is None:
        logger.error("Failed to complete M_01 experiment (backbone training). Exiting.")
        return
    
    logger.info(f"M_01 experiment completed successfully - will use as backbone source")

    # PHASE 2: Run all remaining experiments with backbone import
    logger.info("Running remaining experiments with backbone import from M_01")
    successful_experiments = 1  # M_01 already completed
    failed_experiments = 0
    
    for i, (run_id, config, description, cov_key) in enumerate(remaining_experiments, 2):
        logger.info(f"Running experiment {i}/{len(experiments)}: {run_id}")
        
        # Import backbone from M_01 orchestrator
        result = run_experiment(config, m01_orchestrator, ed_df_0, ptd_df_0, ed_df_1, ptd_df_1, runs_dir)
        
        if result is not None:
            successful_experiments += 1
        else:
            failed_experiments += 1
    
    # Final summary
    logger.info("=" * 100)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 100)
    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Successful: {successful_experiments}")
    logger.info(f"Failed: {failed_experiments}")
    logger.info(f"Success rate: {successful_experiments/len(experiments)*100:.1f}%")
    logger.info(f"Backbone provider: M_01 (Myopic + Full mode)")
    logger.info("=" * 100)

    print("🛬 Finishing comprehensive synthetic study...")

if __name__ == "__main__":
    main()
