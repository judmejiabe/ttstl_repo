# Standard library imports
import copy
import json
import logging
import random
from datetime import datetime
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import torch

# Local imports
from ttstl import orchestrator
from ttstl.orchestrator import Orchestrator, OrchestratorConfig
from ttstl.utils.data_downloader import import_to_pandas

# Import CNN-LSTM pretraining functions
from market_data_cnn_lstm_pretraining import load_pretrained_model


def setup_logging():
    """Setup logging to see orchestrator operations."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"market_data_experiments_{timestamp}.log"
    
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

def set_trained_model_to_orchestrator(orchestrator_obj, trained_model):
    """
    Set the trained CNN-LSTM model to all backbone bars in the orchestrator.
    
    Args:
        orchestrator_obj: TTSTL Orchestrator instance
        trained_model: Trained CNN-LSTM model
    """
    logger = logging.getLogger(__name__)
    logger.info("Setting trained CNN-LSTM model to orchestrator backbone bars")
    
    # Access the stacks from the orchestrator
    if not hasattr(orchestrator_obj, 'stacks') or not orchestrator_obj.stacks:
        logger.error("Orchestrator has no initialized stacks")
        return
    
    # Set the trained model to each backbone through stacks -> interface -> backbone
    models_set = 0
    for stack_key, stack in orchestrator_obj.stacks.items():
        logger.info(f"Setting trained model to backbone in stack {stack_key}")
        
        try:
            # Access backbone through: stack (adapter) -> interface -> backbone
            backbone = stack.interface.backbone
            
            # Create an independent copy of the trained model for this backbone
            # This prevents cross-training issues between different backbone bars
            independent_model = copy.deepcopy(trained_model)
            
            # Set the model - this is the only attribute the backbone actually has
            backbone.model = independent_model
            
            # Mark backbone as having a pretrained model
            backbone.pretrained = True
            
            models_set += 1
            logger.debug(f"Successfully set trained model to backbone in {stack_key}")
            
        except Exception as e:
            logger.error(f"Failed to set trained model to backbone in {stack_key}: {str(e)}")
    
    logger.info(f"Trained model set to {models_set} backbone bars (independent copies)")

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

def get_base_config():
    """Get the base configuration that will be modified for each experiment."""
    return OrchestratorConfig(
        # General settings (same as market_data_ttstl_test)
        ticker='INTC',
        bdelta=7_200 * 1_000_000_000.0,  # Two hours
        adelta=600 * 1_000_000_000.0,    # Ten minutes
        sdelta=2 * 1_000_000_000.0,    # 2 seconds
        seed=19991810,
        session_start=35_100 * 1_000_000_000.0,  # 9:45 AM in nanoseconds
        session_finish=56_700 * 1_000_000_000.0,  # 15:45 PM in nanoseconds
        target_col_name='micro_return',
        
        # Backbone knobs (will be modified per experiment)
        parsimonious_mode=False,  # Default full mode
        neural_covariates_only=False,
        manual_covariates_only=False,
        epochs=10,  # Increased for proper training on May data
        batch_size=32,
        validation_proportion_backbone=0.1,
        buffer_proportion=0.05,
        quantile=0.005,
        adf_rejection_rate_threshold=0.8,
        kpss_verification_threshold=0.2,
        na_verification_threshold=0.2,
        learning_rate=0.0001,
        correlation_threshold=0.95,

        # Interface knobs (same as market_data_ttstl_test)
        user_defined_variables=None,  # Will be modified per experiment
        alpha_enet_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        lambda_enet_grid=np.logspace(-4, 1, 25).tolist(),
        tau_adalasso_grid=[0.25, 0.5, 1.0, 1.5, 2.0],
        lambda_adalasso_grid=np.logspace(0, 1, 25).tolist(),
        validation_proportion_interface=0.2,
        selection_threshold=0.29,  # L = 12, alpha = 0.00005
        tolerance_threshold=1e-6,
        ols_significance_level=0.05,
        wald_significance_level=0.05,
        n_lags=5,

        # Adapter knobs (same as market_data_ttstl_test)
        update_method='simple_average',
        arma_order=(1, 1),
        garch_order=(1, 1),
        dist='nig',
        model='eGARCH',
        archm=False,
        update_parameters={'alpha': 3.0}
    )

def get_experiment_configs():
    """Get configurations for the three experiments."""
    return {
        'main': {
            'description': 'Full mode configuration (all covariates)',
            'run_id': 'MAIN_FULL',
            'parsimonious_mode': False,
            'neural_covariates_only': False,
            'manual_covariates_only': False,
            'epochs': 10,
            'user_defined_variables': None,
            'arma_order': (1, 1)  # Default ARMA order
        },
        'parsimonious': {
            'description': 'Parsimonious mode configuration with ARMA(1,1)',
            'run_id': 'PARSIMONIOUS',
            'parsimonious_mode': True,
            'neural_covariates_only': False,
            'manual_covariates_only': False,
            'epochs': 0,  # No training, uses imported backbone
            'user_defined_variables': None,  # Uses default parsimonious covariate selection
            'arma_order': (1, 1)  # ARMA(1,1) order
        },
        'benchmark': {
            'description': 'Benchmark configuration (forward_pass_0 only, parsimonious)',
            'run_id': 'BENCHMARK_FP0',
            'parsimonious_mode': True,
            'neural_covariates_only': True,
            'manual_covariates_only': False,
            'epochs': 0,  # No training, uses imported backbone
            'user_defined_variables': ['forward_pass_0'],
            'arma_order': (0, 0)  # Default ARMA order
        },
        'no_covariates': {
            'description': 'No covariates configuration with ARMA(1,1)',
            'run_id': 'NO_COVARIATES',
            'parsimonious_mode': True,
            'neural_covariates_only': False,
            'manual_covariates_only': False,
            'epochs': 0,  # No training necessary
            'user_defined_variables': [],  # Empty list, no covariates
            'arma_order': (1, 1)  # ARMA(1,1) order
        }
    }

def create_experiment_config(base_config, exp_config):
    """Create a configuration for a specific experiment."""
    config = OrchestratorConfig(**vars(base_config))
    
    # Apply experiment-specific configuration
    config.run_id = exp_config['run_id']
    config.parsimonious_mode = exp_config['parsimonious_mode']
    config.neural_covariates_only = exp_config['neural_covariates_only']
    config.manual_covariates_only = exp_config['manual_covariates_only']
    config.epochs = exp_config['epochs']
    config.user_defined_variables = exp_config['user_defined_variables']
    config.arma_order = exp_config['arma_order']
    
    return config

def load_date_data(date_str, ticker="INTC"):
    """
    Load data for a specific date.
    
    Args:
        date_str: Date string in format MMDDYYYY (e.g., "06032024")
        ticker: Stock ticker
        
    Returns:
        Tuple of (ed_df, ptd_df) or (None, None) if loading fails
    """
    logger = logging.getLogger(__name__)
    
    credentials_path = 'credentials.json'
    bucket_name = 'evolving-microstructural-tk-data'
    gcs_market_data_folder = 'market_data/'
    
    try:
        # Load event driven data
        ed_blob_name = f"{gcs_market_data_folder}{date_str}_{ticker}_event_driven_recorded_lob.parquet"
        ed_df = import_to_pandas(
            blob_name=ed_blob_name,
            credentials_path=credentials_path,
            bucket_name=bucket_name
        )
        
        # Load physical time driven data
        ptd_blob_name = f"{gcs_market_data_folder}{date_str}_{ticker}_physical_time_driven_recorded_data.parquet"
        ptd_df = import_to_pandas(
            blob_name=ptd_blob_name,
            credentials_path=credentials_path,
            bucket_name=bucket_name
        )
        
        # Handle timestamp index/column conversion for PTD data
        if ptd_df.index.name == 'timestamp' or 'timestamp' in ptd_df.index.names:
            ptd_df = ptd_df.reset_index()
        
        # Calculate micro returns
        ptd_df['micro_return'] = ptd_df['micro'].pct_change() * 1e3
        ptd_df['micro_return'] = ptd_df['micro_return'].shift(-1)
        ptd_df = ptd_df.iloc[:-1]
        
        logger.info(f"Loaded {date_str} data: ED {ed_df.shape}, PTD {ptd_df.shape}")
        return ed_df, ptd_df
        
    except Exception as e:
        logger.warning(f"Could not load data for {date_str}: {str(e)}")
        return None, None

def run_single_date_pair_experiment(config, global_date, local_date, runs_dir, base_orchestrator=None, previous_orchestrator=None, trained_model=None):
    """
    Run a single date pair experiment with the given configuration.
    
    Args:
        config: Orchestrator configuration for this specific date pair
        global_date: Date string for global update (e.g., "05312024")
        local_date: Date string for local adaptation (e.g., "06032024")
        runs_dir: Directory to save results
        base_orchestrator: Orchestrator to import backbone from (for non-main configs)
        previous_orchestrator: Previous orchestrator from same config type (for state transfer)
        trained_model: Pretrained CNN-LSTM model (only for first main config)
        
    Returns:
        Orchestrator instance if successful, None if failed
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running single date pair experiment: {config.run_id}")
    logger.info(f"Date pair: {global_date} → {local_date}")
    
    try:
        # Create orchestrator for this specific date pair
        orchestrator = Orchestrator(runs_dir, config)
        
        # Handle orchestrator state initialization
        if base_orchestrator is not None:
            # For non-main configurations: copy from base orchestrator (from main experiment)
            logger.info(f"Copying orchestrator state from main to {config.run_id}")
            logger.info("This allows full retraining capability while preserving learned state")
            
            # Copy the trained models from base orchestrator
            orchestrator.import_backbone_models_from_orchestrator(base_orchestrator)
            
            
        elif previous_orchestrator is not None:
            # For subsequent date pairs within same experiment: copy from previous orchestrator
            logger.info(f"Copying orchestrator state from previous date pair to {config.run_id}")
            logger.info("This enables sequential learning: previous knowledge + new data training")

            # Copy the trained models from previous orchestrator and KEEP training capability
            orchestrator.import_backbone_models_from_orchestrator(previous_orchestrator, keep_epochs=True)
            
        elif trained_model is not None:
            # For the first date pair of main configuration: use pretrained model
            logger.info(f"Setting initial pretrained CNN-LSTM model to orchestrator {config.run_id}")
            logger.info("This provides the starting point for the main experiment's training sequence")
            set_trained_model_to_orchestrator(orchestrator, trained_model)
        
        # Log configuration for this date pair
        logger.info(f"Configuration for {config.run_id}:")
        logger.info(f"  - Date pair: {global_date} → {local_date}")
        logger.info(f"  - Epochs: {config.epochs}")
        logger.info(f"  - Parsimonious mode: {config.parsimonious_mode}")
        logger.info(f"  - Directory: {orchestrator.run_dir}")
        
        # Load data for global update date
        logger.info(f"Loading data for GLOBAL UPDATE: {global_date}")
        global_ed_df, global_ptd_df = load_date_data(global_date)
        
        if global_ed_df is None or global_ptd_df is None:
            logger.error(f"Failed to load global update data for {global_date}")
            return None
        
        # Load data for local adapt date  
        logger.info(f"Loading data for LOCAL ADAPT: {local_date}")
        local_ed_df, local_ptd_df = load_date_data(local_date)
        
        if local_ed_df is None or local_ptd_df is None:
            logger.error(f"Failed to load local adapt data for {local_date}")
            return None
        
        # Run global update (this will retrain the neural network if epochs > 0)
        logger.info(f"Running GLOBAL UPDATE for {config.run_id} on {global_date}")
        logger.info(f"Neural network retraining: {'ENABLED' if config.epochs > 0 else 'DISABLED'}")
        try:
            global_result = orchestrator.global_update(global_ed_df, global_ptd_df)
            logger.info(f"Global update completed for {config.run_id} on {global_date}")
        except Exception as e:
            logger.error(f"Failed global update for {config.run_id} on {global_date}: {str(e)}")
            return None
        
        # Run local adaptation
        logger.info(f"Running LOCAL ADAPTATION for {config.run_id} on {local_date}")
        try:
            local_result = orchestrator.local_adapt(local_ed_df, local_ptd_df)
            logger.info(f"Local adaptation completed for {config.run_id} on {local_date}")
            
            logger.info(f"Successfully completed {config.run_id}")
            logger.info(f"Results saved to directory: {orchestrator.run_dir}")
            return orchestrator
            
        except Exception as e:
            logger.error(f"Failed local adaptation for {config.run_id} on {local_date}: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create or run experiment {config.run_id}: {str(e)}")
        logger.exception("Full traceback:")
        return None

def run_independent_ttstl_experiments(base_config, experiment_configs, runs_dir, trained_model):
    """
    Run TTSTL experiments date-pair by date-pair.
    For each date pair: run main configuration (trains backbone), then other configurations import it.
    
    Args:
        base_config: Base orchestrator configuration
        experiment_configs: Dictionary of experiment configurations
        runs_dir: Directory to save results
        trained_model: Pretrained CNN-LSTM model (used as starting point for main config)
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting date-pair-wise TTSTL experiments")
    
    # Define the sequence of date pairs for experiments (same as in run_experiment)
    experiment_pairs = [
        ("05312024", "06032024"),  # May 31st (Global Update) & June 3rd (Local Adapt)
        ("06032024", "06052024"),  # June 3rd (Global Update) & June 5th (Local Adapt)
        ("06052024", "06062024"),  # June 5th (Global Update) & June 6th (Local Adapt)
        ("06062024", "06102024"),  # June 6th (Global Update) & June 10th (Local Adapt)
        ("06102024", "06112024"),  # June 10th (Global Update) & June 11th (Local Adapt)
        ("06112024", "06242024"),  # June 11th (Global Update) & June 24th (Local Adapt)
        ("06242024", "06252024"),  # June 24th (Global Update) & June 25th (Local Adapt)
        ("06252024", "06262024"),  # June 25th (Global Update) & June 26th (Local Adapt)
        ("06262024", "06272024"),  # June 26th (Global Update) & June 27th (Local Adapt)
        ("06272024", "06282024"),  # June 27th (Global Update) & June 28th (Local Adapt)
    ]
    
    # Prepare experiment configurations
    main_config = create_experiment_config(base_config, experiment_configs['main'])
    benchmark_config = create_experiment_config(base_config, experiment_configs['benchmark'])
    no_covariates_config = create_experiment_config(base_config, experiment_configs['no_covariates'])
    parsimonious_config = create_experiment_config(base_config, experiment_configs['parsimonious'])
    
    logger.info(f"Will process {len(experiment_pairs)} date pairs")
    logger.info("SEQUENTIAL LEARNING DESIGN (Design B):")
    logger.info("  - Date pair 1: Start with pretrained model → train 10 epochs")
    logger.info("  - Date pair 2: Start with date pair 1 results → train 10 epochs") 
    logger.info("  - Date pair 3: Start with date pair 2 results → train 10 epochs")
    logger.info("  - And so on... (progressive learning across time)")
    logger.info("For each date pair: Main (train backbone) → Parsimonious (import) → Benchmark (import) → No-covariates (import)")
    
    # Track orchestrators for state transfer between date pairs
    previous_main_orchestrator = None
    
    # Counters for summary
    total_experiments = len(experiment_pairs) * 4  # 4 configs per date pair
    successful_experiments = 0
    failed_experiments = 0
    
    # Process each date pair
    for pair_idx, (global_date, local_date) in enumerate(experiment_pairs):
        logger.info(f"\n{'='*100}")
        logger.info(f"PROCESSING DATE PAIR {pair_idx+1}/{len(experiment_pairs)}: {global_date} → {local_date}")
        logger.info(f"{'='*100}")
        
        # Step 1: Run MAIN configuration for this date pair
        logger.info(f"\n--- STEP 1: Running MAIN configuration for pair {pair_idx+1} ---")
        
        # Create config for this specific date pair
        main_pair_config = OrchestratorConfig(**vars(main_config))
        main_pair_config.run_id = f"{main_config.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        
        # Run single date pair for main configuration
        main_orchestrator = run_single_date_pair_experiment(
            main_pair_config, global_date, local_date, runs_dir,
            base_orchestrator=None,  # No base for main
            previous_orchestrator=previous_main_orchestrator,  # Transfer from previous main
            trained_model=trained_model if pair_idx == 0 else None  # Only first pair gets pretrained model
        )
        
        if main_orchestrator is not None:
            successful_experiments += 1
            previous_main_orchestrator = main_orchestrator
            logger.info(f"✓ MAIN configuration completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ MAIN configuration failed for pair {pair_idx+1}")
            # Continue to next pair if main fails
            continue
        
        # Step 2: Run PARSIMONIOUS configuration for this date pair
        logger.info(f"\n--- STEP 2: Running PARSIMONIOUS configuration for pair {pair_idx+1} ---")
        
        parsimonious_pair_config = OrchestratorConfig(**vars(parsimonious_config))
        parsimonious_pair_config.run_id = f"{parsimonious_config.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        
        parsimonious_orchestrator = run_single_date_pair_experiment(
            parsimonious_pair_config, global_date, local_date, runs_dir,
            base_orchestrator=main_orchestrator,  # Import backbone from main
            previous_orchestrator=None,  # Transfer from previous parsimonious
            trained_model=None  # No pretrained model needed
        )
        
        if parsimonious_orchestrator is not None:
            successful_experiments += 1
            logger.info(f"✓ PARSIMONIOUS configuration completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ PARSIMONIOUS configuration failed for pair {pair_idx+1}")
        
        # Step 3: Run BENCHMARK configuration for this date pair
        logger.info(f"\n--- STEP 3: Running BENCHMARK configuration for pair {pair_idx+1} ---")
        
        benchmark_pair_config = OrchestratorConfig(**vars(benchmark_config))
        benchmark_pair_config.run_id = f"{benchmark_config.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        
        benchmark_orchestrator = run_single_date_pair_experiment(
            benchmark_pair_config, global_date, local_date, runs_dir,
            base_orchestrator=main_orchestrator,  # Import backbone from main
            previous_orchestrator=None,  # Transfer from previous benchmark
            trained_model=None  # No pretrained model needed
        )
        
        if benchmark_orchestrator is not None:
            successful_experiments += 1
            logger.info(f"✓ BENCHMARK configuration completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ BENCHMARK configuration failed for pair {pair_idx+1}")
        
        # Step 4: Run NO-COVARIATES configuration for this date pair
        logger.info(f"\n--- STEP 4: Running NO-COVARIATES configuration for pair {pair_idx+1} ---")
        
        no_covariates_pair_config = OrchestratorConfig(**vars(no_covariates_config))
        no_covariates_pair_config.run_id = f"{no_covariates_config.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        
        no_covariates_orchestrator = run_single_date_pair_experiment(
            no_covariates_pair_config, global_date, local_date, runs_dir,
            base_orchestrator=main_orchestrator,  # Import backbone from main
            previous_orchestrator=None,  # Transfer from previous no-covariates
            trained_model=None  # No pretrained model needed
        )
        
        if no_covariates_orchestrator is not None:
            successful_experiments += 1
            logger.info(f"✓ NO-COVARIATES configuration completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ NO-COVARIATES configuration failed for pair {pair_idx+1}")
        
        logger.info(f"\nCompleted all configurations for date pair {pair_idx+1}: {global_date} → {local_date}")
    
    # Final summary
    logger.info("=" * 100)
    logger.info("DATE-PAIR-WISE TTSTL EXPERIMENTS COMPLETED")
    logger.info("=" * 100)
    logger.info(f"Total date pairs processed: {len(experiment_pairs)}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successful: {successful_experiments}")
    logger.info(f"Failed: {failed_experiments}")
    logger.info(f"Success rate: {successful_experiments/total_experiments*100:.1f}%")
    logger.info(f"Results saved to: {runs_dir}")
    logger.info("=" * 100)

def load_pretrained_model_for_experiments(model_path=None):
    """
    Load a pretrained CNN-LSTM model for use in TTSTL experiments.
    
    Args:
        model_path: Path to the pretrained model file. If None, will look for 
                   the most recent model in cnn_lstm_pretraining_market_data directory.
        
    Returns:
        Loaded CNN-LSTM model or None if loading fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        if model_path is None:
            # Look for the most recent pretrained model
            pretraining_dir = Path('cnn_lstm_pretraining_market_data')
            if not pretraining_dir.exists():
                logger.error(f"Pretraining directory not found: {pretraining_dir}")
                logger.info("Please run cnn_lstm_pretraining.py first to create a pretrained model")
                return None
            
            # Find the most recent .pth file
            model_files = list(pretraining_dir.glob("*.pth"))
            if not model_files:
                logger.error(f"No pretrained model files (.pth) found in {pretraining_dir}")
                logger.info("Please run cnn_lstm_pretraining.py first to create a pretrained model")
                return None
            
            # Use the most recent model file
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using most recent pretrained model: {model_path}")
        
        # Load the pretrained model
        trained_model = load_pretrained_model(model_path)
        logger.info("Successfully loaded pretrained CNN-LSTM model for experiments")
        return trained_model
        
    except Exception as e:
        logger.error(f"Failed to load pretrained model: {str(e)}")
        logger.info("Please ensure you have run cnn_lstm_pretraining.py successfully")
        return None

def main():
    """
    Main function to orchestrate the market data TTSTL experiments.
    
    This script focuses only on running TTSTL experiments with different configurations
    using a pretrained CNN-LSTM model. The model should be pretrained using the 
    cnn_lstm_pretraining.py script first.
    """
    print("🚀 Starting market data TTSTL experiments...")
    print("="*100)

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
    logger.info("Starting market data TTSTL experiments for INTC")
    logger.info(f"Global random seed set to: {SEED}")

    # Load pretrained CNN-LSTM model
    logger.info("Loading pretrained CNN-LSTM model...")
    trained_model = load_pretrained_model_for_experiments()
    
    if trained_model is None:
        logger.error("Failed to load pretrained model. Please run cnn_lstm_pretraining.py first.")
        return
    
    logger.info("Pretrained CNN-LSTM model loaded successfully")
    
    # Create runs directory
    runs_dir = Path('runs_market_data_experiments')
    runs_dir.mkdir(exist_ok=True)

    # Get base configuration and experiment configurations
    base_config = get_base_config()
    experiment_configs = get_experiment_configs()

    logger.info("=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    for exp_key, exp_config in experiment_configs.items():
        logger.info(f"{exp_key.upper()}: {exp_config['description']}")
        logger.info(f"  - Run ID: {exp_config['run_id']}")
        logger.info(f"  - Parsimonious mode: {exp_config['parsimonious_mode']}")
        logger.info(f"  - User defined variables: {exp_config['user_defined_variables']}")
        logger.info(f"  - Epochs: {exp_config['epochs']} (using pre-trained model)")
    logger.info("=" * 80)

    # Run experiments independently with separate directories
    run_independent_ttstl_experiments(
        base_config, experiment_configs, runs_dir, 
        trained_model
    )
    
    logger.info("All TTSTL experiments completed successfully!")
    return

if __name__ == "__main__":
    main()