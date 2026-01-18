# Standard library imports
import copy
import json
import logging
import os
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
from market_data_cnn_lstm_pretraining import load_pretrained_model, load_full_pretrained_setup


def setup_logging(past_returns=False):
    """Setup logging to see orchestrator operations."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp and experiment type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_suffix = "_concurrent" if past_returns else "_future"
    log_file = log_dir / f"market_data_experiments{experiment_suffix}_{timestamp}.log"
    
    # Configure logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Write to file
            logging.StreamHandler()         # Also show in terminal
        ]
    )
    
    experiment_type = "CONCURRENT RETURNS" if past_returns else "FUTURE RETURNS"
    print(f"📝 {experiment_type} logs will be saved to: {log_file}")

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
            
            # Get the target device (where the backbone's current model is)
            target_device = backbone.device
            
            # Create an independent copy of the trained model for this backbone
            # This prevents cross-training issues between different backbone bars
            independent_model = copy.deepcopy(trained_model)
            
            # Move the copied model to the target device to ensure consistency
            independent_model = independent_model.to(target_device)
            
            # Set the model - this is the only attribute the backbone actually has
            backbone.model = independent_model
            
            # Mark backbone as having a pretrained model
            backbone.pretrained = True
            
            models_set += 1
            logger.debug(f"Successfully set trained model to backbone in {stack_key} on device {target_device}")
            
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
        past_returns=False,  # Default to future returns (will be overridden if needed)
        epochs=10,  # Increased for proper training on May data
        batch_size=32,
        validation_proportion_backbone=0.1,
        buffer_proportion=0.05,
        adf_rejection_rate_threshold=0.8,
        kpss_verification_threshold=0.2,
        na_verification_threshold=0.2,
        learning_rate=0.0001,
        correlation_threshold=0.95,
        
        # Preprocessing components for lookahead bias prevention
        pretrained_scaler=None,      # Will be set when loading pretrained model
        lower_bounds=None,           # Will be set when loading pretrained model
        upper_bounds=None,           # Will be set when loading pretrained model

        # Interface knobs (same as market_data_ttstl_test)
        user_defined_variables=None,  # Will be modified per experiment
        second_stage_mse=False,
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

        # Adapter knobs (based on synthetic study results)
        update_method='exponential_smoothing',
        arma_order=(0, 1),
        garch_order=(1, 1),
        dist='nig',
        model='eGARCH',
        archm=False,
        update_parameters={'alpha': 0.1},
        compare_with_cnn_lstm=True  # Enable comparison with raw CNN-LSTM outputs
    )

def get_experiment_configs():
    """Get configurations for covariate experiments based on synthetic study."""
    return {
        '01': {
            'description': 'Full mode (all covariates, parsimonious_mode=False)',
            'run_id': 'ES010_01',
            'parsimonious_mode': False,
            'neural_covariates_only': False,
            'manual_covariates_only': False,
            'epochs': 10,
            'user_defined_variables': None,
            'arma_order': (0, 1)
        },
        '02': {
            'description': 'Parsimonious mode (all covariates, parsimonious_mode=True)',
            'run_id': 'ES010_02',
            'parsimonious_mode': True,
            'neural_covariates_only': False,
            'manual_covariates_only': False,
            'epochs': 0,
            'user_defined_variables': None,
            'arma_order': (0, 1)
        },
        '03': {
            'description': 'Manual covariates only (no neural network training)',
            'run_id': 'ES010_03',
            'parsimonious_mode': False,
            'neural_covariates_only': False,
            'manual_covariates_only': True,
            'epochs': 0,
            'user_defined_variables': None,
            'arma_order': (0, 1)
        },
        '04': {
            'description': 'Neural covariates only (no manual features)',
            'run_id': 'ES010_04',
            'parsimonious_mode': False,
            'neural_covariates_only': True,
            'manual_covariates_only': False,
            'epochs': 0,
            'user_defined_variables': None,
            'arma_order': (0, 1)
        },
        '05': {
            'description': 'Only forward_pass_0 as an user defined covariate (relaxed KPSS for less stationary forward_pass)',
            'run_id': 'ES010_05',
            'parsimonious_mode': True,
            'neural_covariates_only': True,
            'manual_covariates_only': False,
            'epochs': 0,
            'user_defined_variables': ['forward_pass_0'],
            'arma_order': (0, 1),
            'kpss_verification_threshold': 0.8  # Relaxed threshold for forward_pass (tends to be less stationary compared to synthetic study)
        },
        '06': {
            'description': 'Locally Adapting ARMA-GARCH model (no covariates)',
            'run_id': 'ES010_06',
            'parsimonious_mode': True,
            'neural_covariates_only': False,
            'manual_covariates_only': False,
            'epochs': 0,
            'user_defined_variables': [],
            'arma_order': (0, 1)
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
    
    # Apply optional configuration overrides if present
    if 'kpss_verification_threshold' in exp_config:
        config.kpss_verification_threshold = exp_config['kpss_verification_threshold']
    
    return config

def load_date_data(date_str, ticker="INTC", past_returns=False):
    """
    Load data for a specific date, dynamically resolving the GCS bucket.
    Priority: 1. Environment Var -> 2. credentials.json -> 3. Hardcoded Default
        
    Args:
        date_str: Date string in format MMDDYYYY (e.g., "06032024")
        ticker: Stock ticker
        past_returns: If True, compute past returns instead of future returns
        
    Returns:
        Tuple of (ed_df, ptd_df) or (None, None) if loading fails
    """
    logger = logging.getLogger(__name__)
    
    credentials_path = 'credentials.json'
    
    # 1. Try Environment Variable (Best Practice)
    bucket_name = os.environ.get('GCS_BUCKET_NAME')
    
    # 2. Try loading from credentials.json (Convenience)
    if not bucket_name and os.path.exists(credentials_path):
        try:
            with open(credentials_path, 'r') as f:
                creds = json.load(f)
                bucket_name = creds.get('bucket_name')
        except Exception as e:
            logger.debug(f"Could not read bucket_name from credentials: {e}")

    # 3. Fallback to default (Legacy)
    if not bucket_name:
        bucket_name = 'evolving-microstructural-tk-data'
        logger.info(f"Using default bucket: {bucket_name}")
    else:
        logger.info(f"Using configured bucket: {bucket_name}")

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
        
        # Calculate micro returns based on experiment type
        ptd_df['micro_return'] = ptd_df['micro'].pct_change() * 1e3
        
        if past_returns:
            # For concurrent returns: no shift needed, delete first observation
            ptd_df = ptd_df.iloc[1:]
            logger.info(f"Computed concurrent (past) returns for {date_str}")
        else:
            # For future returns: shift by -1, delete last observation
            ptd_df['micro_return'] = ptd_df['micro_return'].shift(-1)
            ptd_df = ptd_df.iloc[:-1]
            logger.info(f"Computed future returns for {date_str}")
        
        logger.info(f"Loaded {date_str} data: ED {ed_df.shape}, PTD {ptd_df.shape}")
        return ed_df, ptd_df
        
    except Exception as e:
        logger.warning(f"Could not load data for {date_str}: {str(e)}")
        return None, None

def run_single_date_pair_experiment(config, global_date, local_date, runs_dir, base_orchestrator=None, previous_orchestrator=None, trained_model=None, past_returns=False):
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
        past_returns: If True, use concurrent returns; if False, use future returns
        
    Returns:
        Orchestrator instance if successful, None if failed
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running single date pair experiment: {config.run_id}")
    logger.info(f"Date pair: {global_date} → {local_date}")
    experiment_type = "CONCURRENT RETURNS" if past_returns else "FUTURE RETURNS"
    logger.info(f"Using {experiment_type}")
    
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
        global_ed_df, global_ptd_df = load_date_data(global_date, past_returns=past_returns)
        
        if global_ed_df is None or global_ptd_df is None:
            logger.error(f"Failed to load global update data for {global_date}")
            return None
        
        # Load data for local adapt date  
        logger.info(f"Loading data for LOCAL ADAPT: {local_date}")
        local_ed_df, local_ptd_df = load_date_data(local_date, past_returns=past_returns)
        
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

def run_independent_ttstl_experiments(base_config, experiment_configs, runs_dir, trained_model, past_returns=False, seed=19991810):
    """
    Run TTSTL experiments date-pair by date-pair following synthetic study design.
    For each date pair: run config 01 (trains backbone), then other configurations import it.
    
    Args:
        base_config: Base orchestrator configuration
        experiment_configs: Dictionary of experiment configurations
        runs_dir: Directory to save results
        trained_model: Pretrained CNN-LSTM model (used as starting point for config 01)
        past_returns: If True, use concurrent returns; if False, use future returns
        seed: Base random seed for generating date-pair-specific seeds
    """
    logger = logging.getLogger(__name__)
    experiment_type = "CONCURRENT RETURNS" if past_returns else "FUTURE RETURNS"
    logger.info(f"Starting date-pair-wise TTSTL experiments with {experiment_type}")
    logger.info("Based on synthetic study: ES010 (alpha=0.1), MA(1), configs 01, 02, 03, 05, 06")
    
    # Define the sequence of date pairs for experiments
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
    
    # Prepare experiment configurations (matching synthetic study order: 01, 02, 03, 04, 05, 06)
    config_01 = create_experiment_config(base_config, experiment_configs['01'])
    config_02 = create_experiment_config(base_config, experiment_configs['02'])
    config_03 = create_experiment_config(base_config, experiment_configs['03'])
    config_04 = create_experiment_config(base_config, experiment_configs['04'])
    config_05 = create_experiment_config(base_config, experiment_configs['05'])
    config_06 = create_experiment_config(base_config, experiment_configs['06'])
    logger.info(f"Will process {len(experiment_pairs)} date pairs")
    logger.info("SEQUENTIAL LEARNING DESIGN:")
    logger.info("  - Date pair 1: Start with pretrained model → train 10 epochs")
    logger.info("  - Date pair 2: Start with date pair 1 results → train 10 epochs") 
    logger.info("  - Date pair 3: Start with date pair 2 results → train 10 epochs")
    logger.info("  - And so on... (progressive learning across time)")
    logger.info("For each date pair: Config 01 (train backbone) → Configs 02, 03, 04, 05, 06 (import backbone)")
    
    # Set random seed for reproducible random seed generation
    np.random.seed(seed)
    
    # Track orchestrators for state transfer between date pairs
    previous_01_orchestrator = None
    
    # Counters for summary
    total_experiments = len(experiment_pairs) * 6  # 6 configs per date pair
    successful_experiments = 0
    failed_experiments = 0
    
    # Process each date pair
    for pair_idx, (global_date, local_date) in enumerate(experiment_pairs):
        logger.info(f"\n{'='*100}")
        logger.info(f"PROCESSING DATE PAIR {pair_idx+1}/{len(experiment_pairs)}: {global_date} → {local_date}")
        logger.info(f"{'='*100}")
        
        # Step 1: Run CONFIG 01 for this date pair (trains backbone)
        logger.info(f"\n--- STEP 1: Running CONFIG 01 for pair {pair_idx+1} ---")
        
        # Generate random seed for this date pair to vary validation split
        # This same seed will be used for all configurations in this date pair
        date_pair_seed = np.random.randint(0, 2**31 - 1)
        logger.info(f"Using randomly generated seed {date_pair_seed} for ALL configurations in date pair {pair_idx+1}")
        
        config_01_pair = OrchestratorConfig(**vars(config_01))
        config_01_pair.run_id = f"{config_01.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_01_pair.seed = date_pair_seed
        
        orchestrator_01 = run_single_date_pair_experiment(
            config_01_pair, global_date, local_date, runs_dir,
            base_orchestrator=None,
            previous_orchestrator=previous_01_orchestrator,
            trained_model=trained_model if pair_idx == 0 else None,
            past_returns=past_returns
        )
        
        if orchestrator_01 is not None:
            successful_experiments += 1
            previous_01_orchestrator = orchestrator_01
            logger.info(f"✓ CONFIG 01 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 01 failed for pair {pair_idx+1}")
            continue
        
        # Step 2: Run CONFIG 02 for this date pair
        logger.info(f"\n--- STEP 2: Running CONFIG 02 for pair {pair_idx+1} ---")
        
        config_02_pair = OrchestratorConfig(**vars(config_02))
        config_02_pair.run_id = f"{config_02.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_02_pair.seed = date_pair_seed  # Use same seed as config 01
        
        orchestrator_02 = run_single_date_pair_experiment(
            config_02_pair, global_date, local_date, runs_dir,
            base_orchestrator=orchestrator_01,
            previous_orchestrator=None,
            trained_model=None,
            past_returns=past_returns
        )
        
        if orchestrator_02 is not None:
            successful_experiments += 1
            logger.info(f"✓ CONFIG 02 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 02 failed for pair {pair_idx+1}")
        
        # Step 3: Run CONFIG 03 for this date pair
        logger.info(f"\n--- STEP 3: Running CONFIG 03 for pair {pair_idx+1} ---")
        
        config_03_pair = OrchestratorConfig(**vars(config_03))
        config_03_pair.run_id = f"{config_03.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_03_pair.seed = date_pair_seed  # Use same seed as config 01
        
        orchestrator_03 = run_single_date_pair_experiment(
            config_03_pair, global_date, local_date, runs_dir,
            base_orchestrator=orchestrator_01,
            previous_orchestrator=None,
            trained_model=None,
            past_returns=past_returns
        )
        
        if orchestrator_03 is not None:
            successful_experiments += 1
            logger.info(f"✓ CONFIG 03 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 03 failed for pair {pair_idx+1}")
        
        # Step 4: Run CONFIG 04 for this date pair
        logger.info(f"\n--- STEP 4: Running CONFIG 04 for pair {pair_idx+1} ---")
        
        config_04_pair = OrchestratorConfig(**vars(config_04))
        config_04_pair.run_id = f"{config_04.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_04_pair.seed = date_pair_seed  # Use same seed as config 01
        
        orchestrator_04 = run_single_date_pair_experiment(
            config_04_pair, global_date, local_date, runs_dir,
            base_orchestrator=orchestrator_01,
            previous_orchestrator=None,
            trained_model=None,
            past_returns=past_returns
        )
        
        if orchestrator_04 is not None:
            successful_experiments += 1
            logger.info(f"✓ CONFIG 04 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 04 failed for pair {pair_idx+1}")
        
        # Step 6: Run CONFIG 06 for this date pair
        logger.info(f"\n--- STEP 6: Running CONFIG 06 for pair {pair_idx+1} ---")
        
        config_05_pair = OrchestratorConfig(**vars(config_05))
        config_05_pair.run_id = f"{config_05.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_05_pair.seed = date_pair_seed  # Use same seed as config 01
        
        orchestrator_05 = run_single_date_pair_experiment(
            config_05_pair, global_date, local_date, runs_dir,
            base_orchestrator=orchestrator_01,
            previous_orchestrator=None,
            trained_model=None,
            past_returns=past_returns
        )
        
        if orchestrator_05 is not None:
            successful_experiments += 1
            logger.info(f"✓ CONFIG 05 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 05 failed for pair {pair_idx+1}")
        
        # Step 5: Run CONFIG 06 for this date pair
        logger.info(f"\n--- STEP 5: Running CONFIG 06 for pair {pair_idx+1} ---")
        
        config_06_pair = OrchestratorConfig(**vars(config_06))
        config_06_pair.run_id = f"{config_06.run_id}_PAIR{pair_idx+1:02d}_{global_date}_{local_date}"
        config_06_pair.seed = date_pair_seed  # Use same seed as config 01
        
        orchestrator_06 = run_single_date_pair_experiment(
            config_06_pair, global_date, local_date, runs_dir,
            base_orchestrator=orchestrator_01,
            previous_orchestrator=None,
            trained_model=None,
            past_returns=past_returns
        )
        
        if orchestrator_06 is not None:
            successful_experiments += 1
            logger.info(f"✓ CONFIG 06 completed for pair {pair_idx+1}")
        else:
            failed_experiments += 1
            logger.error(f"✗ CONFIG 06 failed for pair {pair_idx+1}")
        
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

def load_pretrained_model_for_experiments(pretraining_dir):
    """
    Load a pretrained CNN-LSTM model and preprocessing components for use in TTSTL experiments.
    
    Args:
        pretraining_dir: Path to the directory containing the pretrained model files.
                        Should contain .pth, _bounds.json, and _metadata.json files.
        
    Returns:
        Tuple of (trained_model, scaler, lower_bounds, upper_bounds) or None if loading fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        pretraining_dir = Path(pretraining_dir)
        
        if not pretraining_dir.exists():
            logger.error(f"Pretraining directory not found: {pretraining_dir}")
            logger.info("Please provide a valid pretraining directory path")
            return None
        
        # Find the .pth file in the directory
        model_files = list(pretraining_dir.glob("*.pth"))
        if not model_files:
            logger.error(f"No pretrained model files (.pth) found in {pretraining_dir}")
            logger.info("Please ensure the directory contains a pretrained model")
            return None
        
        if len(model_files) > 1:
            logger.warning(f"Multiple .pth files found in {pretraining_dir}, using the first one")
        
        # Use the first model file found
        model_path = model_files[0]
        logger.info(f"Using pretrained model: {model_path}")
        
        # Extract directory and model name from the full path
        experiments_dir = model_path.parent
        model_name = model_path.stem  # Gets filename without .pth extension
        
        # Load the complete pretrained setup (model + preprocessing components)
        trained_model, scaler, lower_bounds, upper_bounds = load_full_pretrained_setup(
            experiments_dir, model_name
        )
        logger.info(f"Successfully loaded pretrained CNN-LSTM model and preprocessing components from {pretraining_dir}")
        
        return trained_model, scaler, lower_bounds, upper_bounds
        
    except Exception as e:
        logger.error(f"Failed to load pretrained model and preprocessing components: {str(e)}")
        logger.info("Please ensure the directory contains valid pretrained model files")
        return None

def main(past_returns=False):
    """
    Main function to orchestrate the market data TTSTL experiments.
    
    This script focuses only on running TTSTL experiments with different configurations
    using a pretrained CNN-LSTM model. The model should be pretrained using the 
    cnn_lstm_pretraining.py script first.
    
    Args:
        past_returns: If True, run experiments with concurrent returns; if False, use future returns
    """
    experiment_type = "CONCURRENT RETURNS" if past_returns else "FUTURE RETURNS"
    print(f"🚀 Starting market data TTSTL experiments with {experiment_type}...")
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
    
    setup_logging(past_returns)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting market data TTSTL experiments for INTC with {experiment_type}")
    logger.info(f"Global random seed set to: {SEED}")

    # Specify the pretrained model directory
    pretraining_dir = Path('outputs/cnn_lstm_pretraining_market_data')
    
    # Load pretrained CNN-LSTM model and preprocessing components
    logger.info(f"Loading pretrained CNN-LSTM model and preprocessing components from {pretraining_dir}...")
    model_data = load_pretrained_model_for_experiments(pretraining_dir)
    
    if model_data is None:
        logger.error(f"Failed to load pretrained model and preprocessing components from {pretraining_dir}.")
        return
    
    trained_model, scaler, lower_bounds, upper_bounds = model_data
    logger.info(f"Pretrained CNN-LSTM model and preprocessing components loaded successfully for {experiment_type}")
    
    # Create runs directory with experiment type suffix
    experiment_suffix = "_concurrent" if past_returns else ""
    runs_dir = Path(f'outputs/runs_market_data_experiments{experiment_suffix}')
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Get base configuration and experiment configurations
    base_config = get_base_config()
    
    # Update base configuration with experiment type and preprocessing components
    base_config.past_returns = past_returns  # Set the experiment type
    base_config.pretrained_scaler = scaler
    base_config.lower_bounds = lower_bounds
    base_config.upper_bounds = upper_bounds
    
    experiment_configs = get_experiment_configs()

    logger.info("=" * 80)
    logger.info(f"EXPERIMENT SUMMARY - {experiment_type}")
    logger.info("Based on synthetic study results:")
    logger.info("  - Update method: Exponential Smoothing (alpha=0.1)")
    logger.info("  - ARMA order: (0, 1)")
    logger.info("  - Covariate configurations: 01, 02, 03, 04, 05, 06")
    logger.info("=" * 80)
    for exp_key, exp_config in experiment_configs.items():
        logger.info(f"CONFIG {exp_key}: {exp_config['description']}")
        logger.info(f"  - Run ID: {exp_config['run_id']}")
        logger.info(f"  - Parsimonious mode: {exp_config['parsimonious_mode']}")
        logger.info(f"  - User defined variables: {exp_config['user_defined_variables']}")
        logger.info(f"  - Epochs: {exp_config['epochs']}")
        logger.info(f"  - ARMA order: {exp_config['arma_order']}")
    logger.info("=" * 80)

    # Run experiments independently with separate directories
    run_independent_ttstl_experiments(
        base_config, experiment_configs, runs_dir, 
        trained_model, past_returns, seed=SEED
    )
    
    logger.info(f"All TTSTL experiments with {experiment_type} completed successfully!")
    return

if __name__ == "__main__":
    main()