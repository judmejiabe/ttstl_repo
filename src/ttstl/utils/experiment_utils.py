
import logging
import copy
import json
import pickle
import os
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from ttstl.orchestrator import OrchestratorConfig
from ttstl.utils.data_downloader import import_to_pandas
from ttstl.architectures.cnn_lstm import CNNLSTM


def setup_experiment_logging(log_filename_prefix, past_returns=False):
    """
    Setup logging to see operations.
    
    Args:
        log_filename_prefix: Prefix for the log filename (e.g., 'market_data_experiments')
        past_returns: Whether the experiment uses past returns (adds suffix)
    """
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp and experiment type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_suffix = "_concurrent" if past_returns else "_future"
    log_file = log_dir / f"{log_filename_prefix}{experiment_suffix}_{timestamp}.log"
    
    # Configure logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Write to file
            logging.StreamHandler()         # Also show in terminal
        ],
        force=True # Force reconfiguration of logging
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
    
    # Locate credentials.json relative to project root (2 levels up if calling from tests/integration)
    # Or current directory if calling from root
    
    # Try current directory first
    credentials_path = 'credentials.json'
    if not os.path.exists(credentials_path):
        # Try finding it relative to this file
        # src/ttstl/utils/experiment_utils.py -> project_root/credentials.json
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        credentials_path = project_root / 'credentials.json'
    
    logger.debug(f"Looking for credentials at: {credentials_path}")
    

    # Strict loading from credentials.json
    bucket_name = None
    if os.path.exists(credentials_path):
        try:
            with open(credentials_path, 'r') as f:
                creds = json.load(f)
                bucket_name = creds.get('bucket_name')
        except Exception as e:
            logger.error(f"Failed to read credentials.json: {e}")
            raise ValueError(f"Failed to read credentials.json: {e}. Please ensure it is a valid JSON file.")

    if not bucket_name:
        error_msg = (
            f"Could not find 'bucket_name' in {credentials_path}. "
            "The credentials.json file must contain a 'bucket_name' key. "
            "Please refer to the 'Credentials Configuration' section in README.md."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Using configured bucket: {bucket_name}")

    gcs_market_data_folder = 'market_data/'
    
    try:
        # Load event driven data
        ed_blob_name = f"{gcs_market_data_folder}{date_str}_{ticker}_event_driven_recorded_lob.parquet"
        ed_df = import_to_pandas(
            blob_name=ed_blob_name,
            credentials_path=str(credentials_path),
            bucket_name=bucket_name
        )
        
        # Load physical time driven data
        ptd_blob_name = f"{gcs_market_data_folder}{date_str}_{ticker}_physical_time_driven_recorded_data.parquet"
        ptd_df = import_to_pandas(
            blob_name=ptd_blob_name,
            credentials_path=str(credentials_path),
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

def select_random_gpu():
    """Select a random GPU if available, otherwise use CPU."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # Select a random GPU
            selected_gpu = random.randint(0, num_gpus - 1)
            device = torch.device(f'cuda:{selected_gpu}')
        else:
            device = torch.device('cuda:0')
        
        logger = logging.getLogger(__name__)
        logger.info(f"Selected GPU {device} out of {num_gpus} available GPU(s)")
        return device
    else:
        logger = logging.getLogger(__name__)
        logger.info("No GPU available, using CPU")
        return torch.device('cpu')

def load_pretrained_model(model_path, device=None):
    """
    Load a pretrained CNN-LSTM model from disk.
    
    Args:
        model_path: Path to the saved model state dict (.pth file)
        device: PyTorch device to load the model on (if None, will auto-select)
        
    Returns:
        Loaded CNN-LSTM model
    """
    logger = logging.getLogger(__name__)
    
    if device is None:
        device = select_random_gpu()
    
    # Initialize model architecture
    model = CNNLSTM()
    
    # Load the state dict
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move to device
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    logger.info(f"Loaded pretrained CNN-LSTM model from: {model_path}")
    logger.info(f"Model loaded on device: {device}")
    
    return model

def load_preprocessing_components(scaler_path, bounds_path):
    """
    Load the pretrained scaler and winsorization bounds from disk.
    
    Args:
        scaler_path: Path to the saved scaler (.pkl file)
        bounds_path: Path to the saved bounds (.json file)
        
    Returns:
        Tuple of (pretrained_scaler, lower_bounds, upper_bounds)
    """
    logger = logging.getLogger(__name__)
    
    # Load scaler
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        pretrained_scaler = pickle.load(f)
    logger.info(f"Loaded pretrained scaler from: {scaler_path}")
    
    # Load bounds
    bounds_path = Path(bounds_path)
    if not bounds_path.exists():
        raise FileNotFoundError(f"Bounds file not found: {bounds_path}")
    
    with open(bounds_path, 'r') as f:
        bounds_data = json.load(f)
    
    lower_bounds = bounds_data['lower_bounds']
    upper_bounds = bounds_data['upper_bounds']
    logger.info(f"Loaded winsorization bounds from: {bounds_path}")
    logger.info(f"Bounds available for features: {list(lower_bounds.keys())}")
    
    return pretrained_scaler, lower_bounds, upper_bounds

def load_full_pretrained_setup(experiments_dir, model_name, device=None):
    """
    Load the complete pretrained setup: model, scaler, and bounds.
    
    Args:
        experiments_dir: Directory containing the saved artifacts
        model_name: Base name of the saved files
        device: PyTorch device to load the model on (if None, will auto-select)
        
    Returns:
        Tuple of (model, pretrained_scaler, lower_bounds, upper_bounds)
    """
    logger = logging.getLogger(__name__)
    experiments_dir = Path(experiments_dir)
    
    # Load model
    model_path = experiments_dir / f"{model_name}.pth"
    model = load_pretrained_model(model_path, device)
    
    # Load preprocessing components
    scaler_path = experiments_dir / f"{model_name}_scaler.pkl"
    bounds_path = experiments_dir / f"{model_name}_bounds.json"
    pretrained_scaler, lower_bounds, upper_bounds = load_preprocessing_components(
        scaler_path, bounds_path
    )
    
    logger.info("Successfully loaded complete pretrained setup")
    return model, pretrained_scaler, lower_bounds, upper_bounds

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
            # Try resolving relative path if needed?
            # Assuming pretraining_dir is relative to CWD
            # If called from tests/integration, CWD might be tests/integration
            # But the orchestrator setup usually sets runs_dir
            pass
            
        if not pretraining_dir.exists():
             # Try project root relative if CWD failed
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            alt_path = project_root / pretraining_dir
            if alt_path.exists():
                pretraining_dir = alt_path
            else:
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
