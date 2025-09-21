# Standard library imports
import calendar
import copy
import json
import logging
import random
from datetime import datetime
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

# Local imports
from ttstl import orchestrator
from ttstl.architectures.cnn_lstm import CNNLSTM
from ttstl.orchestrator import Orchestrator, OrchestratorConfig
from ttstl.utils.data_downloader import import_to_pandas
from ttstl.utils.data_preprocessor import get_dataset_cnn_lstm


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

def load_may_training_data():
    """
    Loads May 2024 event-driven data for INTC and splits into train/validation sets.
    Uses May 2, 4, 7 for validation and remaining days for training.
    
    Returns:
        Tuple of (train_ed_dfs, val_ed_dfs): Lists of individual day DataFrames
        - train_ed_dfs: List of training day DataFrames 
        - val_ed_dfs: List of validation day DataFrames
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading May 2024 event-driven training data for INTC")
    
    credentials_path = 'credentials.json'
    bucket_name = 'evolving-microstructural-tk-data'
    gcs_market_data_folder = 'market_data/'

    # Get May 2024 trading days that satisfy inclusion criteria (revised)
    may_trading_days = [
        2, 7, 9, 10, 13, 14, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31
    ]

    # Separate validation and training days (using 2, 7, 9 for validation)
    validation_days = [2, 7, 9]
    training_days = [day for day in may_trading_days if day not in validation_days]
    
    train_ed_dfs = []
    val_ed_dfs = []
    
    for day in may_trading_days:
        date_str = f"05{day:02d}2024"
        
        try:
            # Load event driven data only
            ed_blob_name = f"{gcs_market_data_folder}{date_str}_INTC_event_driven_recorded_lob.parquet"
            ed_df = import_to_pandas(
                blob_name=ed_blob_name,
                credentials_path=credentials_path,
                bucket_name=bucket_name
            )
            
            # Add date information for reference
            ed_df['trading_day'] = day  # Add the day number for reference
            ed_df['date'] = f"2024-05-{day:02d}"  # Add full date string for reference
            
            # Log timestamp range to check for after-hours trading
            min_timestamp = ed_df['timestamp'].min()
            max_timestamp = ed_df['timestamp'].max()
            
            # Convert timestamps to readable format (assuming nanoseconds since epoch)
            try:
                min_time = pd.to_datetime(min_timestamp, unit='ns').strftime('%H:%M:%S.%f')
                max_time = pd.to_datetime(max_timestamp, unit='ns').strftime('%H:%M:%S.%f')
                logger.info(f"  Raw timestamp range: {min_time} to {max_time}")
            except:
                # If timestamp format is different, just log raw values
                logger.info(f"  Raw timestamp range: {min_timestamp} to {max_timestamp}")
            
            # Filter to regular trading hours: 9:40 AM to 3:50 PM (15:50)
            # Convert time boundaries to the same timestamp format
            try:
                # Define trading hours boundaries (in nanoseconds)
                open_plus_10 = 34800 * 1_000_000_000    # 9:40 AM
                close_plus_10 = 57600 * 1_000_000_000   # 3:50 PM
                
                # Filter to trading hours using timestamp directly
                trading_hours_mask = (ed_df['timestamp'] >= open_plus_10) & (ed_df['timestamp'] <= close_plus_10)
                ed_df_filtered = ed_df[trading_hours_mask].copy()
                
                logger.info(f"  Filtered from {len(ed_df)} to {len(ed_df_filtered)} samples (9:40 AM - 3:50 PM)")
                
                # Use filtered data
                ed_df = ed_df_filtered
                
            except Exception as filter_error:
                logger.warning(f"  Could not filter trading hours for May {day}, using all data: {str(filter_error)}")
            
            # Check if we have data left after filtering
            if len(ed_df) == 0:
                logger.warning(f"  No data remaining after trading hours filter for May {day}")
                continue
            
            # Add to appropriate list based on validation split
            if day in validation_days:
                val_ed_dfs.append(ed_df)
                logger.info(f"Loaded VALIDATION ED data for May {day}, 2024: {ed_df.shape}")
            else:
                train_ed_dfs.append(ed_df)
                logger.info(f"Loaded TRAINING ED data for May {day}, 2024: {ed_df.shape}")
            
        except Exception as e:
            logger.warning(f"Could not load data for May {day}, 2024: {str(e)}")
            continue
    
    if not train_ed_dfs:
        raise ValueError("No May 2024 training data could be loaded")
    if not val_ed_dfs:
        raise ValueError("No May 2024 validation data could be loaded")
    
    # Return lists of individual day DataFrames instead of concatenated data
    logger.info(f"Loaded {len(train_ed_dfs)} training days: {sorted([df['trading_day'].iloc[0] for df in train_ed_dfs])}")
    logger.info(f"Loaded {len(val_ed_dfs)} validation days: {sorted([df['trading_day'].iloc[0] for df in val_ed_dfs])}")
    
    # Log total samples across all days
    total_train_samples = sum(len(df) for df in train_ed_dfs)
    total_val_samples = sum(len(df) for df in val_ed_dfs)
    logger.info(f"Total training samples across all days: {total_train_samples}")
    logger.info(f"Total validation samples across all days: {total_val_samples}")
    
    return train_ed_dfs, val_ed_dfs

def prepare_daily_training_data(train_ed_dfs, val_ed_dfs, delta=0.5 * 1_000_000_000.0, quantile=0.005, batch_size=32):
    """
    Prepare training data for CNN-LSTM by processing each day individually using backbone's preprocessing logic.
    This avoids issues with temporal gaps between days affecting forward-looking return computation.
    
    Args:
        train_ed_dfs: List of training event-driven DataFrames (individual days)
        val_ed_dfs: List of validation event-driven DataFrames (individual days)
        delta: Time horizon for target computation (same as backbone's sdelta)
        quantile: Winsorization quantile
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, feature_names, device)
    """
    logger = logging.getLogger(__name__)
    logger.info("Preparing training data using day-by-day preprocessing with backbone's logic")
    
    # Process each training day individually
    train_datasets = []
    total_train_samples_before = 0
    total_train_samples_after = 0
    
    logger.info("Processing training days individually...")
    for i, train_ed_df in enumerate(train_ed_dfs):
        day = train_ed_df['trading_day'].iloc[0]
        logger.info(f"Processing training day {day} ({i+1}/{len(train_ed_dfs)}): {len(train_ed_df)} samples")
        total_train_samples_before += len(train_ed_df)
        
        # Remove the temporary columns we added for tracking
        df_clean = train_ed_df.drop(['trading_day', 'date'], axis=1)
        
        try:
            train_dataset = get_dataset_cnn_lstm(
                df_clean, delta=delta, quantile=quantile, train=True
            )
            train_datasets.append(train_dataset)
            total_train_samples_after += len(train_dataset)
            logger.info(f"  Day {day}: {len(df_clean)} → {len(train_dataset)} samples ({len(train_dataset)/len(df_clean)*100:.1f}% retention)")
        except Exception as e:
            logger.warning(f"  Day {day}: Failed to process - {str(e)}")
            continue
    
    # Process each validation day individually
    val_datasets = []
    total_val_samples_before = 0
    total_val_samples_after = 0
    
    logger.info("Processing validation days individually...")
    for i, val_ed_df in enumerate(val_ed_dfs):
        day = val_ed_df['trading_day'].iloc[0]
        logger.info(f"Processing validation day {day} ({i+1}/{len(val_ed_dfs)}): {len(val_ed_df)} samples")
        total_val_samples_before += len(val_ed_df)
        
        # Remove the temporary columns we added for tracking
        df_clean = val_ed_df.drop(['trading_day', 'date'], axis=1)
        
        try:
            val_dataset = get_dataset_cnn_lstm(
                df_clean, delta=delta, quantile=quantile, train=True
            )
            val_datasets.append(val_dataset)
            total_val_samples_after += len(val_dataset)
            logger.info(f"  Day {day}: {len(df_clean)} → {len(val_dataset)} samples ({len(val_dataset)/len(df_clean)*100:.1f}% retention)")
        except Exception as e:
            logger.warning(f"  Day {day}: Failed to process - {str(e)}")
            continue
    
    if not train_datasets:
        raise ValueError("No training datasets could be created")
    if not val_datasets:
        raise ValueError("No validation datasets could be created")
    
    # Combine datasets from all days
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)
    
    # Calculate overall retention rates
    train_retention = total_train_samples_after / total_train_samples_before * 100
    val_retention = total_val_samples_after / total_val_samples_before * 100
    
    logger.info(f"OVERALL TRAINING: {total_train_samples_before} → {total_train_samples_after} samples ({train_retention:.2f}% retention)")
    logger.info(f"OVERALL VALIDATION: {total_val_samples_before} → {total_val_samples_after} samples ({val_retention:.2f}% retention)")
    
    # Create data loaders using backbone's configuration
    device = select_random_gpu()
    
    if device != torch.device('cpu'):
        # GPU configuration: num_workers=4, pin_memory=True
        train_loader = DataLoader(
            combined_train_dataset, batch_size=batch_size, 
            num_workers=4, pin_memory=True, shuffle=False  
        )
        val_loader = DataLoader(
            combined_val_dataset, batch_size=batch_size, 
            num_workers=4, pin_memory=True, shuffle=False  
        )
    else:
        # CPU configuration: no num_workers or pin_memory
        train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get feature names from the first day's data (order flow features contain 'of')
    feature_names = [col for col in train_ed_dfs[0].columns if 'of' in col]
    
    logger.info(f"Using {len(feature_names)} order flow features for training")
    logger.info(f"Final dataset sizes: Training {len(combined_train_dataset)}, Validation {len(combined_val_dataset)}")
    
    return train_loader, val_loader, feature_names, device

def train_cnn_lstm_model(train_loader, val_loader, device, num_epochs=10, learning_rate=0.0001, seed=19991810):
    """
    Train CNN-LSTM model using backbone's architecture and training logic.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        device: PyTorch device to use for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        seed: Random seed for reproducibility
        
    Returns:
        Trained CNN-LSTM model
    """
    logger = logging.getLogger(__name__)
    logger.info("Training CNN-LSTM model using backbone's architecture")
    
    # Set all random seeds for reproducibility
    logger.info(f"Setting random seeds to {seed} for reproducible training")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    
    # Additional reproducibility settings for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Get input dimensions from a sample batch
    sample_batch = next(iter(train_loader))
    input_sample = sample_batch[0]
    input_size = input_sample.shape[-1]  # Number of features
    
    logger.info(f"Input size: {input_size}")
    logger.info(f"Input shape: {input_sample.shape}")
    
    # Initialize model using the same architecture as backbone (no parameters needed)
    model = CNNLSTM()
    
    # Move model to the selected device
    model.to(device)
    logger.info(f"Using device: {device}")
    
    # Use the same loss and optimizer as backbone
    criterion = nn.MSELoss(reduction='sum')  # Same as backbone
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Same as backbone
    
    # Training loop using backbone's logic
    logger.info("Starting training...")
    train_losses = []
    validation_losses = []
    best_model = model
    best_validation_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase - update weights
        model.train()
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute training loss after epoch (without updating weights)
        model.eval()
        total_train_loss = 0.0
        n_train_samples = 0
        
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                batch_size = inputs.size(0)
                n_train_samples += batch_size
                total_train_loss += loss.item()

        logger.info(f"Total Train Loss: {total_train_loss:.4f}")
        logger.info(f"Number of training samples: {n_train_samples}")
        train_loss = total_train_loss / n_train_samples
        
        # Validation phase
        total_val_loss = 0.0
        n_val_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                batch_size = inputs.size(0)
                n_val_samples += batch_size
                total_val_loss += loss.item()
        
        val_loss = total_val_loss / n_val_samples
        
        logger.info(f"Total Validation Loss: {total_val_loss:.4f}")
        logger.info(f"Number of validation samples: {n_val_samples}")
        train_losses.append(train_loss)
        validation_losses.append(val_loss)
        
        # Track best model
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch + 1
        
        logger.info(f"Mean Train Loss: {train_loss:.4f}, Mean Validation Loss: {val_loss:.4f}")
    
    logger.info(f"Training complete. Best Mean Validation Loss: {best_validation_loss:.4f} at epoch {best_epoch}")
    
    # Store training history (similar to backbone)
    training_history_df = pd.DataFrame({
        "epoch": range(1, num_epochs + 1),
        "train_loss": train_losses,
        "validation_loss": validation_losses
    })
    
    # Use the best model
    model = best_model
    model.training_history_df = training_history_df
    
    logger.info("CNN-LSTM training completed")
    return model

def save_pretrained_model(model, experiments_dir, model_name="pretrained_cnnlstm"):
    """
    Save the pretrained CNN-LSTM model and its training history.
    
    Args:
        model: Trained CNN-LSTM model with training_history_df attribute
        experiments_dir: Directory to save the model and history
        model_name: Base name for saved files
    """
    logger = logging.getLogger(__name__)
    
    # Create experiments directory if it doesn't exist
    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(exist_ok=True)
    
    # Save model state dict
    model_path = experiments_dir / f"{model_name}.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model state dict to: {model_path}")
    
    # Save training history in CSV format
    if hasattr(model, 'training_history_df'):
        history_path = experiments_dir / f"{model_name}_training_history.csv"
        model.training_history_df.to_csv(history_path, index=False)
        logger.info(f"Saved training history to: {history_path}")
    else:
        history_path = None
        logger.warning("No training history found in model")
    
    # Save model metadata
    metadata = {
        'model_type': 'CNN-LSTM',
        'architecture': 'CNNLSTM',
        'training_completed': datetime.now().isoformat(),
        'num_epochs': len(model.training_history_df) if hasattr(model, 'training_history_df') and model.training_history_df is not None else None,
        'best_validation_loss': model.training_history_df['validation_loss'].min() if hasattr(model, 'training_history_df') and model.training_history_df is not None and len(model.training_history_df) > 0 else None,
        'best_epoch': (model.training_history_df['validation_loss'].idxmin() + 1) if hasattr(model, 'training_history_df') and model.training_history_df is not None and len(model.training_history_df) > 0 else None,
        'final_train_loss': model.training_history_df['train_loss'].iloc[-1] if hasattr(model, 'training_history_df') and model.training_history_df is not None and len(model.training_history_df) > 0 else None,
        'final_validation_loss': model.training_history_df['validation_loss'].iloc[-1] if hasattr(model, 'training_history_df') and model.training_history_df is not None and len(model.training_history_df) > 0 else None
    }
    
    metadata_path = experiments_dir / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {metadata_path}")
    
    logger.info(f"Model artifacts saved to directory: {experiments_dir}")
    return {
        'model_path': model_path,
        'history_path': history_path,
        'metadata_path': metadata_path
    }

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
        sdelta=0.5 * 1_000_000_000.0,    # 0.5 seconds
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
        update_method='power_law',
        arma_order=(0, 0),
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
            'arma_order': (0, 0)  # Default ARMA order
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
            'description': 'No covariates configuration with ARMA(0,1)',
            'run_id': 'NO_COVARIATES_ARMA01',
            'parsimonious_mode': True,
            'neural_covariates_only': False,
            'manual_covariates_only': False,
            'epochs': 0,  # No training, uses imported backbone
            'user_defined_variables': [],  # Empty list, no covariates
            'arma_order': (0, 1)  # ARMA(0,1) order
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
            logger.info("This preserves learned state while allowing retraining on new data")

            # Copy the trained models from previous orchestrator in this experiment
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
        # ("06052024", "06062024"),  # June 5th (Global Update) & June 6th (Local Adapt)
        # ("06062024", "06102024"),  # June 6th (Global Update) & June 10th (Local Adapt)
        # ("06102024", "06112024"),  # June 10th (Global Update) & June 11th (Local Adapt)
        # ("06112024", "06242024"),  # June 11th (Global Update) & June 24th (Local Adapt)
        # ("06242024", "06252024"),  # June 24th (Global Update) & June 25th (Local Adapt)
        # ("06252024", "06262024"),  # June 25th (Global Update) & June 26th (Local Adapt)
        # ("06262024", "06272024"),  # June 26th (Global Update) & June 27th (Local Adapt)
        # ("06272024", "06282024"),  # June 27th (Global Update) & June 28th (Local Adapt)
    ]
    
    # Prepare experiment configurations
    main_config = create_experiment_config(base_config, experiment_configs['main'])
    benchmark_config = create_experiment_config(base_config, experiment_configs['benchmark'])
    no_covariates_config = create_experiment_config(base_config, experiment_configs['no_covariates'])
    
    logger.info(f"Will process {len(experiment_pairs)} date pairs")
    logger.info("For each date pair: Main (train backbone) → Benchmark (import) → No-covariates (import)")
    
    # Track orchestrators for state transfer between date pairs
    previous_main_orchestrator = None
    
    # Counters for summary
    total_experiments = len(experiment_pairs) * 3  # 3 configs per date pair
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
        
        # Step 2: Run BENCHMARK configuration for this date pair
        logger.info(f"\n--- STEP 2: Running BENCHMARK configuration for pair {pair_idx+1} ---")
        
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
        
        # Step 3: Run NO-COVARIATES configuration for this date pair
        logger.info(f"\n--- STEP 3: Running NO-COVARIATES configuration for pair {pair_idx+1} ---")
        
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

def main():
    """
    Main function to orchestrate the market data experiments.
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

    # Load training data (May 2024 event-driven only)
    logger.info("Loading May 2024 event-driven data for neural network training...")
    train_ed_dfs, val_ed_dfs = load_may_training_data()

    # PHASE 1: Train CNN-LSTM neural network with May 2024 data
    logger.info("PHASE 1: Training CNN-LSTM neural network with May 2024 event-driven data")
    
    # Prepare training data using backbone's preprocessing and day-by-day processing
    train_loader, val_loader, feature_names, device = prepare_daily_training_data(
        train_ed_dfs, val_ed_dfs,
        delta=0.5 * 1_000_000_000.0,  # Same as sdelta in config
        quantile=0.005,  # Same as config
        batch_size=32    # Same as config
    )
    
    # Train the model using backbone's training logic
    trained_model = train_cnn_lstm_model(
        train_loader, val_loader, device,
        num_epochs=0, # Testing purposes
        learning_rate=0.0001,  # Same as backbone
        seed=SEED  # Use global seed for reproducibility
    )
    
    logger.info("Neural network training completed successfully")
    
    # Create experiments directory and save the pretrained model
    experiments_dir = Path('experiments_market_data')
    logger.info("Saving pretrained CNN-LSTM model and training history...")
    
    saved_paths = save_pretrained_model(
        trained_model, 
        experiments_dir, 
        model_name=f"cnnlstm_may2024_trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    logger.info("Pretrained model saved successfully")
    for path_type, path in saved_paths.items():
        if path is not None:
            logger.info(f"  {path_type}: {path}")
    
    # PHASE 2: Run TTSTL experiments with sequential date-by-date processing
    logger.info("PHASE 2: Running TTSTL experiments with sequential date-by-date processing")
    
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
