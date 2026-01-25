# Standard library imports
import copy
import json
import logging
import os
import pickle
import random
from datetime import datetime
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

# Local imports
from ttstl.architectures.cnn_lstm import CNNLSTM
from ttstl.utils.data_downloader import import_to_pandas
from ttstl.utils.data_preprocessor import create_pretrained_scaler, get_dataset_cnn_lstm


def setup_logging(past_returns=False):
    """Setup logging to see pretraining operations."""

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp and experiment type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_suffix = "_concurrent" if past_returns else "_future"
    log_file = log_dir / f"cnn_lstm_pretraining{experiment_suffix}_{timestamp}.log"

    # Configure logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Write to file
            logging.StreamHandler(),  # Also show in terminal
        ],
    )

    experiment_type = "CONCURRENT RETURNS" if past_returns else "FUTURE RETURNS"
    print(f"📝 {experiment_type} logs will be saved to: {log_file}")


def select_random_gpu():
    """Select a random GPU if available, otherwise use CPU."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # Select a random GPU
            selected_gpu = random.randint(0, num_gpus - 1)
            device = torch.device(f"cuda:{selected_gpu}")
        else:
            device = torch.device("cuda:0")

        logger = logging.getLogger(__name__)
        logger.info(f"Selected GPU {device} out of {num_gpus} available GPU(s)")
        return device
    else:
        logger = logging.getLogger(__name__)
        logger.info("No GPU available, using CPU")
        return torch.device("cpu")


def load_may_training_data():
    """
    Loads May 2024 event-driven data for INTC and splits into train/validation sets.
    Uses May 2, 7, 9 for validation and remaining days for training.

    Returns:
        Tuple of (train_ed_dfs, val_ed_dfs): Lists of individual day DataFrames
        - train_ed_dfs: List of training day DataFrames
        - val_ed_dfs: List of validation day DataFrames
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading May 2024 event-driven training data for INTC")

    # Locate credentials.json
    credentials_path = "credentials.json"
    if not os.path.exists(credentials_path):
        project_root = Path(__file__).resolve().parent.parent.parent
        credentials_path = project_root / "credentials.json"

    # Strict loading from credentials.json
    bucket_name = None
    if os.path.exists(credentials_path):
        try:
            with open(credentials_path, "r") as f:
                creds = json.load(f)
                bucket_name = creds.get("bucket_name")
        except Exception as e:
            logger.error(f"Failed to read credentials.json: {e}")
            raise ValueError(
                f"Failed to read credentials.json: {e}. Please ensure it is a valid JSON file."
            )

    if not bucket_name:
        error_msg = (
            f"Could not find 'bucket_name' in {credentials_path}. "
            "The credentials.json file must contain a 'bucket_name' key. "
            "Please refer to the 'Credentials Configuration' section in README.md."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    gcs_market_data_folder = "market_data/"

    # Get May 2024 trading days that satisfy inclusion criteria (revised)
    may_trading_days = [2, 7, 9, 10, 13, 14, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]

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
                credentials_path=str(credentials_path),
                bucket_name=bucket_name,
            )

            # Add date information for reference
            ed_df["trading_day"] = day  # Add the day number for reference
            ed_df["date"] = f"2024-05-{day:02d}"  # Add full date string for reference

            # Log timestamp range to check for after-hours trading
            min_timestamp = ed_df["timestamp"].min()
            max_timestamp = ed_df["timestamp"].max()

            # Convert timestamps to readable format (assuming nanoseconds since epoch)
            try:
                min_time = pd.to_datetime(min_timestamp, unit="ns").strftime(
                    "%H:%M:%S.%f"
                )
                max_time = pd.to_datetime(max_timestamp, unit="ns").strftime(
                    "%H:%M:%S.%f"
                )
                logger.info(f"  Raw timestamp range: {min_time} to {max_time}")
            except:
                # If timestamp format is different, just log raw values
                logger.info(
                    f"  Raw timestamp range: {min_timestamp} to {max_timestamp}"
                )

            # Filter to regular trading hours: 9:45 AM to 3:45 PM
            # Convert time boundaries to the same timestamp format
            try:
                # Define trading hours boundaries (in nanoseconds)
                open_plus_15 = 35100 * 1_000_000_000  # 9:45 AM
                close_less_15 = 56700 * 1_000_000_000  # 3:45 PM

                # Filter to trading hours using timestamp directly
                trading_hours_mask = (ed_df["timestamp"] >= open_plus_15) & (
                    ed_df["timestamp"] <= close_less_15
                )
                ed_df_filtered = ed_df[trading_hours_mask].copy()

                logger.info(
                    f"  Filtered from {len(ed_df)} to {len(ed_df_filtered)} samples (9:45 AM - 3:45 PM)"
                )

                # Use filtered data
                ed_df = ed_df_filtered

            except Exception as filter_error:
                logger.warning(
                    f"  Could not filter trading hours for May {day}, using all data: {str(filter_error)}"
                )

            # Check if we have data left after filtering
            if len(ed_df) == 0:
                logger.warning(
                    f"  No data remaining after trading hours filter for May {day}"
                )
                continue

            # Add to appropriate list based on validation split
            if day in validation_days:
                val_ed_dfs.append(ed_df)
                logger.info(
                    f"Loaded VALIDATION ED data for May {day}, 2024: {ed_df.shape}"
                )
            else:
                train_ed_dfs.append(ed_df)
                logger.info(
                    f"Loaded TRAINING ED data for May {day}, 2024: {ed_df.shape}"
                )

        except Exception as e:
            logger.warning(f"Could not load data for May {day}, 2024: {str(e)}")
            continue

    if not train_ed_dfs:
        raise ValueError("No May 2024 training data could be loaded")
    if not val_ed_dfs:
        raise ValueError("No May 2024 validation data could be loaded")

    # Return lists of individual day DataFrames instead of concatenated data
    logger.info(
        f"Loaded {len(train_ed_dfs)} training days: {sorted([df['trading_day'].iloc[0] for df in train_ed_dfs])}"
    )
    logger.info(
        f"Loaded {len(val_ed_dfs)} validation days: {sorted([df['trading_day'].iloc[0] for df in val_ed_dfs])}"
    )

    # Log total samples across all days
    total_train_samples = sum(len(df) for df in train_ed_dfs)
    total_val_samples = sum(len(df) for df in val_ed_dfs)
    logger.info(f"Total training samples across all days: {total_train_samples}")
    logger.info(f"Total validation samples across all days: {total_val_samples}")

    return train_ed_dfs, val_ed_dfs


def prepare_daily_training_data(
    train_ed_dfs,
    val_ed_dfs,
    delta=0.5 * 1_000_000_000.0,
    batch_size=32,
    past_returns=False,
):
    """
    Prepare training data for CNN-LSTM using pretrained scaler approach to avoid lookahead bias.

    This function creates a pretrained scaler from all training data, then uses it consistently
    for both training and validation preprocessing. This prevents lookahead bias by ensuring
    validation data preprocessing uses only information available during training.

    Args:
        train_ed_dfs: List of training event-driven DataFrames (individual days)
        val_ed_dfs: List of validation event-driven DataFrames (individual days)
        delta: Time horizon for target computation (same as backbone's sdelta)
        batch_size: Batch size for data loaders
        past_returns: If True, compute past returns instead of future returns

    Returns:
        Tuple of (train_loader, val_loader, feature_names, device, pretrained_scaler, lower_bounds, upper_bounds)
        - train_loader: DataLoader for training data
        - val_loader: DataLoader for validation data
        - feature_names: List of order flow feature names
        - device: PyTorch device being used
        - pretrained_scaler: Fitted StandardScaler for future use
        - lower_bounds: Lower bounds for winsorization
        - upper_bounds: Upper bounds for winsorization
    """
    logger = logging.getLogger(__name__)
    returns_type = "CONCURRENT RETURNS" if past_returns else "FUTURE RETURNS"
    logger.info(
        f"Preparing {returns_type} training data using day-by-day preprocessing with pretrained scaler approach"
    )

    # Step 1: Create pretrained scaler from all training data to avoid lookahead bias
    logger.info("Creating pretrained scaler from all training data...")

    # Combine all training data for scaler fitting
    all_train_data = []
    for train_ed_df in train_ed_dfs:
        df_clean = train_ed_df.drop(["trading_day", "date"], axis=1)
        all_train_data.append(df_clean)

    combined_train_data = pd.concat(all_train_data, ignore_index=True)
    logger.info(
        f"Combined training data for scaler fitting: {len(combined_train_data)} samples from {len(train_ed_dfs)} days"
    )

    # Create pretrained scaler from combined training data
    pretrained_scaler, lower_bounds, upper_bounds = create_pretrained_scaler(
        combined_train_data, quantile=0.005
    )
    logger.info(
        f"Pretrained scaler created with bounds for {len(lower_bounds)} features"
    )

    # Step 2: Process each training day individually using pretrained scaler
    train_datasets = []
    total_train_samples_before = 0
    total_train_samples_after = 0

    returns_desc = "PAST RETURNS" if past_returns else "FUTURE RETURNS"
    logger.info(
        f"Processing training days individually with pretrained scaler ({returns_desc})..."
    )
    for i, train_ed_df in enumerate(train_ed_dfs):
        day = train_ed_df["trading_day"].iloc[0]
        logger.info(
            f"Processing training day {day} ({i+1}/{len(train_ed_dfs)}): {len(train_ed_df)} samples"
        )
        total_train_samples_before += len(train_ed_df)

        # Remove the temporary columns we added for tracking
        df_clean = train_ed_df.drop(["trading_day", "date"], axis=1)

        try:
            train_dataset = get_dataset_cnn_lstm(
                df_clean,
                delta=delta,
                pretrained_scaler=pretrained_scaler,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                train=True,
                past_returns=past_returns,
            )
            train_datasets.append(train_dataset)
            total_train_samples_after += len(train_dataset)
            logger.info(
                f"  Day {day}: {len(df_clean)} → {len(train_dataset)} samples ({len(train_dataset)/len(df_clean)*100:.1f}% retention)"
            )
        except Exception as e:
            logger.warning(f"  Day {day}: Failed to process - {str(e)}")
            continue

    # Step 3: Process each validation day individually using the SAME pretrained scaler
    val_datasets = []
    total_val_samples_before = 0
    total_val_samples_after = 0

    logger.info(
        f"Processing validation days individually with SAME pretrained scaler ({returns_desc})..."
    )
    for i, val_ed_df in enumerate(val_ed_dfs):
        day = val_ed_df["trading_day"].iloc[0]
        logger.info(
            f"Processing validation day {day} ({i+1}/{len(val_ed_dfs)}): {len(val_ed_df)} samples"
        )
        total_val_samples_before += len(val_ed_df)

        # Remove the temporary columns we added for tracking
        df_clean = val_ed_df.drop(["trading_day", "date"], axis=1)

        try:
            val_dataset = get_dataset_cnn_lstm(
                df_clean,
                delta=delta,
                pretrained_scaler=pretrained_scaler,  # Use SAME scaler as training
                lower_bounds=lower_bounds,  # Use SAME bounds as training
                upper_bounds=upper_bounds,  # Use SAME bounds as training
                train=True,
                past_returns=past_returns,
            )
            val_datasets.append(val_dataset)
            total_val_samples_after += len(val_dataset)
            logger.info(
                f"  Day {day}: {len(df_clean)} → {len(val_dataset)} samples ({len(val_dataset)/len(df_clean)*100:.1f}% retention)"
            )
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

    logger.info(
        f"OVERALL TRAINING ({returns_type}): {total_train_samples_before} → {total_train_samples_after} samples ({train_retention:.2f}% retention)"
    )
    logger.info(
        f"OVERALL VALIDATION ({returns_type}): {total_val_samples_before} → {total_val_samples_after} samples ({val_retention:.2f}% retention)"
    )

    # Create data loaders using backbone's configuration
    device = select_random_gpu()

    if device != torch.device("cpu"):
        # GPU configuration: num_workers=4, pin_memory=True
        train_loader = DataLoader(
            combined_train_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )
        val_loader = DataLoader(
            combined_val_dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )
    else:
        # CPU configuration: no num_workers or pin_memory
        train_loader = DataLoader(
            combined_train_dataset, batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            combined_val_dataset, batch_size=batch_size, shuffle=False
        )

    # Get feature names from the first day's data (order flow features contain 'of')
    feature_names = [col for col in train_ed_dfs[0].columns if "of" in col]

    logger.info(
        f"Using {len(feature_names)} order flow features for {returns_type} training"
    )
    logger.info(
        f"Final dataset sizes: Training {len(combined_train_dataset)}, Validation {len(combined_val_dataset)}"
    )

    return (
        train_loader,
        val_loader,
        feature_names,
        device,
        pretrained_scaler,
        lower_bounds,
        upper_bounds,
    )


def train_cnn_lstm_model(
    train_loader, val_loader, device, num_epochs=10, learning_rate=0.0001, seed=19991810
):
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
    criterion = nn.MSELoss(reduction="sum")  # Same as backbone
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Same as backbone

    # Training loop using backbone's logic
    logger.info("Starting training...")
    train_losses = []
    validation_losses = []
    best_model = model
    best_validation_loss = float("inf")
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

        logger.info(
            f"Mean Train Loss: {train_loss:.4f}, Mean Validation Loss: {val_loss:.4f}"
        )

    logger.info(
        f"Training complete. Best Mean Validation Loss: {best_validation_loss:.4f} at epoch {best_epoch}"
    )

    # Store training history (similar to backbone)
    training_history_df = pd.DataFrame(
        {
            "epoch": range(1, num_epochs + 1),
            "train_loss": train_losses,
            "validation_loss": validation_losses,
        }
    )

    # Use the best model
    model = best_model
    model.training_history_df = training_history_df

    logger.info("CNN-LSTM training completed")
    return model


def save_pretrained_model(
    model,
    experiments_dir,
    model_name="pretrained_cnnlstm",
    pretrained_scaler=None,
    lower_bounds=None,
    upper_bounds=None,
    past_returns=False,
):
    """
    Save the pretrained CNN-LSTM model, its training history, and preprocessing components.

    Args:
        model: Trained CNN-LSTM model with training_history_df attribute
        experiments_dir: Directory to save the model and history
        model_name: Base name for saved files
        pretrained_scaler: StandardScaler fitted on training data
        lower_bounds: Dictionary of lower quantile bounds for winsorization
        upper_bounds: Dictionary of upper quantile bounds for winsorization
        past_returns: Whether this model was trained on past returns (concurrent) or future returns
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
    if hasattr(model, "training_history_df"):
        history_path = experiments_dir / f"{model_name}_training_history.csv"
        model.training_history_df.to_csv(history_path, index=False)
        logger.info(f"Saved training history to: {history_path}")
    else:
        history_path = None
        logger.warning("No training history found in model")

    # Save preprocessing components (scaler and bounds)
    scaler_path = None
    bounds_path = None

    if pretrained_scaler is not None:
        scaler_path = experiments_dir / f"{model_name}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(pretrained_scaler, f)
        logger.info(f"Saved pretrained scaler to: {scaler_path}")

    if lower_bounds is not None and upper_bounds is not None:
        bounds_data = {"lower_bounds": lower_bounds, "upper_bounds": upper_bounds}
        bounds_path = experiments_dir / f"{model_name}_bounds.json"
        with open(bounds_path, "w") as f:
            json.dump(bounds_data, f, indent=2)
        logger.info(f"Saved winsorization bounds to: {bounds_path}")

    # Save model metadata
    metadata = {
        "model_type": "CNN-LSTM",
        "architecture": "CNNLSTM",
        "experiment_type": "concurrent_returns" if past_returns else "future_returns",
        "training_completed": datetime.now().isoformat(),
        "has_pretrained_scaler": pretrained_scaler is not None,
        "has_winsorization_bounds": lower_bounds is not None
        and upper_bounds is not None,
        "feature_names": (
            list(lower_bounds.keys()) if lower_bounds is not None else None
        ),
        "num_epochs": (
            len(model.training_history_df)
            if hasattr(model, "training_history_df")
            and model.training_history_df is not None
            else None
        ),
        "best_validation_loss": (
            model.training_history_df["validation_loss"].min()
            if hasattr(model, "training_history_df")
            and model.training_history_df is not None
            and len(model.training_history_df) > 0
            else None
        ),
        "best_epoch": (
            (model.training_history_df["validation_loss"].idxmin() + 1)
            if hasattr(model, "training_history_df")
            and model.training_history_df is not None
            and len(model.training_history_df) > 0
            else None
        ),
        "final_train_loss": (
            model.training_history_df["train_loss"].iloc[-1]
            if hasattr(model, "training_history_df")
            and model.training_history_df is not None
            and len(model.training_history_df) > 0
            else None
        ),
        "final_validation_loss": (
            model.training_history_df["validation_loss"].iloc[-1]
            if hasattr(model, "training_history_df")
            and model.training_history_df is not None
            and len(model.training_history_df) > 0
            else None
        ),
        "past_returns": past_returns,
        "returns_direction": "past" if past_returns else "future",
    }

    metadata_path = experiments_dir / f"{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {metadata_path}")

    logger.info(f"Model artifacts saved to directory: {experiments_dir}")
    return {
        "model_path": model_path,
        "history_path": history_path,
        "metadata_path": metadata_path,
        "scaler_path": scaler_path,
        "bounds_path": bounds_path,
    }


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

    with open(scaler_path, "rb") as f:
        pretrained_scaler = pickle.load(f)
    logger.info(f"Loaded pretrained scaler from: {scaler_path}")

    # Load bounds
    bounds_path = Path(bounds_path)
    if not bounds_path.exists():
        raise FileNotFoundError(f"Bounds file not found: {bounds_path}")

    with open(bounds_path, "r") as f:
        bounds_data = json.load(f)

    lower_bounds = bounds_data["lower_bounds"]
    upper_bounds = bounds_data["upper_bounds"]
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


def main(num_epochs=50, past_returns=False):
    """
    Main function to train and save CNN-LSTM model using May 2024 data.

    Args:
        past_returns: If True, train on concurrent returns instead of future returns
    """
    experiment_type = "CONCURRENT RETURNS" if past_returns else "FUTURE RETURNS"
    print(f"🚀 Starting CNN-LSTM {experiment_type} pretraining with May 2024 data...")
    if past_returns:
        print(
            "📊 This experiment trains on past/concurrent returns instead of future returns"
        )
    print("=" * 100)

    # Set global random seeds for reproducibility
    SEED = 19931002  # Same seed as in config
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # PyTorch reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    setup_logging(past_returns=past_returns)
    logger = logging.getLogger(__name__)
    logger.info(
        f"Starting CNN-LSTM {experiment_type} pretraining for INTC with May 2024 data"
    )
    logger.info(f"Global random seed set to: {SEED}")
    if past_returns:
        logger.info(
            "📊 EXPERIMENT TYPE: CONCURRENT RETURNS (regression on observed returns)"
        )

    # Load training data (May 2024 event-driven only)
    logger.info(
        f"Loading May 2024 event-driven data for {experiment_type} neural network training..."
    )
    train_ed_dfs, val_ed_dfs = load_may_training_data()

    # Prepare training data using pretrained scaler approach to avoid lookahead bias
    (
        train_loader,
        val_loader,
        feature_names,
        device,
        pretrained_scaler,
        lower_bounds,
        upper_bounds,
    ) = prepare_daily_training_data(
        train_ed_dfs,
        val_ed_dfs,
        delta=2 * 1_000_000_000.0,  # Same as sdelta in TTSTL config
        batch_size=32,  # Same as TTSTL config
        past_returns=past_returns,
    )

    # Train the model using backbone's training logic
    trained_model = train_cnn_lstm_model(
        train_loader,
        val_loader,
        device,
        num_epochs=num_epochs,  # Set to desired number of epochs
        learning_rate=0.0001,  # Same as backbone
        seed=SEED,  # Use global seed for reproducibility
    )

    logger.info(f"{experiment_type} neural network training completed successfully")

    # Create experiments directory and save the pretrained model with preprocessing components
    dir_suffix = "_concurrent" if past_returns else ""
    experiments_dir = Path(f"outputs/cnn_lstm_pretraining_market_data{dir_suffix}")
    logger.info(
        f"Saving pretrained CNN-LSTM {experiment_type} model, scaler, bounds, and training history..."
    )

    model_suffix = "_concurrent" if past_returns else ""
    saved_paths = save_pretrained_model(
        trained_model,
        experiments_dir,
        model_name=f"cnnlstm{model_suffix}_may2024_trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        pretrained_scaler=pretrained_scaler,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        past_returns=past_returns,
    )

    logger.info(f"Pretrained {experiment_type} model saved successfully")
    for path_type, path in saved_paths.items():
        if path is not None:
            logger.info(f"  {path_type}: {path}")

    logger.info(f"CNN-LSTM {experiment_type} pretraining completed successfully!")
    logger.info(f"Model artifacts saved to: {experiments_dir}")
    if past_returns:
        logger.info(
            "📊 This model is trained for regression on concurrent/observed returns"
        )

    return saved_paths


if __name__ == "__main__":
    main()
