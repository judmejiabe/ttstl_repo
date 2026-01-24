"""
TTSTL Orchestrator - Coordinates Backbone, Interface, and Adapter modules.

This module provides the main orchestration logic for the Two Time-Scale Transfer Learning
framework, managing the execution of backbone training, interface variable selection,
and adapter parameter updates across trading sessions.
"""

# Standard library imports
import copy
import json
import logging
import os
import random
import secrets
import sys
import warnings
import yaml
from datetime import datetime
from dataclasses import dataclass, asdict, field
from matplotlib import transforms
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Local imports
from ttstl.backbone import Backbone
from ttstl.interface import Interface  
from ttstl.adapter import Adapter
from ttstl.architectures.cnn_lstm import CNNLSTM


@dataclass
class OrchestratorConfig:
    """
    Configuration for the TTSTL Orchestrator.
    
    This dataclass contains all configuration parameters for the Two Time-Scale Transfer
    Learning framework, including backbone neural network training, interface variable
    selection, and adapter ARMAX-GARCH modeling.
    
    Attributes
    ----------
    ticker : str
        Asset identifier (e.g., "MSFT", "AAPL")
    bdelta : float
        Backbone time bar duration in seconds
    adelta : float
        Adapter time bar duration in seconds
    sdelta : float
        Sampling time bar duration in seconds
    seed : int
        Random number generator seed for reproducibility
    session_start : float
        Trading session start timestamp
    session_finish : float
        Trading session end timestamp
    target_col_name : str
        Name of target variable column (default: 'micro_return')
    architecture : str
        Neural network architecture ('cnn_lstm' supported)
    parsimonious_mode : bool
        Use reduced feature set for backbone
    neural_covariates_only : bool
        Exclude manual/stationary features
    manual_covariates_only : bool
        Exclude neural network features
    past_returns : bool
        Use concurrent returns instead of forward-looking returns
    epochs : int
        Number of training epochs
    batch_size : int
        Training batch size
    validation_proportion_backbone : float
        Validation split proportion
    buffer_proportion : float
        Buffer zone proportion for temporal splits
    adf_rejection_rate_threshold : float
        Minimum ADF test rejection rate for stationarity
    kpss_verification_threshold : float
        Maximum KPSS test rejection rate
    na_verification_threshold : float
        Maximum proportion of missing values allowed
    learning_rate : float
        Optimizer learning rate
    correlation_threshold : float
        Maximum correlation for feature redundancy filtering
    pretrained_scaler : Optional[StandardScaler]
        Pre-fitted StandardScaler (prevents lookahead bias)
    lower_bounds : Optional[Dict[str, float]]
        Windsorization lower bounds per feature
    upper_bounds : Optional[Dict[str, float]]
        Windsorization upper bounds per feature
    alpha_enet_grid : Optional[List[float]]
        Elastic net alpha values (0=ridge, 1=lasso)
    lambda_enet_grid : Optional[List[float]]
        Elastic net regularization strengths
    tau_adalasso_grid : Optional[List[float]]
        Adaptive lasso tau exponents
    lambda_adalasso_grid : Optional[List[float]]
        Adaptive lasso regularization strengths
    user_defined_variables : Optional[List[str]]
        Bypass variable selection with fixed list
    second_stage_mse : bool
        Use MSE instead of BIC for hyperparameter selection
    validation_proportion_interface : float
        Validation split for variable selection
    selection_threshold : float
        Minimum coefficient magnitude for variable retention
    tolerance_threshold : float
        Convergence tolerance for optimization
    ols_significance_level : float
        P-value threshold for OLS coefficient testing
    wald_significance_level : float
        P-value threshold for Wald testing
    n_lags : int
        Number of autoregressive lags
    update_method : str
        Parameter update rule ('myopic', 'simple_average', 'power_law',
        'exponential_smoothing', 'newton_raphson', 'fischer_scoring',
        'gradient_ascent')
    arma_order : Tuple[int, int]
        ARMA(p,q) order for mean equation
    garch_order : Tuple[int, int]
        GARCH(m,s) order for volatility equation
    dist : str
        Error distribution ('norm', 'std', 'ged', 'snorm', 'sstd', 'sged')
    model : str
        GARCH variant ('sGARCH', 'eGARCH', 'gjrGARCH', etc.)
    archm : bool
        Include ARCH-in-mean term
    update_parameters : dict
        Additional parameters for update method (e.g., alpha, regularization)
    compare_with_cnn_lstm : bool
        Generate CNN-LSTM benchmark comparison metrics
    run_id : Optional[str]
        Override auto-generated run identifier
    """
    
    # Identity & timing
    ticker: str                      # e.g. "MSFT"
    bdelta: float                    # Defines backbone time bar length
    adelta: float                    # Defines adapter time bar length
    sdelta: float                    # Defines sampling time bar length
    seed: int                        # RNG seed
    session_start: float             # start of the session in the same units as the recorded timestamp
    session_finish: float            # end of the session in the same units as the recorded timestamp
    target_col_name: str = 'micro_return' # Target variable column name in all datasets

    # Backbone knobs
    architecture: str = 'cnn_lstm'
    parsimonious_mode: bool = False
    neural_covariates_only: bool = False  # Use only neural covariates (no manual/stationary features)
    manual_covariates_only: bool = False  # Use only manual/stationary covariates (no neural features)
    past_returns: bool = False           # Whether to use past returns (concurrent returns) instead of future returns
    epochs: int = 12 
    batch_size: int = 32
    validation_proportion_backbone: float = 0.1
    buffer_proportion: float = 0.05
    adf_rejection_rate_threshold : float = 0.8
    kpss_verification_threshold : float = 0.2
    na_verification_threshold : float = 0.1
    learning_rate: float = 0.0001  # Learning rate for backbone optimizer
    correlation_threshold: float = 0.95  # Correlation threshold for feature filtering
    
    # Preprocessing components for lookahead bias prevention
    pretrained_scaler: Optional[StandardScaler] = None  # Pretrained scaler from preprocessing
    lower_bounds: Optional[Dict[str, float]] = None     # Lower windsorization bounds
    upper_bounds: Optional[Dict[str, float]] = None     # Upper windsorization bounds

    # Interface knobs
    # Hyperparameter grids (using default factory approach)
    alpha_enet_grid: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    lambda_enet_grid: Optional[List[float]] = field(default_factory=lambda: np.logspace(-4, 2, 20).tolist())
    tau_adalasso_grid: Optional[List[float]] = field(default_factory=lambda: [0.25, 0.5, 1.0, 1.5, 2.0])
    lambda_adalasso_grid: Optional[List[float]] = field(default_factory=lambda: np.logspace(-4, 2, 20).tolist())

    # Variable selection
    user_defined_variables: Optional[List[str]] = None  # Bypass adalasso with user-defined variables
    second_stage_mse: bool = False  # Use MSE instead of BIC for adalasso hyperparameter selection
    
    validation_proportion_interface : float = 0.2
    selection_threshold: float = 0.40
    tolerance_threshold: float = 1e-6
    ols_significance_level: float = 0.05
    wald_significance_level: float = 0.05
    n_lags: int = 5

    # Adapter knobs
    update_method : str = 'simple_average'
    arma_order : Tuple[int, int] = (1, 1)
    garch_order : Tuple[int, int] = (1, 1)
    dist : str = 'norm' 
    model : str = 'sGARCH'
    archm : bool = False
    update_parameters : dict = field(default_factory=dict)
    compare_with_cnn_lstm : bool = False  # Whether to compare TTSTL performance with raw CNN-LSTM outputs

    # Optional run-id override
    run_id: Optional[str] = None

    


class Orchestrator:
    """
    TTSTL Orchestrator - Coordinates backbone, interface, and adapter modules.
    
    Manages the execution flow of the Two Time-Scale Transfer Learning framework,
    including:
    - Training neural network backbone models on event-driven data
    - Variable selection through interface module using adaptive lasso
    - Adaptive ARMAX-GARCH parameter estimation via adapter module
    - Artifact persistence and provenance tracking
    - Model interpretability through SHAP-style contribution visualizations
    
    The orchestrator operates in two primary modes:
    
    1. Global Update Mode:
       - Trains backbone neural networks on in-sample data
       - Selects relevant features via interface
       - Fits ARMAX-GARCH models across adapter bars
       - Generates performance metrics and interpretability plots
       - Saves all artifacts to structured directories
       
    2. Local Adaptation Mode:
       - Uses frozen backbone models for out-of-sample evaluation
       - Applies pre-fitted scalers and target statistics from global update
       - Updates adapter parameters on new data
       - Generates forecasts and performance metrics
       - Prevents lookahead bias through consistent preprocessing
    
    Key Features:
    - Multi-scale temporal decomposition (backbone, adapter, sampling bars)
    - Reproducible random seeding across Python, NumPy, and PyTorch
    - Automatic GPU selection when CUDA is available
    - Comprehensive artifact generation (plots, data, metadata)
    - JSON manifest tracking for provenance
    
    Attributes:
        runs_root (Path): Root directory for all experiment runs
        cfg (OrchestratorConfig): Configuration object with all parameters
        run_slug (str): Auto-generated unique identifier for this run
        run_dir (Path): Directory for this specific run's artifacts
        stacks (Dict): Dictionary of Adapter instances (one per backbone bar)
        scalers (Dict): Pre-fitted StandardScalers for each backbone bar
        target_scales (Dict): Target variable statistics for each backbone bar
        logger (Logger): Logging instance for orchestrator
    """

    def __init__(self, runs_root: Path = None, cfg: OrchestratorConfig = None) -> None:
        """
        Initialize the Orchestrator.
        
        Args:
            runs_root: Root directory for all runs (defaults to 'runs' at project root)
            cfg: Configuration object containing all parameters
        """
        if runs_root is None:
            # Default to 'runs' directory at project root
            # Find project root by looking for setup.py or pyproject.toml
            current_dir = Path(__file__).parent
            project_root = current_dir
            while project_root.parent != project_root:
                if (project_root / "setup.py").exists() or (project_root / "pyproject.toml").exists():
                    break
                project_root = project_root.parent
            runs_root = project_root / "runs"
        
        self.runs_root = Path(runs_root)
        self.cfg = cfg
        self.logger = logging.getLogger("ttstl.orchestrator")
        
        # Initialize backbone counter
        self._bb_idx = -1

        # Placeholder for the number of backbone bars
        self.total_backbone_bars = None

        # Flag for skipping the backbone training
        self.skip_backbone_training = False
        
        # Seed RNGs for reproducibility
        self._seed_rngs()
        
        # Create run directory and write initial artifacts
        self.run_slug = self._create_run_slug()
        self.run_dir = self.runs_root / self.run_slug
        self._create_run_directory()
        
        # Initialize components
        self._initialize_components()
        self._initialize_scalers()
        self._initialize_target_scales()
        
        self.logger.info(f"Orchestrator initialized with run: {self.run_slug}")

    def _seed_rngs(self) -> None:
        """Seed all random number generators for reproducibility with independent seeds."""
        # Generate independent seeds from base seed using Python's random module
        base_rng = random.Random(self.cfg.seed)
        
        # Generate independent seeds for each library
        python_seed = self.cfg.seed  # Use base seed for Python
        numpy_seed = base_rng.randint(0, 2**32 - 1)  # Independent seed for NumPy
        torch_seed = base_rng.randint(0, 2**32 - 1)  # Independent seed for PyTorch
        
        # Seed Python RNG
        random.seed(python_seed)
        
        # Seed NumPy
        np.random.seed(numpy_seed)
        
        # Seed PyTorch
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(torch_seed)
            torch.cuda.manual_seed_all(torch_seed)
            # For deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.logger.debug(f"Seeded RNGs - Base: {self.cfg.seed}, Python: {python_seed}, NumPy: {numpy_seed}, PyTorch: {torch_seed}")

    def _create_run_slug(self) -> str:
        """Create a reproducible run slug."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        run_id = self.cfg.run_id or secrets.token_hex(3)
        
        slug = (f"{date_str}_{self.cfg.ticker}"
                f"_bdelta{int(self.cfg.bdelta)}"
                f"_adelta{int(self.cfg.adelta)}"
                f"_sdelta{int(self.cfg.sdelta)}"
                f"_seed{self.cfg.seed}"
                f"_id{run_id}")
        
        return slug

    def _create_run_directory(self) -> None:
        """Create the run directory and write initial artifacts."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Write config.yaml with tuple conversion
        config_path = self.run_dir / "config.yaml"
        config_dict = asdict(self.cfg)
        # Convert tuples to lists for YAML serialization
        config_dict['arma_order'] = list(config_dict['arma_order'])
        config_dict['garch_order'] = list(config_dict['garch_order'])
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        # Write initial manifest.json
        manifest = {
            "run_slug": self.run_slug,
            "ticker": self.cfg.ticker,
            "bdelta": self.cfg.bdelta,
            "adelta": self.cfg.adelta,
            "sdelta": self.cfg.sdelta,
            "seed": self.cfg.seed,
            "session_start": self.cfg.session_start,
            "session_finish": self.cfg.session_finish,
            "created_at": datetime.now().isoformat(),
            "code_provenance": self._get_code_provenance(),
            "environment": self._get_environment_info(),
            "artifacts": []
        }
        
        manifest_path = self.run_dir / "manifest.json"
        self._atomic_write_json(manifest_path, manifest)

    def _initialize_scalers(self) -> None:
        """Initialize scalers for the physical time data."""
        # Calculate session length and number of backbone bars
        session_length = self.cfg.session_finish - self.cfg.session_start
        num_backbone_bars = int(np.rint(session_length / self.cfg.bdelta))
        residual_length = session_length - (num_backbone_bars * self.cfg.bdelta)
        
        # Check if we need to add one more bar or warn about dropped interval
        if residual_length > 0:
            residual_percentage = residual_length / self.cfg.bdelta
            if residual_percentage > 0.01:  # More than 1%
                num_backbone_bars += 1
        
        self.logger.info(f"Creating {num_backbone_bars} scaler placeholders")
        
        # Initialize stacks dictionary
        keys = [f"backbone_{i}" for i in range(num_backbone_bars)]
        self.scalers = {key : None for key in keys}

        self.logger.info(f"Successfully initialized {len(self.scalers)} scaler placeholders")

    def _initialize_target_scales(self) -> None:
        """Initialize scales for the target variable. These metrics are to be transferred in local adapt mode"""
        session_length = self.cfg.session_finish - self.cfg.session_start
        num_backbone_bars = int(np.rint(session_length / self.cfg.bdelta))
        residual_length = session_length - (num_backbone_bars * self.cfg.bdelta)
        
        # Check if we need to add one more bar or warn about dropped interval
        if residual_length > 0:
            residual_percentage = residual_length / self.cfg.bdelta
            if residual_percentage > 0.01:  # More than 1%
                num_backbone_bars += 1
        
        self.logger.info(f"Creating {num_backbone_bars} target variable scales placeholders")
        
        # Initialize stacks dictionary
        keys = [f"backbone_{i}" for i in range(num_backbone_bars)]
        self.target_scales = {key : None for key in keys}

        self.logger.info(f"Successfully initialized {len(self.target_scales)} target variable scales placeholders")

    def _initialize_components(self) -> None:
        """Initialize multiple Backbone-Interface-Adapter stacks for the trading session."""
        # Calculate session length and number of backbone bars
        session_length = self.cfg.session_finish - self.cfg.session_start
        num_backbone_bars = int(np.rint(session_length / self.cfg.bdelta))
        residual_length = session_length - (num_backbone_bars * self.cfg.bdelta)
        
        # Check if we need to add one more bar or warn about dropped interval
        if residual_length > 0:
            residual_percentage = residual_length / self.cfg.bdelta
            if residual_percentage < 0.01:  # Less than 1%
                self.logger.warning(
                    f"Small interval of length {residual_length:.2f} ({residual_percentage*100:.3f}% of bdelta) "
                    f"will be dropped from the session"
                )
            else:
                num_backbone_bars += 1
                self.logger.warning(
                    f"Last backbone bar will have smaller length: {residual_length:.2f} "
                    f"({residual_percentage*100:.1f}% of bdelta)"
                )
        
        self.logger.info(f"Creating {num_backbone_bars} backbone-interface-adapter stacks for session")
        
        # Select a random GPU if CUDA is available
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            selected_gpu = random.randint(0, num_gpus - 1)
            device = torch.device(f'cuda:{selected_gpu}')
            self.logger.info(f"Selected GPU {selected_gpu} out of {num_gpus} available GPUs")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU (no CUDA available)")
        
        # Initialize stacks dictionary
        self.stacks = {}
        
        # Create backbone-interface-adapter stacks
        for i in range(num_backbone_bars):
            # Initialize neural network model based on architecture
            if self.cfg.architecture == 'cnn_lstm':
                model = CNNLSTM().to(device)
            else:
                raise ValueError(f"Unsupported architecture: {self.cfg.architecture}")
            
            # Initialize Backbone
            backbone = Backbone(
                model=model,
                delta_adapter=self.cfg.adelta,
                delta_sample=self.cfg.sdelta,
                parsimonious_mode=self.cfg.parsimonious_mode,
                neural_covariates_only=self.cfg.neural_covariates_only,
                manual_covariates_only=self.cfg.manual_covariates_only,
                pretrained_scaler=self.cfg.pretrained_scaler,
                lower_bounds=self.cfg.lower_bounds,
                upper_bounds=self.cfg.upper_bounds,
                past_returns=self.cfg.past_returns
            )
            
            # Initialize Interface with backbone module and hyperparameter grids
            interface = Interface(
                backbone_module=backbone,
                alpha_enet_grid=self.cfg.alpha_enet_grid,
                lambda_enet_grid=self.cfg.lambda_enet_grid,
                tau_adalasso_grid=self.cfg.tau_adalasso_grid,
                lambda_adalasso_grid=self.cfg.lambda_adalasso_grid,
                user_defined_variables=self.cfg.user_defined_variables,
                session_start=self.cfg.session_start
            )
            
            # Initialize Adapter with interface and adapter configuration
            adapter = Adapter(
                interface=interface,
                update_method=self.cfg.update_method,
                arma_order=self.cfg.arma_order,
                garch_order=self.cfg.garch_order,
                dist=self.cfg.dist,
                model=self.cfg.model,
                archm=self.cfg.archm,
                parameters={},  # Start with empty parameters
                compare_with_cnn_lstm=self.cfg.compare_with_cnn_lstm
            )
            
            # Store only the adapter (backbone and interface are accessible through it)
            stack_key = f"backbone_{i}"
            self.stacks[stack_key] = adapter
            
            self.logger.debug(f"Initialized stack {stack_key}")
        
        self.logger.info(f"Successfully initialized {len(self.stacks)} backbone-interface-adapter stacks")
    
    def import_backbone_models_from_orchestrator(self, 
                                                 source_orchestrator: 'Orchestrator', 
                                                 keep_epochs: bool = False) -> None:
        """
        Import backbone modules from another orchestrator instance.
        
        This method copies the trained backbone modules from a source orchestrator
        and integrates them into the current orchestrator's stacks, effectively
        enabling transfer learning by reusing pre-trained backbone models.
        
        Args:
            source_orchestrator: Another Orchestrator instance with trained backbone modules
            
        Raises:
            ValueError: If source orchestrator doesn't have trained backbones
            RuntimeError: If there's a mismatch in stack configurations
        """
        if not hasattr(source_orchestrator, 'stacks') or not source_orchestrator.stacks:
            raise ValueError("Source orchestrator has no initialized stacks")
        
        # Check if source orchestrator has completed training
        if source_orchestrator._bb_idx < 0:
            raise ValueError("Source orchestrator has not completed any backbone training")
        
        # Log the import operation
        self.logger.info("=" * 80)
        self.logger.info("Importing backbone modules from source orchestrator")
        self.logger.info("=" * 80)
        self.logger.info(f"Source orchestrator run: {source_orchestrator.run_slug}")
        self.logger.info(f"Target orchestrator run: {self.run_slug}")
        self.logger.info(f"Target stacks available: {len(self.stacks)}")
        
        # Validate stack compatibility
        source_stack_count = len(source_orchestrator.stacks)
        target_stack_count = len(self.stacks)
        
        if source_stack_count != target_stack_count:
            self.logger.warning(
                f"Stack count mismatch: source has {source_stack_count}, "
                f"target has {target_stack_count}. Will import available stacks."
            )
        
        # Import backbone modules from source stacks
        imported_count = 0
        failed_count = 0
        
        for stack_key in self.stacks.keys():
            if stack_key in source_orchestrator.stacks:
                try:
                    # Get source and target backbones
                    source_backbone = source_orchestrator.stacks[stack_key].interface.backbone
                    target_backbone = self.stacks[stack_key].interface.backbone

                    # Copy the trained model and move to target device
                    # First, get the target device (where the target backbone's model currently is)
                    target_device = target_backbone.device
                    
                    # Deep copy the source model
                    copied_model = copy.deepcopy(source_backbone.model)
                    
                    # Move the copied model to the target device to ensure consistency
                    copied_model = copied_model.to(target_device)
                    
                    # Assign the copied and relocated model
                    target_backbone.model = copied_model
                    # Device remains unchanged (already set correctly at target_device)

                    # Copy learned state for sequential learning
                    target_backbone.pretrained = True  # Mark as having pretrained weights

                    if not keep_epochs:
                        self.cfg.epochs = 0  # Disable further training

                    imported_count += 1
                    self.logger.debug(f"Successfully imported backbone for {stack_key} to device {target_device}")
                    
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Failed to import backbone for {stack_key}: {str(e)}")
            else:
                failed_count += 1
                self.logger.warning(f"Stack {stack_key} not found in source orchestrator")
        
        # Set the skip backbone training flag
        # self.skip_backbone_training = True
        
        # Log import summary
        self.logger.info(f"Backbone import completed:")
        self.logger.info(f"Successfully imported: {imported_count} backbones")
        self.logger.info(f"Failed imports: {failed_count}")
        self.logger.info(f'Epochs set to zero to avoid further neural network training')
        # self.logger.info(f"Skip backbone training flag set to: {self.skip_backbone_training}")
        self.logger.info("=" * 80)
        
        if imported_count == 0:
            raise RuntimeError("No backbone modules were successfully imported")

    def _create_backbone_directories(self, mode: str) -> Path:
        """Create directory structure for a backbone bar."""
        bb_dir = self.run_dir / f"bb_{self._bb_idx:03d}" / mode
        
        # Create subdirectories based on mode
        if mode == "global_update":
            # Global update needs full subdirectory structure for training artifacts
            subdirs = [
                "backbone/architecture/checkpoints",
                "backbone/stationarity", 
                "backbone/correlation",
                "interface",
                "adapter"
            ]
        elif mode == "local_adapt":
            # Local adapt only needs basic backbone directory for evaluation results
            subdirs = [
                "backbone",
                "interface",
                "adapter"
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        for subdir in subdirs:
            (bb_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        return bb_dir

    def _add_time_bar_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add backbone_bar and adapter_bar columns if not present."""
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        
        # Adjust timestamps relative to session start
        adjusted_timestamps = df['timestamp'] - self.cfg.session_start
        
        # Add backbone_bar column - partitions session into backbone-sized chunks
        if 'backbone_bar' not in df.columns:
            df['backbone_bar'] = ((adjusted_timestamps + 1e-6) // self.cfg.bdelta).astype(int)
        
        # Add adapter_bar column - partitions session into adapter-sized chunks
        if 'adapter_bar' not in df.columns:
            df['adapter_bar'] = ((adjusted_timestamps + 1e-6) // self.cfg.adelta).astype(int)

        # Add sampling_time_bar column - used for OFI calculations
        # Sampling time bar as a name can be misleading as it is a timestamp rather than an index. 
        # Consider changing with a lot of care
        if 'sampling_time_bar' not in df.columns:
            df['sampling_time_bar'] = np.rint((np.floor((df['timestamp'] - self.cfg.session_start + 1e-6) / self.cfg.sdelta) * self.cfg.sdelta) + self.cfg.session_start)

        return df

    def global_update(self, 
                      event_df: pd.DataFrame, 
                      phys_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Perform global update for all backbone bars in the session automatically.
        
        Args:
            event_df: Full event-driven trading session data
            phys_df: Full physical-time trading session data
            
        Returns:
            List of results for each backbone bar processed
        """
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("Starting a global update for the full trading session")
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)

        
        # Continue Here: event_df_processed and phys_df_processed should be filtered before getting processed. 

        event_df = event_df[(event_df['timestamp'] >= self.cfg.session_start) & 
                           (event_df['timestamp'] <= self.cfg.session_finish)].copy()
        phys_df = phys_df[(phys_df['timestamp'] >= self.cfg.session_start) & 
                         (phys_df['timestamp'] <= self.cfg.session_finish)].copy()

        # Add time bar columns if not present
        event_df_processed = self._add_time_bar_columns(event_df.copy())
        phys_df_processed = self._add_time_bar_columns(phys_df.copy())
        
        # Get all backbone bars to process
        backbone_bars = sorted(event_df_processed['backbone_bar'].unique())
        total_backbone_bars = len(backbone_bars)
        self.total_backbone_bars = total_backbone_bars

        self.logger.info(f"Processing {total_backbone_bars} backbone bars")

        all_results = []
        
        for i, bb in enumerate(backbone_bars):
            self.logger.info("="*80)
            self.logger.info("="*80)
            self.logger.info(f"Processing backbone bar {bb} ({i+1}/{total_backbone_bars})")
            self.logger.info("="*80)
            self.logger.info("="*80)
            
            # Filter data for this specific backbone bar
            event_bb_data = event_df_processed[event_df_processed['backbone_bar'] == bb].copy()
            phys_bb_data = phys_df_processed[phys_df_processed['backbone_bar'] == bb].copy()
            
            # Run global update for this backbone bar
            try:
                results = self._global_update_single_bar(event_bb_data, phys_bb_data)
                all_results.append(results)
                self.logger.info("="*80)
                self.logger.info("="*80)
                self.logger.info(f"Backbone bar {bb} completed successfully")
                self.logger.info("="*80)
                self.logger.info("="*80)
            except Exception as e:
                self.logger.error(f"Backbone bar {bb} failed: {str(e)}")
                raise RuntimeError(f"Global update failed for backbone bar {bb}: {str(e)}")
            
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info(f"Global update session completed successfully for {total_backbone_bars} backbone bars")
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        return all_results

    def _global_update_single_bar(self, 
                                 event_df: pd.DataFrame, 
                                 phys_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform global update for a single backbone bar.
        
        Note: This method is intended for internal use by the orchestrator.
        
        Args:
            event_df: Event-driven data for one backbone bar
            phys_df: Physical-time data for one backbone bar
            
        Returns:
            Dict containing key metrics, selected variables, and artifact paths
        """
        # Increment backbone index and create directory structure
        self._bb_idx += 1
        bb_dir = self._create_backbone_directories("global_update")

        
        try:
            # Add time bar columns if not present
            event_df_processed = self._add_time_bar_columns(event_df.copy())
            phys_df_processed = self._add_time_bar_columns(phys_df.copy())
            

            if not self.skip_backbone_training:
                # Step 1: Backbone training and feature extraction
                backbone_results = self._run_backbone_global_update(
                    event_df_processed, phys_df_processed, bb_dir
                )
            
            else:
                backbone_results = self._run_backbone_local_adapt(
                    event_df_processed, phys_df_processed, bb_dir
                )
            
            # Step 2: Interface global selection
            interface_results = self._run_interface_global_update(
                event_df_processed, phys_df_processed, bb_dir, backbone_results
            )
            
            # Step 2.5: Fit scalers and target scales for use in local adaptation
            self._fit_scalers_and_target_scales(event_df_processed, phys_df_processed)
            
            # Step 3: Adapter fitting
            adapter_results = self._run_adapter_global_update(
                event_df_processed, phys_df_processed, bb_dir
            )
            
            # Step 4: Update manifest and return results
            results = {
                "backbone_bar": self._bb_idx,
                "backbone_results": backbone_results,
                "interface_results": interface_results,
                "adapter_results": adapter_results,
                "artifact_paths": self._get_artifact_paths(bb_dir)
            }
            
            self._update_manifest("global_update", results)

            self.logger.info("="*80)
            self.logger.info("="*80)
            self.logger.info(f"Global update completed for backbone bar {self._bb_idx:03d}")
            self.logger.info("="*80)
            self.logger.info("="*80)
            return results
            
        except Exception as e:
            self.logger.error(f"Global update failed for backbone bar {self._bb_idx:03d}: {str(e)}")
            raise RuntimeError(f"Cannot proceed without global update artifacts: {str(e)}")

    def _fit_scalers_and_target_scales(self, event_df: pd.DataFrame, phys_df: pd.DataFrame) -> None:
        """
        Fit scalers and compute target variable scales for use in local adaptation.
        
        This method extracts the scaler fitting logic from _run_adapter_global_update
        so it can be reused in local adaptation without recomputing the scalers.
        
        Args:
            event_df: Event-driven data for the current backbone bar
            phys_df: Physical-time data for the current backbone bar
        """
        self.logger.info(f"Fitting scalers and target scales for backbone bar {self._bb_idx}")
        
        # Get the current adapter from the stack
        current_adapter = self.stacks[f"backbone_{self._bb_idx}"]
        if current_adapter is None:
            self.logger.error("No adapter available for fitting scalers")
            raise RuntimeError("No adapter available for fitting scalers")
        
        # Create and fit scaler
        scaler = StandardScaler()
        augmented_df = current_adapter.augment_data(
            event_df, 
            phys_df.drop(columns=['backbone_bar', 'adapter_bar']),
            target_col_name=self.cfg.target_col_name
        )
        
        # Fit scaler on all covariates except timestamp, target, and time bar columns
        # This must match the filtering logic in adapter.augment_data()
        covariate_columns = [col for col in augmented_df.columns 
                           if col not in ['timestamp', self.cfg.target_col_name, 'adapter_bar', 'backbone_bar']]
        scaler.fit(augmented_df[covariate_columns])
        self.scalers[f"backbone_{self._bb_idx}"] = scaler

        # Compute and store target variable statistics
        self.target_scales[f"backbone_{self._bb_idx}"] = {
            "mean": augmented_df[self.cfg.target_col_name].mean(),
            "std": augmented_df[self.cfg.target_col_name].std()
        }
        
        self.logger.info(f"Fitted scaler for {len(covariate_columns)} covariates and computed target scales ")

    def _run_backbone_global_update(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path) -> Dict[str, Any]:
        """Run backbone global update step."""
        
        # Filter data for the current backbone bar
        current_bb_event_df = event_df[event_df['backbone_bar'] == self._bb_idx].copy()
        current_bb_phys_df = phys_df[phys_df['backbone_bar'] == self._bb_idx].copy()
        
        self.logger.info(f"Backbone bar {self._bb_idx}: {len(current_bb_event_df)} event samples, {len(current_bb_phys_df)} physical time samples")
        
        # Check if we have sufficient data
        if len(current_bb_phys_df) < 10:
            self.logger.warning(f"Insufficient physical time data for backbone bar {self._bb_idx}: {len(current_bb_phys_df)} samples")
            return {"status": "insufficient_data", "event_samples": len(current_bb_event_df), "phys_samples": len(current_bb_phys_df)}
        
        # Get the current backbone for this backbone bar
        current_backbone = self.stacks[f"backbone_{self._bb_idx}"].interface.backbone
        
        # Create criterion and optimizer
        # Use configurable learning rate from orchestrator configuration
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(current_backbone.model.parameters(), lr=self.cfg.learning_rate)
        
        try:
            # Run actual backbone global update with configuration parameters
            backbone_result = current_backbone.global_update(
                event_driven_backbone_time_bar_data=current_bb_event_df,
                physical_time_driven_backbone_time_bar_data=current_bb_phys_df,
                criterion=criterion,
                optimizer=optimizer,
                epochs=self.cfg.epochs,
                validation_proportion=self.cfg.validation_proportion_backbone,
                buffer_proportion=self.cfg.buffer_proportion,
                batch_size=self.cfg.batch_size,
                adf_rejection_rate_threshold=self.cfg.adf_rejection_rate_threshold,
                kpss_verification_threshold=self.cfg.kpss_verification_threshold,
                na_verification_threshold=self.cfg.na_verification_threshold,
                kpss_drop=True,
                correlation_threshold=self.cfg.correlation_threshold,
                target_col_name=self.cfg.target_col_name,
                save_dir=str(bb_dir / "backbone"),  # Enable checkpoints and nn_weights saving
                session_start=self.cfg.session_start
            )
            
            return {
                "status": "completed", 
                "backbone_result": backbone_result,
                "event_samples": len(current_bb_event_df),
                "phys_samples": len(current_bb_phys_df),
                "artifacts_created": 11  # Updated count: added nn_weights.npz and checkpoints
            }
            
        except Exception as e:
            self.logger.error(f"Backbone global update failed: {str(e)}")
            #raise RuntimeError(f"Backbone global update failed for backbone bar {self._bb_idx}: {str(e)}")
            return {
                "status": "failed", 
                "error": str(e), 
                "event_samples": len(current_bb_event_df),
                "phys_samples": len(current_bb_phys_df),
                "artifacts_created": 8
            }

    def _run_interface_global_update(self, 
                                    event_df: pd.DataFrame, 
                                    phys_df: pd.DataFrame, 
                                    bb_dir: Path, 
                                    backbone_results: Dict) -> Dict[str, Any]:
        """Run interface global update step."""
        
        try:
            # Get the current backbone for this backbone bar
            current_backbone = self.stacks[f"backbone_{self._bb_idx}"].interface.backbone
            
            # Get the interface for this backbone bar 
            current_interface = self.stacks[f"backbone_{self._bb_idx}"].interface
            
            # Filter data for current backbone bar
            current_bb = self._bb_idx
            current_event_df = event_df[event_df['backbone_bar'] == current_bb].copy()
            current_phys_df = phys_df[phys_df['backbone_bar'] == current_bb].copy()
            
            self.logger.info(f"Interface processing backbone bar {current_bb}: {len(current_event_df)} event samples, {len(current_phys_df)} physical samples")
            
            # Run interface global_update
            interface_result = current_interface.global_update(
                event_driven_backbone_time_bar_data=current_event_df,
                physical_time_driven_backbone_time_bar_data=current_phys_df,
                target_col_name=self.cfg.target_col_name,
                distribution_model=self.cfg.dist,
                n_lags=self.cfg.n_lags,
                validation_proportion=self.cfg.validation_proportion_interface,
                selection_threshold=self.cfg.selection_threshold,
                tolerance_threshold=self.cfg.tolerance_threshold,
                ols_significance_level=self.cfg.ols_significance_level,
                second_stage_mse=self.cfg.second_stage_mse,
                save_dir=str(bb_dir / "interface")
            )
                     
            return {
                "status": "completed",
                "interface_result": interface_result,
                "selected_variables": interface_result['selected_variables'],
                "n_selected": interface_result['n_selected'],
                "n_total_variables": interface_result['n_total_variables'],
                "artifacts_created": 4
            }
            
        except Exception as e:
            self.logger.error(f"Interface global update failed: {str(e)}")
            raise RuntimeError(f"Interface global update failed: {str(e)}")

    def _run_adapter_global_update(self, 
                                   event_df: pd.DataFrame, 
                                   phys_df: pd.DataFrame, 
                                   bb_dir: Path) -> Dict[str, Any]:
        """
        Run adapter global update step.
        
        This method performs ARMAX-GARCH model fitting across all adapter bars within
        a backbone bar, evaluates forecast performance, and generates interpretability
        visualizations.
        
        Args:
            event_df: Event-driven data for the current backbone bar
            phys_df: Physical time data for the current backbone bar
            bb_dir: Directory for saving artifacts
            
        Returns:
            Dict containing adapter global update results with keys:
                - 'status': 'completed' or 'failed'
                - 'total_adapter_bars': Number of adapter bars processed
                - 'successful_updates': Number of successful adapter updates
                - 'failed_updates': Number of failed updates
                - 'success_rate': Ratio of successful to total updates
                - 'parameters_by_bar': Dict mapping adapter_bar to parameter dict
                - 'processing_stats': Processing statistics for each bar
                - 'artifacts_created': Number of artifacts generated
                
        Artifacts Generated:
            - param_series.png: Time series plot of all adapter parameters
            - armax_garch_corridor.png: Significance corridor plot for ARMAX-GARCH coefficients
            - model_validity_corridor.png: GARCH diagnostic test results (if available)
            - param_series.parquet: Parameter series data
            - performance.parquet: Forecast performance metrics
            - shap_summary.png: SHAP-style feature contribution visualization
            
        Note:
            Relative contribution values are computed for visualization but not stored
            as parquet files. Only the SHAP-style summary plot is saved for interpretability.
        """
        
        
        try:
            self.logger.info(f"Running adapter bar updates for backbone bar {self._bb_idx}")
            # Create adapter bars using adapter delta
            phys_df = phys_df.copy()
            event_df = event_df.copy()

            # Create adapter bar assignments
            phys_df['adapter_bar'] = (np.floor((phys_df['timestamp'] - self.cfg.session_start + 1e-6) / self.cfg.adelta).astype(int))
            event_df['adapter_bar'] = (np.floor((event_df['timestamp'] - self.cfg.session_start + 1e-6) / self.cfg.adelta).astype(int))
            
            # Get unique adapter bars
            adapter_bars = sorted(phys_df['adapter_bar'].unique())
            self.logger.info(f"Found {len(adapter_bars)} adapter bars for processing")
            
            # Get the adapter from the current stack
            current_adapter = self.stacks[f"backbone_{self._bb_idx}"]
            if current_adapter is None:
                self.logger.error("No adapter available for global update")
                raise RuntimeError("No adapter available for global update")
        
            # Use pre-fitted scaler from global update
            scaler = self.scalers[f"backbone_{self._bb_idx}"]
            if scaler is None:
                self.logger.error("No scaler available - global update must be run first")
                raise RuntimeError("No scaler available - global update must be run first")

            # Initialize results tracking
            successful_updates = 0
            failed_updates = 0
            all_parameters = {}
            all_adapter_bar_specific_parameters = {}
            all_adapter_bar_specific_p_values = {}
            all_garch_diagnostics = {}  # Track GARCH model validity tests
            processing_stats = {}
            relative_contributions = []
            scaled_features = []
            performance_metrics = []
            valid_local_estimate = False
            first_valid_local_estimate_found = False
            
            # Process each adapter bar
            for i, adapter_bar in enumerate(adapter_bars):
                bar_logger_prefix = f"adapter_bar_{adapter_bar:03d}"
                
                try:
                    # Filter physical time data for this adapter bar
                    bar_ptdata = phys_df[phys_df['adapter_bar'] == adapter_bar].copy()
                    
                    # Filter event data for this adapter bar with lookback strategy
                    # CRITICAL FIX: Include last sampling bar from previous adapter bar to prevent NaN hidden activations
                    # 
                    # Problem: The first physical observation in each adapter bar had no prior event data
                    # for neural feature extraction, causing NaN values in hidden_activations columns.
                    # This created downstream issues in GARCH model fitting and data processing.
                    #
                    # Solution: For each adapter bar (except the first), include the last sampling time bar
                    # from the previous adapter bar. This provides the necessary temporal lookback window
                    # for neural covariate extraction while maintaining temporal integrity (only using past data).
                    #
                    # Benefits: 
                    # - Eliminates NaN hidden activations for all but the very first observation globally
                    # - Maintains temporal precedence requirements 
                    # - Minimal memory overhead (only one sampling bar worth of events per adapter bar)
                    # - Preserves independence of adapter bar processing
                    
                    if i > 0:  # Not the first adapter bar - apply lookback strategy
                        prev_adapter_bar = adapter_bars[i-1]
                        
                        # Get event data from previous adapter bar to find last sampling bar
                        prev_event_data = event_df[event_df['adapter_bar'] == prev_adapter_bar]
                        
                        if len(prev_event_data) > 0 and 'sampling_time_bar' in prev_event_data.columns:
                            # Find the last sampling time bar from previous adapter bar
                            last_sampling_bar = prev_event_data['sampling_time_bar'].max()
                            prev_lookback = prev_event_data[
                                prev_event_data['sampling_time_bar'] == last_sampling_bar
                            ].copy()
                            
                            # Get current adapter bar event data
                            current_event_data = event_df[event_df['adapter_bar'] == adapter_bar].copy()
                            
                            # Combine lookback data with current adapter bar data
                            bar_edlob = pd.concat([prev_lookback, current_event_data], 
                                                ignore_index=True).sort_values('timestamp')

                            self.logger.info(f"Added {len(prev_lookback)} lookback events from "
                                             f"sampling bar {last_sampling_bar} of previous adapter bar")
                        else:
                            # Fallback: use current adapter bar data only
                            bar_edlob = event_df[event_df['adapter_bar'] == adapter_bar].copy()
                            self.logger.info("No previous sampling data available, using current bar only")
                    else:
                        # First adapter bar - no previous data available (will still have NaN for first observation)
                        bar_edlob = event_df[event_df['adapter_bar'] == adapter_bar].copy()
                        self.logger.info("First adapter bar - no lookback data available")

                    self.logger.info(f"Processing {bar_logger_prefix}: "
                                    f"{len(bar_ptdata)} physical samples, "
                                    f"{len(bar_edlob)} event samples (the latter including lookback)")
                    
                    # Skip bars with insufficient data
                    min_samples = 60  # Minimum samples required for fitting
                    if len(bar_ptdata) < min_samples:
                        self.logger.warning(f"Skipping {bar_logger_prefix}: insufficient data "
                                          f"({len(bar_ptdata)} < {min_samples})")
                        processing_stats[adapter_bar] = {
                            'status': 'skipped',
                            'reason': 'insufficient_data',
                            'physical_samples': len(bar_ptdata),
                            'event_samples': len(bar_edlob)
                        }
                        continue
                    
                    # Call adapter.update_adapter method
                    self.logger.info(f"Running update_adapter for {bar_logger_prefix}")
                    # Print the number of rows with NaNs in bar_edlob
                    
                    adapter_result = current_adapter.update_adapter(
                        event_driven_adapter_time_bar_data=bar_edlob,
                        physical_time_driven_adapter_time_bar_data=bar_ptdata,
                        scaler=scaler,
                        adapter_bar_number=successful_updates + 1,  # 1-indexed
                        target_col_name=self.cfg.target_col_name,
                        update_parameters=self.cfg.update_parameters
                    )

                    valid_local_estimate = adapter_result['valid_local_estimate']
                    
                    # Fall back if saddle point detected
                    if not valid_local_estimate:
                        self.logger.warning(f"No valid estimate found for {bar_logger_prefix}")
                        if not first_valid_local_estimate_found:
                            self.logger.error(f"No previous adapter state to revert to for {bar_logger_prefix}")
                            raise RuntimeError(f"Cannot revert adapter state for first adapter bar {bar_logger_prefix}")
                        # Fallback to previous adapter state recording the failure
                        failed_updates += 1
                    else:
                        first_valid_local_estimate_found = True


                    if i < len(adapter_bars) - 1:
                        next_adapter_bar = adapter_bars[i+1]
                    
                        out_of_sample_phys_df = phys_df[phys_df['adapter_bar'] == next_adapter_bar].copy()
                        out_of_sample_event_df = event_df[event_df['adapter_bar'] == next_adapter_bar].copy()

                        if out_of_sample_phys_df.empty or out_of_sample_event_df.empty:
                            self.logger.warning(f"Skipping forecast evaluation for {bar_logger_prefix}: next bar {next_adapter_bar} is empty.")
                            continue

                        self.logger.info(f"Calculating forecast performance using the carried estimates from {bar_logger_prefix} on adapter time bar {next_adapter_bar}")
                        metrics_all = current_adapter.calculate_performance_metrics(
                            in_sample_event_df=bar_edlob,
                            in_sample_phys_df=bar_ptdata,
                            out_of_sample_event_df=out_of_sample_event_df,
                            out_of_sample_phys_df=out_of_sample_phys_df,
                            benchmark_forecast=self.target_scales[f"backbone_{self._bb_idx}"]['mean'],
                            scaler=scaler,
                            target_col_name=self.cfg.target_col_name
                        )
                        metrics = metrics_all['metrics']
                        
                        # Collect relative contributions
                        contrib_df = metrics_all['relative_contributions_df'].copy()
                        contrib_df['adapter_bar'] = next_adapter_bar
                        relative_contributions.append(contrib_df)
                        
                        # Collect scaled features for SHAP-style plotting
                        features_df = metrics_all['scaled_features_df'].copy()
                        features_df['adapter_bar'] = next_adapter_bar
                        scaled_features.append(features_df)

                        metrics['adapter_bar'] = next_adapter_bar  # Log metrics against the bar they were tested on
                        performance_metrics.append(metrics)

                        
                    
                    # Store results
                    if valid_local_estimate:
                        all_parameters[adapter_bar] = copy.deepcopy(current_adapter.parameters)
                        all_adapter_bar_specific_parameters[adapter_bar] = dict(zip(copy.deepcopy(adapter_result['coefficient_names']), 
                                                                                    copy.deepcopy(adapter_result['coefficients'])))
                        all_adapter_bar_specific_p_values[adapter_bar] = dict(zip(copy.deepcopy(adapter_result['coefficient_names']), 
                                                                                    copy.deepcopy(adapter_result['p_values'])))
                    
                        # Store GARCH diagnostic test results
                        garch_diagnostics = {}
                        if 'gof_pval' in adapter_result:
                            gof_pvals = adapter_result['gof_pval']
                            if isinstance(gof_pvals, (list, np.ndarray)) and len(gof_pvals) >= 4:
                                garch_diagnostics['gof20'] = gof_pvals[0]
                                garch_diagnostics['gof30'] = gof_pvals[1] 
                                garch_diagnostics['gof40'] = gof_pvals[2]
                                garch_diagnostics['gof50'] = gof_pvals[3]
                        
                        if 'lb_pval' in adapter_result:
                            lb_pvals = adapter_result['lb_pval']
                            if isinstance(lb_pvals, (list, np.ndarray)) and len(lb_pvals) >= 3:
                                garch_diagnostics['lb1'] = lb_pvals[0]
                                garch_diagnostics['lb5'] = lb_pvals[1]
                                garch_diagnostics['lb9'] = lb_pvals[2]
                        
                        if 'lb_sq_pval' in adapter_result:
                            lb_sq_pvals = adapter_result['lb_sq_pval']
                            if isinstance(lb_sq_pvals, (list, np.ndarray)) and len(lb_sq_pvals) >= 3:
                                garch_diagnostics['lbsq1'] = lb_sq_pvals[0]
                                garch_diagnostics['lbsq5'] = lb_sq_pvals[1]
                                garch_diagnostics['lbsq9'] = lb_sq_pvals[2]
                    
                        all_garch_diagnostics[adapter_bar] = garch_diagnostics


                        processing_stats[adapter_bar] = {
                            'status': 'completed',
                            'physical_samples': len(bar_ptdata),
                            'event_samples': len(bar_edlob)
                        }

                        successful_updates += 1
                    
                    else:
                        failed_updates += 1
                        processing_stats[adapter_bar] = {
                            'status': 'saddle_point_detected',
                            'physical_samples': len(bar_ptdata),
                            'event_samples': len(bar_edlob)
                        }
                    
                    
                    
                except Exception as e:
                    failed_updates += 1
                    error_msg = f"Failed to process {bar_logger_prefix}: {str(e)}"
                    self.logger.warning(error_msg)
                    processing_stats[adapter_bar] = {
                        'status': 'failed',
                        'error': str(e),
                        'physical_samples': len(bar_ptdata) if 'bar_ptdata' in locals() else 0,
                        'event_samples': len(bar_edlob) if 'bar_edlob' in locals() else 0
                    }
                    continue


            # Create artifacts directory
            artifacts_dir = bb_dir / "adapter"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            if not first_valid_local_estimate_found:
                error_msg = "No successful adapter updates - cannot generate artifacts"
                self.logger.error(error_msg)
                
                # Save error message to file
                error_file = artifacts_dir / "error.txt"
                with open(error_file, 'w') as f:
                    f.write(error_msg)
                self.logger.info(f"Saved error message to {error_file}")

                raise RuntimeError("Adapter global update failed completely: no successful adapter updates")
                

            #self.logger.info(f"Adapter parameters by bar: {all_parameters}")
            all_parameters_df = pd.DataFrame.from_dict(all_parameters, orient='index')
            all_adapter_bar_specific_parameters_df = pd.DataFrame.from_dict(all_adapter_bar_specific_parameters, orient='index')
            all_adapter_bar_specific_p_values_df = pd.DataFrame.from_dict(all_adapter_bar_specific_p_values, orient='index')
            all_parameters_df.index.name = 'adapter_bar'
            all_adapter_bar_specific_parameters_df.index.name = 'adapter_bar'
            all_adapter_bar_specific_p_values_df.index.name = 'adapter_bar'

            selected_vars = current_adapter.interface.selected_variables
            mxreg_mapping = {}
            mxreg_count = 1
            for name in all_parameters_df.columns:
                if name.startswith('mxreg'):
                    if mxreg_count <= len(selected_vars):
                        mxreg_mapping[name] = selected_vars[mxreg_count - 1]
                        mxreg_count += 1
                    else:
                        self.logger.warning(f"More mxreg parameters ({mxreg_count}) than selected variables ({len(selected_vars)})")
                        mxreg_mapping[name] = name  # fallback if mapping fails
                else:
                    mxreg_mapping[name] = name
            
            # Rename columns using the mapping
            all_parameters_df.rename(columns=mxreg_mapping, inplace=True)
            all_adapter_bar_specific_parameters_df.rename(columns=mxreg_mapping, inplace=True)
            all_adapter_bar_specific_p_values_df.rename(columns=mxreg_mapping, inplace=True)



            # Save processing statistics
            # stats_df = pd.DataFrame.from_dict(processing_stats, orient='index')
            # stats_file = artifacts_dir / "processing_stats.parquet"
            # stats_df.to_parquet(stats_file)
            
            # Create convergence performance summary
            # Convert processing_stats keys to strings for JSON serialization
            processing_stats_serializable = {str(k): v for k, v in processing_stats.items()}
            
            convergence_performance_data = {
                'total_adapter_bars': len(adapter_bars),
                'successful_updates': successful_updates,
                'failed_updates': failed_updates,
                'success_rate': successful_updates / len(adapter_bars) if adapter_bars else 0.0,
                'processing_stats': processing_stats_serializable  # Include detailed stats for each bar
            }
            
            # Save convergence performance data as JSON
            convergence_file = artifacts_dir / "convergence_performance.json"
            with open(convergence_file, 'w') as f:
                json.dump(convergence_performance_data, f, indent=2)
            self.logger.info(f"Saved convergence performance data to {convergence_file}")
            
            # Calculate actual artifacts created count
            artifacts_created = 4  # param_series.png, armax_garch_corridor.png, param_series.parquet, all_parameters.parquet
            if all_garch_diagnostics and not pd.DataFrame.from_dict(all_garch_diagnostics, orient='index').empty:
                artifacts_created += 1  # model_validity_corridor.png
            
            # Compile final results
            adapter_global_results = {
                'status': 'completed',
                'total_adapter_bars': len(adapter_bars),
                'successful_updates': successful_updates,
                'failed_updates': failed_updates,
                'success_rate': convergence_performance_data['success_rate'],
                'parameters_by_bar': all_parameters,
                'processing_stats': processing_stats,
                'artifacts_created': artifacts_created
            }
            
            self.logger.info(f"Adapter global update completed: "
                           f"{successful_updates}/{len(adapter_bars)} bars successful "
                           f"({convergence_performance_data['success_rate']:.1%} success rate)")
            
            # Generate and save the parameter series plot
            plot_save_path = artifacts_dir / 'param_series.png'
            self._plot_parameter_series(
                all_parameters_df=all_parameters_df,
                all_adapter_bar_specific_parameters_df=all_adapter_bar_specific_parameters_df,
                save_path=plot_save_path
            )

            # Drop non-coefficient parameters for the corridor plot
            cols_to_drop = ['skew', 'shape', 'ghlambda']
            
            # Create temporary dataframes for plotting, dropping columns without modifying originals
            plot_params_df = all_adapter_bar_specific_parameters_df.drop(
                columns=cols_to_drop, errors='ignore'
            )
            plot_pvalues_df = all_adapter_bar_specific_p_values_df.drop(
                columns=cols_to_drop, errors='ignore'
            )

            plot_save_path = artifacts_dir / 'armax_garch_corridor.png'
            self._plot_parameter_significance_corridor(
                parameters_df=plot_params_df,
                p_values_df=plot_pvalues_df,
                title='ARMAX-GARCH Corridor',
                save_path=plot_save_path
            )

            # Generate model validity corridor plot for GARCH diagnostics
            if all_garch_diagnostics:
                garch_diagnostics_df = pd.DataFrame.from_dict(all_garch_diagnostics, orient='index')
                garch_diagnostics_df.index.name = 'adapter_bar'
                
                if not garch_diagnostics_df.empty:
                    plot_save_path = artifacts_dir / 'model_validity_corridor.png'
                    self._plot_model_validity_corridor(
                        diagnostics_df=garch_diagnostics_df,
                        save_path=plot_save_path
                    )

            # Save parameter series to parquet
            params_path = artifacts_dir / "param_series.parquet"
            all_parameters_df = all_parameters_df.merge(
                all_adapter_bar_specific_parameters_df,
                on=["adapter_bar"],
                suffixes=("", "_specific"),
                how="inner"
            )
            all_parameters_df.to_parquet(params_path)
            self.logger.info(f"Saved adapter parameter series to {params_path}")

            # Save performance metrics
            performance_df = pd.DataFrame(performance_metrics)
            if not performance_df.empty:
                performance_df.set_index('adapter_bar', inplace=True)
                performance_file = artifacts_dir / "performance.parquet"
                performance_df.to_parquet(performance_file)
                self.logger.info(f"Saved performance metrics to {performance_file}")
            
            # Generate SHAP-style summary plot
            if relative_contributions and scaled_features:
                contributions_df = pd.concat(relative_contributions, ignore_index=False)
                features_df = pd.concat(scaled_features, ignore_index=False)
                shap_plot_path = artifacts_dir / "shap_summary.png"
                self._plot_shap_style_summary(
                    contributions_df=contributions_df,
                    scaled_features_df=features_df,
                    save_path=shap_plot_path
                )
            
            return adapter_global_results
            
        except Exception as e:
            self.logger.error(f"Adapter global update failed: {str(e)}")
            # Create minimal artifacts for failed run
            artifacts_dir = bb_dir / "adapter"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "error.txt").write_text(str(e))
            
            return {
                'status': 'failed',
                'error': str(e),
                'artifacts_created': 1
            }

    def local_adapt(self, 
                    event_df: pd.DataFrame, 
                    phys_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Perform local adaptation for all backbone bars in the session automatically.
        
        Args:
            event_df: Full event-driven trading session data
            phys_df: Full physical-time trading session data
            
        Returns:
            List of results for each backbone bar processed
        """
        if self._bb_idx < 0:
            self.logger.error("Cannot run local_adapt before global_update")
            raise RuntimeError("Cannot run local_adapt before global_update")
        
        event_df = event_df[(event_df['timestamp'] >= self.cfg.session_start) & 
                           (event_df['timestamp'] <= self.cfg.session_finish)].copy()
        phys_df = phys_df[(phys_df['timestamp'] >= self.cfg.session_start) & 
                         (phys_df['timestamp'] <= self.cfg.session_finish)].copy()

        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("Starting local adaptation for full trading session")
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        
        # Add time bar columns if not present
        event_df_processed = self._add_time_bar_columns(event_df.copy())
        phys_df_processed = self._add_time_bar_columns(phys_df.copy())
        
        # Get all backbone bars to process (should match those from global update)
        backbone_bars = sorted(event_df_processed['backbone_bar'].unique())
        total_backbone_bars = len(backbone_bars)
        
        # Validate that we have global update results for all backbone bars
        expected_bb_count = self._bb_idx + 1  # _bb_idx is 0-based
        if total_backbone_bars != expected_bb_count:
            raise RuntimeError(f"Mismatch: found {total_backbone_bars} backbone bars in data, but have global update results for {expected_bb_count}")
        
        self.logger.info(f"Processing local adaptation for {total_backbone_bars} backbone bars: {backbone_bars}")
        
        all_results = []
        
        # Reset backbone index for local adaptation processing
        original_bb_idx = self._bb_idx
        self._bb_idx = -1  # Will be incremented in _local_adapt_single_bar
        
        for i, bb in enumerate(backbone_bars):
            self.logger.info(f"Processing local adaptation for backbone bar {bb} ({i+1}/{total_backbone_bars})")
            
            # Filter data for this specific backbone bar
            event_bb_data = event_df_processed[event_df_processed['backbone_bar'] == bb].copy()
            phys_bb_data = phys_df_processed[phys_df_processed['backbone_bar'] == bb].copy()
            
            self.logger.info(f"Backbone bar {bb} data: {len(event_bb_data):,} observed events, {len(phys_bb_data):,} sampling time bars")
            
            # Run local adaptation for this backbone bar
            try:
                results = self._local_adapt_single_bar(event_bb_data, phys_bb_data)
                all_results.append(results)
                self.logger.info(f"Local adaptation for backbone bar {bb} completed successfully")
            except Exception as e:
                self.logger.error(f"Local adaptation for backbone bar {bb} failed: {str(e)}")
                raise RuntimeError(f"Local adaptation failed for backbone bar {bb}: {str(e)}")
        
        # Restore original backbone index
        self._bb_idx = original_bb_idx
        
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info(f"Local adaptation session completed successfully for {total_backbone_bars} backbone bars")
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        return all_results

    def _local_adapt_single_bar(self, event_df: pd.DataFrame, phys_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform local adaptation for a single backbone bar.
        
        Note: This is the original local_adapt behavior. Most users should use
        local_adapt() which handles all backbone bars automatically.
        
        Args:
            event_df: Event-driven data for one backbone bar
            phys_df: Physical-time data for one backbone bar
            
        Returns:
            Dict containing OOS metrics, adapter bar specific coeffs, and break decisions
        """
        self._bb_idx += 1
        bb_dir = self._create_backbone_directories("local_adapt")
        
        self.logger.info(f"Starting local_adapt for backbone bar {self._bb_idx:03d}")
        
        try:
            # Add time bar columns if not present
            event_df_processed = self._add_time_bar_columns(event_df.copy())
            phys_df_processed = self._add_time_bar_columns(phys_df.copy())
            
            # Step 1: Backbone out-of-sample testing
            backbone_results = self._run_backbone_local_adapt(
                event_df_processed, phys_df_processed, bb_dir
            )
            
            # Step 2: Interface local selection
            interface_results = self._run_interface_local_adapt(
                event_df_processed, phys_df_processed, bb_dir
            )
            
            # Step 3: Adapter local updates
            adapter_results = self._run_adapter_local_adapt(
                event_df_processed, phys_df_processed, bb_dir
            )
            
            # Step 4: Update manifest and return results
            results = {
                "backbone_bar": self._bb_idx,
                "backbone_results": backbone_results,
                "interface_results": interface_results,
                "adapter_results": adapter_results,
                "artifact_paths": self._get_artifact_paths(bb_dir)
            }
            
            self._update_manifest("local_adapt", results)
            
            self.logger.info(f"Local adaptation completed for backbone bar {self._bb_idx:03d}")
            return results
            
        except Exception as e:
            self.logger.warning(f"Local adaptation failed for backbone bar {self._bb_idx:03d}: {str(e)}")
            # Continue processing - local adaptation failures are non-fatal
            return {"error": str(e), "backbone_bar": self._bb_idx}

    def _run_backbone_local_adapt(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path) -> Dict[str, Any]:
        """
        Run backbone local adaptation step - backbone remains FROZEN (no updates).
        
        In local adaptation, the backbone model is not retrained or updated.
        We use the frozen model for out-of-sample evaluation and save test results.
        """
        self.logger.info("="*80)
        self.logger.info("="*80)
        self.logger.info("Running backbone local adaptation (backbone frozen - evaluating out-of-sample)")
        self.logger.info("="*80)
        self.logger.info("="*80)
        
        try:
            # Get the current backbone for this backbone bar
            current_backbone = self.stacks[f"backbone_{self._bb_idx}"].interface.backbone
            
            # Filter data for the current backbone bar
            current_bb_event_df = event_df[event_df['backbone_bar'] == self._bb_idx].copy()
            current_bb_phys_df = phys_df[phys_df['backbone_bar'] == self._bb_idx].copy()
            
            self.logger.info(f"Backbone bar {self._bb_idx}: {len(current_bb_event_df)} event samples for OOS evaluation")
            
            # Log the full covariate list for verification
            if hasattr(current_backbone, 'full_covariate_list') and current_backbone.full_covariate_list:
                self.logger.info("="*80)
                self.logger.info(f"Backbone bar {self._bb_idx} - Full covariate list ({len(current_backbone.full_covariate_list)} variables):")
                self.logger.info("="*80)
                for i, covariate in enumerate(current_backbone.full_covariate_list):
                    self.logger.info(f"  {i+1}. {covariate}")
                self.logger.info("="*80)
            else:
                self.logger.warning(f"Backbone bar {self._bb_idx} - No full_covariate_list found or empty")
            
            # Check if we have sufficient data
            if len(current_bb_event_df) < 10:
                self.logger.warning(f"Insufficient event data for backbone bar {self._bb_idx}: {len(current_bb_event_df)} samples")
                
                # Create minimal test results for insufficient data
                artifacts_dir = bb_dir / "backbone"
                test_results = {
                    'model_class': current_backbone.model.__class__.__name__,
                    'parsimonious_mode': current_backbone.parsimonious_mode,
                    'test_loss': None,
                    'status': 'insufficient_data',
                    'event_samples': len(current_bb_event_df)
                }
                
                test_path = artifacts_dir / "test_os.json"
                with open(test_path, 'w') as f:
                    json.dump(test_results, f, indent=2)
                
                return {
                    "status": "insufficient_data", 
                    "event_samples": len(current_bb_event_df),
                    "test_loss": None,
                    "artifacts_created": 1
                }
            
            # Use MSE loss for evaluation (consistent with backbone training)
            criterion = nn.MSELoss(reduction='sum')
            
            # Evaluate the frozen backbone model on out-of-sample event data
            test_loss = current_backbone.evaluate_out_of_sample(
                event_driven_data=current_bb_event_df,
                criterion=criterion,
                batch_size=self.cfg.batch_size,
                save_dir=str(bb_dir / "backbone")
            )
            
            self.logger.info(f"Backbone bar {self._bb_idx} - Out-of-sample test loss: {test_loss:.6f}")

            self.logger.info("="*80)
            self.logger.info("="*80)
            self.logger.info("Finished running backbone local adaptation")
            self.logger.info("="*80)
            self.logger.info("="*80)
            
            return {
                "status": "completed",
                "test_loss": test_loss,
                "event_samples": len(current_bb_event_df),
                "model_class": current_backbone.model.__class__.__name__,
                "parsimonious_mode": current_backbone.parsimonious_mode,
                "artifacts_created": 1
            }
            
        except Exception as e:
            self.logger.error(f"Backbone local adaptation failed: {str(e)}")
            
            # Create error artifact
            artifacts_dir = bb_dir / "backbone"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "error.txt").write_text(str(e))
            
            return {
                "status": "failed",
                "error": str(e),
                "event_samples": len(current_bb_event_df) if 'current_bb_event_df' in locals() else 0,
                "test_loss": None,
                "artifacts_created": 1
            }

    def _run_interface_local_adapt(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path) -> Dict[str, Any]:
        """Run interface local adaptation step."""
        self.logger.info("Running interface local adaptation...")
        
        try:
            # Get the current interface for this backbone bar
            current_interface = self.stacks[f"backbone_{self._bb_idx}"].interface
            
            # Filter data for current backbone bar
            current_bb = self._bb_idx
            current_event_df = event_df[event_df['backbone_bar'] == current_bb].copy()
            current_phys_df = phys_df[phys_df['backbone_bar'] == current_bb].copy()
            
            self.logger.info(f"Interface local adaptation for backbone bar {current_bb}: {len(current_event_df)} event samples, {len(current_phys_df)} physical samples")
            
            # Run interface local_adapt with the same configuration parameters as global_update
            interface_result = current_interface.local_adapt(
                event_driven_backbone_time_bar_data=current_event_df,
                physical_time_driven_backbone_time_bar_data=current_phys_df,
                target_col_name=self.cfg.target_col_name,
                distribution_model=self.cfg.dist,
                n_lags=self.cfg.n_lags,
                tolerance_threshold=self.cfg.tolerance_threshold,
                ols_significance_level=self.cfg.ols_significance_level,
                wald_significance_level=self.cfg.wald_significance_level,  # Use same level for Wald tests
                save_dir=str(bb_dir / "interface")
            )
            
            return {
                "status": "completed",
                "interface_result": interface_result,
                "structural_breaks": interface_result['structural_breaks'],
                "artifacts_created": 3  # corridor_ada_lasso.png, enet_coef_corridor.png, structural_breaks.parquet
            }
            
        except Exception as e:
            self.logger.error(f"Interface local adaptation failed: {str(e)}")
            
            # Create minimal artifacts for failed run
            artifacts_dir = bb_dir / "interface"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "error.txt").write_text(str(e))
            
            return {
                "status": "failed",
                "error": str(e),
                "artifacts_created": 1
            }

    def _run_adapter_local_adapt(self, 
                                 event_df: pd.DataFrame, 
                                 phys_df: pd.DataFrame, bb_dir: Path) -> Dict[str, Any]:
        """
        Run adapter local adaptation step.

        This method has exactly the same logic as _run_adapter_global_update, but uses
        pre-fitted scalers and target variable scales from the global update phase rather
        than recomputing them. This ensures consistent preprocessing between training
        (global update) and testing (local adaptation).

        Args:
            event_df: Event-driven data for the current backbone bar
            phys_df: Physical-time data for the current backbone bar 
            bb_dir: Directory for saving artifacts
            
        Returns:
            Dict containing adapter local adaptation results (same structure as
            _run_adapter_global_update)
            
        Note:
            The covariate scalers (self.scalers) and target variable statistics
            (self.target_scales) are retrieved from global update, not recomputed.
            This prevents lookahead bias in local adaptation.
        """
        return self._run_adapter_global_update(event_df, phys_df, bb_dir)

    def _plot_shap_style_summary(self,
                                  contributions_df: pd.DataFrame,
                                  scaled_features_df: pd.DataFrame,
                                  save_path: Path) -> None:
        """
        Create and save a SHAP-style summary plot showing relative contributions.
        
        This plot displays feature contributions (coefficient × feature_value) with:
        - One horizontal strip per feature (y-axis)
        - Contribution values on x-axis
        - Point colors representing feature values (blue=low, red=high)
        - Vertical jitter for visibility when many points overlap
        
        Args:
            contributions_df: DataFrame with columns ['timestamp', 'adapter_bar', var1, var2, ...]
                containing relative contributions (coefficient × feature_value)
            scaled_features_df: DataFrame with columns ['timestamp', 'adapter_bar', var1, var2, ...]
                containing scaled feature values for color mapping
            save_path: Path to save the plot
        """
        self.logger.info(f"Generating SHAP-style summary plot and saving to {save_path}")
        
        # Ensure the save directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get feature columns (exclude timestamp and adapter_bar)
        feature_cols = [col for col in contributions_df.columns 
                       if col not in ['timestamp', 'adapter_bar']]
        
        if len(feature_cols) == 0:
            self.logger.warning("No feature columns found in contributions_df. Skipping SHAP plot.")
            return
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create figure with appropriate height based on number of features
        fig_height = max(6, len(feature_cols) * 0.6)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        
        # Prepare data for plotting
        y_positions = np.arange(len(feature_cols))
        
        # Use a colormap similar to the SHAP example (blue to red)
        cmap = plt.cm.coolwarm
        
        # Process each feature
        for i, feature in enumerate(feature_cols):
            # Get contribution values (SHAP values)
            shap_values = contributions_df[feature].values
            
            # Get corresponding feature values for coloring
            feature_values = scaled_features_df[feature].values
            
            # Add vertical jitter to avoid overplotting
            np.random.seed(42)  # For reproducibility
            y_jitter = np.random.normal(0, 0.1, size=len(shap_values))
            y_coords = np.full(len(shap_values), y_positions[i]) + y_jitter
            
            # Create scatter plot with colors based on feature values
            scatter = ax.scatter(
                shap_values,
                y_coords,
                c=feature_values,
                cmap=cmap,
                alpha=0.2,
                s=20,
                edgecolors='none',
                rasterized=True  # For better performance with many points
            )
        
        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(feature_cols)
        ax.set_xlabel('Relative Contribution (Coefficient × Feature Value)', fontsize=12)
        ax.set_title('Feature Contribution Summary (SHAP-style)', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
            vmin=scaled_features_df[feature_cols].values.min(),
            vmax=scaled_features_df[feature_cols].values.max()
        ))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Standardized Feature Value', fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the figure
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"SHAP-style summary plot saved to {save_path}")
        plt.close(fig)

    def _plot_parameter_series(self, 
                               all_parameters_df: pd.DataFrame, 
                               all_adapter_bar_specific_parameters_df: pd.DataFrame, 
                               save_path: Path):
        """
        Create and save a time series plot of adapter parameters, with one panel per parameter.

        Args:
            all_parameters_df: DataFrame of combined parameters (solid lines).
            all_adapter_bar_specific_parameters_df: DataFrame of bar-specific parameters (dashed lines).
            save_path: The file path to save the plot to.
        """
        self.logger.info(f"Generating multi-panel parameter series plot and saving to {save_path}")
        
        # Ensure the save directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Determine grid size
        num_params = len(all_parameters_df.columns)
        n_cols = 3
        n_rows = (num_params + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
        axes = axes.flatten()  # Flatten to 1D array for easy iteration
        
        for i, col in enumerate(all_parameters_df.columns):
            ax = axes[i]
            
            # Plot combined parameters (solid line)
            ax.plot(all_parameters_df.index, all_parameters_df[col], 
                    linestyle='-', color='b', label='Combined')
            
            # Plot bar-specific parameters (dashed line)
            if col in all_adapter_bar_specific_parameters_df.columns:
                ax.plot(all_adapter_bar_specific_parameters_df.index, all_adapter_bar_specific_parameters_df[col], 
                        linestyle='--', color='r', label='Bar-specific')

            ax.set_title(col, fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.legend()

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        fig.suptitle('Adapter Estimates Evolution', fontsize=18, y=1.02)
        fig.text(0.5, 0.01, 'Adapter Bar', ha='center', va='center', fontsize=14)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Successfully saved parameter series plot to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save parameter series plot: {e}")
        finally:
            plt.close(fig)

    def _plot_parameter_significance_corridor(self,
                                            parameters_df: pd.DataFrame,
                                            p_values_df: pd.DataFrame,
                                            title: str,
                                            save_path: Path,
                                            significance_level: float = 0.05) -> None:
        """
        Generate and save a corridor plot for adapter parameters, showing significance.

        Args:
            parameters_df (pd.DataFrame): DataFrame of parameters (adapter bars x variables).
            p_values_df (pd.DataFrame): DataFrame of p-values for the parameters.
            title (str): The title for the plot.
            save_path (Path): The full path (including filename) to save the plot.
            significance_level (float): The p-value threshold for significance.
        """

        if parameters_df.empty:
            self.logger.warning("Parameters DataFrame is empty. Skipping plot generation.")
            return

        # Ensure the save directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine significance
        is_significant_df = p_values_df < significance_level

        # Determine the color map limit
        max_abs_value = np.nanmax(np.abs(parameters_df.values))

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(parameters_df.T, cmap='coolwarm', annot=False,
                    vmin=-max_abs_value, vmax=max_abs_value, ax=ax)

        # Annotate the heatmap with asterisks for significant coefficients
        for i, var_name in enumerate(parameters_df.columns):
            for j, bar_name in enumerate(parameters_df.index):
                if is_significant_df.loc[bar_name, var_name]:
                    ax.text(j + 0.5, i + 0.5, '*',
                            ha='center', va='center', color='white', fontsize=12)

        # Set ticks and labels
        ax.set_yticks(np.arange(len(parameters_df.columns)) + 0.5)
        ax.set_yticklabels(parameters_df.columns, rotation=0, fontsize=8)

        ax.set_xticks(np.arange(len(parameters_df.index)) + 0.5)
        ax.set_xticklabels(parameters_df.index, rotation=90, fontsize=8)

        # Annotate the counts of significant bars on the right margin
        significant_counts = is_significant_df.sum(axis=0)
        total_bars = len(is_significant_df.index)
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for idx, coef_name in enumerate(parameters_df.columns):
            text_str = f"(*) {significant_counts.get(coef_name, 0)}/{total_bars}"
            ax.text(1.005, idx + 0.5, text_str,
                    ha='left', va='center', fontsize=6, color='black',
                    transform=trans)

        plt.title(title)
        plt.xlabel('Adapter Bar')
        plt.ylabel('Covariate')
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for annotations
        
        try:
            plt.savefig(save_path, dpi=300)
            self.logger.info(f"Successfully saved adapter bar-specific corridor plot to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save plot to {save_path}: {e}")
        finally:
            plt.close(fig)

    def _plot_model_validity_corridor(self, 
                                     diagnostics_df: pd.DataFrame,
                                     save_path: Path) -> None:
        """
        Create a corridor plot for GARCH model validity diagnostics.
        
        Args:
            diagnostics_df: DataFrame with adapter_bar as index and diagnostic test p-values as columns
            save_path: Path to save the plot
        """
        if diagnostics_df.empty:
            self.logger.warning("No GARCH diagnostics data available for corridor plot")
            return
            
        # Ensure the save directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
            
        # Define test labels in order
        test_labels = ['gof20', 'gof30', 'gof40', 'gof50', 'lb1', 'lb5', 'lb9', 'lbsq1', 'lbsq5', 'lbsq9']
        
        # Filter to only tests that exist in the data
        available_tests = [test for test in test_labels if test in diagnostics_df.columns]
        
        if not available_tests:
            self.logger.warning("No recognized diagnostic tests found in data")
            return
            
        # Subset and reorder data
        plot_data = diagnostics_df[available_tests]
        
        # Create significance annotations (reject H0 if p < 0.05)
        significance_data = plot_data < 0.05
        
        # Count rejections for each test across all adapter bars
        rejection_counts = significance_data.sum(axis=0)
        
        # Create heatmap with same style as ARMAX-GARCH corridor plot
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(plot_data.T, cmap='RdYlGn', annot=False,
                    vmin=0, vmax=1, ax=ax)
        
        # Add asterisk annotations for significant results (p < 0.05)
        for i, test_name in enumerate(available_tests):
            for j, bar_name in enumerate(plot_data.index):
                if significance_data.loc[bar_name, test_name]:
                    ax.text(j + 0.5, i + 0.5, '*',
                            ha='center', va='center', color='white', fontsize=12)
        
        # Set ticks and labels (matching ARMAX-GARCH style)
        ax.set_yticks(np.arange(len(available_tests)) + 0.5)
        ax.set_yticklabels(available_tests, rotation=0, fontsize=8)

        ax.set_xticks(np.arange(len(plot_data.index)) + 0.5)
        ax.set_xticklabels(plot_data.index, rotation=90, fontsize=8)
        
        # Add rejection count annotations on the right margin (matching ARMAX-GARCH style)
        total_bars = len(plot_data.index)
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for idx, test_name in enumerate(available_tests):
            text_str = f"(*) {rejection_counts.get(test_name, 0)}/{total_bars}"
            ax.text(1.005, idx + 0.5, text_str,
                    ha='left', va='center', fontsize=6, color='black',
                    transform=trans)
        
        plt.title('Adapter Model Validity Diagnostics')
        plt.xlabel('Adapter Bar')
        plt.ylabel('Diagnostic Test')
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for annotations
        
        try:
            plt.savefig(save_path, dpi=300)
            self.logger.info(f"Successfully saved model validity corridor plot to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model validity plot to {save_path}: {e}")
        finally:
            plt.close(fig)


    def _get_backbone_bar_bounds(self, bb_idx: int) -> tuple[float, float]:
        """
        Calculate the start and end timestamp bounds for a given backbone bar index.
        
        Args:
            bb_idx: Backbone bar index (0-based)
            
        Returns:
            Tuple of (start_timestamp, end_timestamp)
        """
        bb_start = self.cfg.session_start + (bb_idx * self.cfg.bdelta)
        bb_end = bb_start + self.cfg.bdelta
        
        # Ensure we don't exceed session bounds
        bb_end = min(bb_end, self.cfg.session_finish)
        
        return {"status": "completed", "artifacts_created": 3}

    def _get_artifact_paths(self, bb_dir: Path) -> Dict[str, List[str]]:
        """Get all artifact paths for the current backbone bar."""
        artifacts = {}
        
        for subdir in ["backbone", "interface", "adapter"]:
            subdir_path = bb_dir / subdir
            if subdir_path.exists():
                files = []
                for file_path in subdir_path.rglob("*"):
                    if file_path.is_file():
                        files.append(str(file_path.relative_to(self.run_dir)))
                artifacts[subdir] = files
        
        return artifacts

    def _update_manifest(self, mode: str, results: Dict[str, Any]) -> None:
        """Update the manifest.json with new artifacts and results."""
        manifest_path = self.run_dir / "manifest.json"
        
        # Read current manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Add new entry
        entry = {
            "backbone_bar": self._bb_idx,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "artifact_paths": results.get("artifact_paths", {}),
            "summary": {
                "backbone_status": results.get("backbone_results", {}).get("status"),
                "interface_status": results.get("interface_results", {}).get("status"),
                "adapter_status": results.get("adapter_results", {}).get("status")
            }
        }
        
        manifest["artifacts"].append(entry)
        
        # Write updated manifest
        self._atomic_write_json(manifest_path, manifest)

    def _get_code_provenance(self) -> Dict[str, Any]:
        """Get code provenance information."""
        # TODO: Implement git provenance
        return {
            "git_commit": "placeholder",
            "git_branch": "main", 
            "git_dirty": False
        }

    def _get_environment_info(self) -> Dict[str, str]:
        """Get environment version information."""
        return {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch": torch.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            # TODO: Add rpy2, R, glmnet versions
        }

    def _atomic_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Atomically write JSON data to file."""
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        os.replace(tmp_path, path)
