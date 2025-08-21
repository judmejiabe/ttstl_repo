"""
TTSTL Orchestrator - Coordinates Backbone, Interface, and Adapter modules.

This module provides the main orchestration logic for the Two Time-Scale Transfer Learning
framework, managing the execution of backbone training, interface variable selection,
and adapter parameter updates across trading sessions.
"""

# Standard library imports
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
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
from ttstl.backbone import Backbone
from ttstl.interface import Interface  
from ttstl.adapter import Adapter
from ttstl.architectures.cnn_lstm import CNNLSTM


@dataclass
class OrchestratorConfig:
    """Configuration for the TTSTL Orchestrator."""
    
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
    epochs: int = 12 
    batch_size: int = 32
    validation_proportion_backbone: float = 0.1
    buffer_proportion: float = 0.05
    quantile: float = 0.005 # Windsorization quantile
    adf_rejection_rate_threshold : float = 0.8
    kpss_verification_threshold : float = 0.2
    na_verification_threshold : float = 0.1

    # Interface knobs
    # Hyperparameter grids (using default factory approach)
    alpha_enet_grid: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    lambda_enet_grid: Optional[List[float]] = field(default_factory=lambda: np.logspace(-4, 2, 20).tolist())
    tau_adalasso_grid: Optional[List[float]] = field(default_factory=lambda: [0.25, 0.5, 1.0, 1.5, 2.0])
    lambda_adalasso_grid: Optional[List[float]] = field(default_factory=lambda: np.logspace(-4, 2, 20).tolist())

    validation_proportion_interface : float = 0.2
    selection_threshold: float = 0.40
    tolerance_threshold: float = 1e-6
    ols_significance_level: float = 0.05
    n_lags: int = 5

    # Adapter knobs
    update_method : str = 'simple_average'
    arma_order : Tuple[int, int] = (1, 1)
    garch_order : Tuple[int, int] = (1, 1)
    dist : str = 'norm' 
    model : str = 'sGARCH'
    archm : bool = False

    # Optional run-id override
    run_id: Optional[str] = None

    


class Orchestrator:
    """
    TTSTL Orchestrator - Coordinates backbone, interface, and adapter modules.
    
    Manages the execution flow of the Two Time-Scale Transfer Learning framework,
    including:
    - Training neural network backbone models
    - Variable selection through interface module
    - Adaptive parameter estimation via adapter module
    - Artifact persistence and provenance tracking
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
        
        # Seed RNGs for reproducibility
        self._seed_rngs()
        
        # Create run directory and write initial artifacts
        self.run_slug = self._create_run_slug()
        self.run_dir = self.runs_root / self.run_slug
        self._create_run_directory()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"Orchestrator initialized with run: {self.run_slug}")

    def global_update_session(self, 
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
        self.logger.info("Starting global_update_session for full trading session")
        
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
            self.logger.info(f"Processing backbone bar {bb} ({i+1}/{total_backbone_bars})")
            
            # Filter data for this specific backbone bar
            event_bb_data = event_df_processed[event_df_processed['backbone_bar'] == bb].copy()
            phys_bb_data = phys_df_processed[phys_df_processed['backbone_bar'] == bb].copy()
            
            # Run global update for this backbone bar
            try:
                results = self.global_update_single_bar(event_bb_data, phys_bb_data)
                all_results.append(results)
                self.logger.info(f"✅ Backbone bar {bb} completed successfully")
            except Exception as e:
                self.logger.error(f"❌ Backbone bar {bb} failed: {str(e)}")
                raise RuntimeError(f"Global update failed for backbone bar {bb}: {str(e)}")
        
        self.logger.info(f"Global update session completed successfully for {total_backbone_bars} backbone bars")
        return all_results

    def local_adapt_session(self, 
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
            self.logger.error("Cannot run local_adapt_session before global_update_session")
            raise RuntimeError("Cannot run local_adapt_session before global_update_session")
            
        self.logger.info("Starting local_adapt_session for full trading session")
        
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
        self._bb_idx = -1  # Will be incremented in local_adapt_single_bar
        
        for i, bb in enumerate(backbone_bars):
            self.logger.info(f"Processing local adaptation for backbone bar {bb} ({i+1}/{total_backbone_bars})")
            
            # Filter data for this specific backbone bar
            event_bb_data = event_df_processed[event_df_processed['backbone_bar'] == bb].copy()
            phys_bb_data = phys_df_processed[phys_df_processed['backbone_bar'] == bb].copy()
            
            self.logger.info(f"Backbone bar {bb} data: {len(event_bb_data):,} observed events, {len(phys_bb_data):,} sampling time bars")
            
            # Run local adaptation for this backbone bar
            try:
                results = self.local_adapt_single_bar(event_bb_data, phys_bb_data)
                all_results.append(results)
                self.logger.info(f"✅ Local adaptation for backbone bar {bb} completed successfully")
            except Exception as e:
                self.logger.error(f"❌ Local adaptation for backbone bar {bb} failed: {str(e)}")
                raise RuntimeError(f"Local adaptation failed for backbone bar {bb}: {str(e)}")
        
        self.logger.info(f"Local adaptation session completed successfully for {total_backbone_bars} backbone bars")
        return all_results

    # New API methods that handle full session processing automatically
    def global_update(self, 
                      event_df: pd.DataFrame, 
                      phys_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Perform global update for the full session (automatic backbone bar iteration).
        
        This is the recommended method that handles all backbone bars automatically.
        For single backbone bar processing, use global_update_single_bar().
        
        Args:
            event_df: Full event-driven trading session data
            phys_df: Full physical-time trading session data
            
        Returns:
            List of results for each backbone bar processed
        """
        return self.global_update_session(event_df, phys_df)

    def local_adapt(self, 
                    event_df: pd.DataFrame, 
                    phys_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Perform local adaptation for the full session (automatic backbone bar iteration).
        
        This is the recommended method that handles all backbone bars automatically.
        For single backbone bar processing, use local_adapt_single_bar().
        
        Args:
            event_df: Full event-driven trading session data
            phys_df: Full physical-time trading session data
            
        Returns:
            List of results for each backbone bar processed
        """
        return self.local_adapt_session(event_df, phys_df)

    # Single backbone bar methods (for advanced users)
    def global_update_single_bar(self, 
                                 event_df: pd.DataFrame, 
                                 phys_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform global update for a single backbone bar.
        
        Note: This is the original global_update behavior. Most users should use
        global_update() which handles all backbone bars automatically.
        
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
            
            # Step 1: Backbone training and feature extraction
            backbone_results = self._run_backbone_global_update(
                event_df_processed, phys_df_processed, bb_dir
            )
            
            # Step 2: Interface global selection
            interface_results = self._run_interface_global_update(
                event_df_processed, phys_df_processed, bb_dir, backbone_results
            )
            
            # Step 3: Adapter fitting
            adapter_results = self._run_adapter_global_update(
                event_df_processed, phys_df_processed, bb_dir, interface_results
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
            
            self.logger.info(f"Global update completed for backbone bar {self._bb_idx:03d}")
            return results
            
        except Exception as e:
            self.logger.error(f"Global update failed for backbone bar {self._bb_idx:03d}: {str(e)}")
            raise RuntimeError(f"Cannot proceed without global update artifacts: {str(e)}")

    def local_adapt_single_bar(self, event_df: pd.DataFrame, phys_df: pd.DataFrame) -> Dict[str, Any]:
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
        if self._bb_idx < 0:
            raise RuntimeError("Cannot run local_adapt before global_update")
            
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
                parsimonious_mode=self.cfg.parsimonious_mode
            )
            
            # Initialize Interface with backbone module and hyperparameter grids
            interface = Interface(
                backbone_module=backbone,
                alpha_enet_grid=self.cfg.alpha_enet_grid,
                lambda_enet_grid=self.cfg.lambda_enet_grid,
                tau_adalasso_grid=self.cfg.tau_adalasso_grid,
                lambda_adalasso_grid=self.cfg.lambda_adalasso_grid
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
                parameters={}  # Start with empty parameters
            )
            
            # Store only the adapter (backbone and interface are accessible through it)
            stack_key = f"backbone_{i}"
            self.stacks[stack_key] = adapter
            
            self.logger.debug(f"Initialized stack {stack_key}")
        
        self.logger.info(f"Successfully initialized {len(self.stacks)} backbone-interface-adapter stacks")
        
        # For backward compatibility, expose the first stack's components as instance attributes
        if self.stacks:
            first_adapter = self.stacks["backbone_0"]
            self.adapter = first_adapter
            self.interface = first_adapter.interface
            self.backbone = first_adapter.interface.backbone

    def _create_backbone_directories(self, mode: str) -> Path:
        """Create directory structure for a backbone bar."""
        bb_dir = self.run_dir / f"bb_{self._bb_idx:03d}" / mode
        
        # Create all required subdirectories
        subdirs = [
            "backbone/architecture/checkpoints",
            "backbone/stationarity", 
            "backbone/correlation",
            "interface",
            "adapter"
        ]
        
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
        # Consider modifying to use the configuration parameters
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(current_backbone.model.parameters(), lr=0.0001)
        
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
                quantile=self.cfg.quantile,
                adf_rejection_rate_threshold=self.cfg.adf_rejection_rate_threshold,
                kpss_verification_threshold=self.cfg.kpss_verification_threshold,
                na_verification_threshold=self.cfg.na_verification_threshold,
                kpss_drop=True,
                correlation_threshold=0.95,  # Could be added to config if needed
                target_col_name=self.cfg.target_col_name,
                save_dir=str(bb_dir / "backbone")  # Enable checkpoints and nn_weights saving
            )
            
            # Save correlation heatmaps using the backbone's method
            # heatmap_dir = bb_dir / "backbone" / "correlation"
            # current_backbone.save_correlation_heatmaps(save_dir=str(heatmap_dir))
            
            # Save training history and model weights to architecture directory
            # architecture_dir = bb_dir / "backbone" / "architecture"
            # current_backbone.save_training_history(save_dir=str(architecture_dir))
            # current_backbone.save_model_weights(save_dir=str(architecture_dir))
            
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

    def _run_interface_global_update(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path, backbone_results: Dict) -> Dict[str, Any]:
        """Run interface global update step."""
        self.logger.info("Running interface global update...")
        
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
                quantile=self.cfg.quantile,
                target_col_name=self.cfg.target_col_name,
                n_lags=self.cfg.n_lags,
                validation_proportion=self.cfg.validation_proportion_interface,
                selection_threshold=self.cfg.selection_threshold,
                tolerance_threshold=self.cfg.tolerance_threshold,
                ols_significance_level=self.cfg.ols_significance_level
            )
            
            # Store interface results in the results structure
            artifacts_dir = bb_dir / "interface"
            
            # Save hyperparameter results to parquet files
            if 'hyperparameter_results' in interface_result:
                hp_results = interface_result['hyperparameter_results']
                
                # Save elastic net hyperparameters (use alpha_grid and lambda_grid)
                if 'elastic_net' in hp_results:
                    enet_data = hp_results['elastic_net']
                    # Create grid results from the grids and best values
                    grid_data = []
                    if 'alpha_grid' in enet_data and 'lambda_grid' in enet_data:
                        for alpha in enet_data['alpha_grid']:
                            for lambda_val in enet_data['lambda_grid']:
                                grid_data.append({
                                    'alpha': alpha,
                                    'lambda': lambda_val,
                                    'best_alpha': enet_data['best_alpha'],
                                    'best_lambda': enet_data['best_lambda']
                                })
                    
                    if grid_data:
                        enet_df = pd.DataFrame(grid_data)
                        enet_df.to_parquet(artifacts_dir / "enet_hyperparams_grid_eval.parquet", index=False)
                
                # Save adaptive lasso hyperparameters (use all_results)
                if 'adaptive_lasso' in hp_results and 'all_results' in hp_results['adaptive_lasso']:
                    ada_results = hp_results['adaptive_lasso']['all_results']
                    if ada_results:
                        ada_df = pd.DataFrame(ada_results)
                        ada_df.to_parquet(artifacts_dir / "adalasso_hyperparams_grid_eval.parquet", index=False)
                
                # Save selected hyperparameters
                selected_hp = {
                    'elastic_net': {
                        'best_alpha': hp_results['elastic_net']['best_alpha'],
                        'best_lambda': hp_results['elastic_net']['best_lambda']
                    },
                    'adaptive_lasso': {
                        'best_tau': hp_results['adaptive_lasso']['best_tau'],
                        'best_lambda': hp_results['adaptive_lasso']['best_lambda']
                    }
                }
                self._atomic_write_json(artifacts_dir / "hyperparams_selected.json", selected_hp)
            
            # Create a placeholder corridor plot (interface doesn't create visualization by default)
            # This matches the expected artifact from the original placeholder
            (artifacts_dir / "corridor_ada_lasso.png").touch()
            
            self.logger.info(f"Interface global update completed: {interface_result['n_selected']} variables selected from {interface_result['n_total_variables']}")
            
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
            # Create placeholder files if interface fails
            artifacts_dir = bb_dir / "interface"
            (artifacts_dir / "corridor_ada_lasso.png").touch()
            (artifacts_dir / "hyperparams_selected.json").touch()
            (artifacts_dir / "enet_hyperparams_grid_eval.parquet").touch()
            (artifacts_dir / "adalasso_hyperparams_grid_eval.parquet").touch()
            
            return {"status": "failed", "error": str(e), "artifacts_created": 4}

    def _run_adapter_global_update(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path, interface_results: Dict) -> Dict[str, Any]:
        """Run adapter global update step."""
        self.logger.info("Running adapter global update...")
        
        try:
            # Create adapter bars using adapter delta
            phys_df = phys_df.copy()
            event_df = event_df.copy()
            
            # Create adapter bar assignments
            phys_df['adapter_bar'] = (np.floor((phys_df['timestamp'] + 1e-6) / self.cfg.adelta).astype(int))
            event_df['adapter_bar'] = (np.floor((event_df['timestamp'] + 1e-6) / self.cfg.adelta).astype(int))
            
            # Get unique adapter bars
            adapter_bars = sorted(phys_df['adapter_bar'].unique())
            self.logger.info(f"Found {len(adapter_bars)} adapter bars for processing")
            
            # Get the adapter from the current stack
            current_adapter = self.adapter
            if current_adapter is None:
                raise RuntimeError("No adapter available for global update")
            
            # Initialize results tracking
            successful_updates = 0
            failed_updates = 0
            all_parameters = {}
            processing_stats = {}
            
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
                            
                            self.logger.debug(f"Added {len(prev_lookback)} lookback events from "
                                            f"sampling bar {last_sampling_bar} of previous adapter bar")
                        else:
                            # Fallback: use current adapter bar data only
                            bar_edlob = event_df[event_df['adapter_bar'] == adapter_bar].copy()
                            self.logger.debug("No previous sampling data available, using current bar only")
                    else:
                        # First adapter bar - no previous data available (will still have NaN for first observation)
                        bar_edlob = event_df[event_df['adapter_bar'] == adapter_bar].copy()
                        self.logger.debug("First adapter bar - no lookback data available")
                    
                    self.logger.debug(f"Processing {bar_logger_prefix}: "
                                    f"{len(bar_ptdata)} physical samples, "
                                    f"{len(bar_edlob)} event samples (including lookback)")
                    
                    # Skip bars with insufficient data
                    min_samples = 10  # Minimum samples required for fitting
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
                    self.logger.debug(f"Running update_adapter for {bar_logger_prefix}")
                    adapter_result = current_adapter.update_adapter(
                        event_driven_adapter_time_bar_data=bar_edlob,
                        physical_time_driven_adapter_time_bar_data=bar_ptdata,
                        adapter_bar_number=i + 1,  # 1-indexed
                        target_col_name=self.cfg.target_col_name
                    )
                    
                    # The adapter.update_adapter method now handles R model summary printing
                    # No need for manual summary generation here
                    
                    # Store results
                    all_parameters[adapter_bar] = adapter_result
                    processing_stats[adapter_bar] = {
                        'status': 'completed',
                        'physical_samples': len(bar_ptdata),
                        'event_samples': len(bar_edlob),
                        'parameters_updated': len(adapter_result) if adapter_result else 0
                    }
                    
                    successful_updates += 1
                    self.logger.info(f"✓ {bar_logger_prefix} completed: "
                                   f"{len(adapter_result) if adapter_result else 0} parameters updated")
                    
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
            
            # Save adapter parameters to file
            if all_parameters:
                params_df = pd.DataFrame.from_dict(all_parameters, orient='index')
                params_file = artifacts_dir / "param_series.parquet"
                params_df.to_parquet(params_file)
                self.logger.debug(f"Saved adapter parameters to {params_file}")
            
            # Save processing statistics
            stats_df = pd.DataFrame.from_dict(processing_stats, orient='index')
            stats_file = artifacts_dir / "processing_stats.parquet"
            stats_df.to_parquet(stats_file)
            
            # Create performance summary (placeholder for now)
            performance_data = {
                'total_adapter_bars': len(adapter_bars),
                'successful_updates': successful_updates,
                'failed_updates': failed_updates,
                'success_rate': successful_updates / len(adapter_bars) if adapter_bars else 0.0
            }
            performance_df = pd.DataFrame([performance_data])
            performance_file = artifacts_dir / "performance_is.parquet"
            performance_df.to_parquet(performance_file)
            
            # Create correlation plot placeholder (actual implementation would require data)
            plot_file = artifacts_dir / "armax_garch_corridor.png"
            plot_file.touch()  # Placeholder for now
            
            # Compile final results
            adapter_global_results = {
                'status': 'completed',
                'total_adapter_bars': len(adapter_bars),
                'successful_updates': successful_updates,
                'failed_updates': failed_updates,
                'success_rate': performance_data['success_rate'],
                'parameters_by_bar': all_parameters,
                'processing_stats': processing_stats,
                'artifacts_created': 4
            }
            
            self.logger.info(f"Adapter global update completed: "
                           f"{successful_updates}/{len(adapter_bars)} bars successful "
                           f"({performance_data['success_rate']:.1%} success rate)")
            
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
        
        return bb_start, bb_end

    def _get_adapter_bar_bounds(self, adapter_bar: int) -> tuple[float, float]:
        """
        Calculate the start and end timestamp bounds for a given adapter bar index.
        
        Args:
            adapter_bar: Adapter bar index (0-based)
            
        Returns:
            Tuple of (start_timestamp, end_timestamp)
        """
        adapter_start = adapter_bar * self.cfg.adelta
        adapter_end = (adapter_bar + 1) * self.cfg.adelta
        return adapter_start, adapter_end

    def _run_backbone_local_adapt(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path) -> Dict[str, Any]:
        """
        Run backbone local adaptation step - backbone remains FROZEN (no updates).
        
        In local adaptation, the backbone model is not retrained or updated.
        We use the frozen model for out-of-sample evaluation and save test results.
        """
        self.logger.info("Running backbone local adaptation (backbone frozen - evaluating out-of-sample)...")
        
        # Filter data for the current backbone bar
        current_bb_event_df = event_df[event_df['backbone_bar'] == self._bb_idx].copy()
        
        self.logger.info(f"Backbone bar {self._bb_idx}: {len(current_bb_event_df)} event samples for evaluation")
        
        # Check if we have sufficient data
        if len(current_bb_event_df) < 10:
            self.logger.warning(f"Insufficient event data for backbone bar {self._bb_idx}: {len(current_bb_event_df)} samples")
            return {"status": "insufficient_data", "event_samples": len(current_bb_event_df)}
        
        # Get the current backbone for this backbone bar
        current_backbone = self.stacks[f"backbone_{self._bb_idx}"].interface.backbone
        
        # The backbone is frozen - no training or updates
        # We only confirm it's ready and perform out-of-sample evaluation
        if not hasattr(current_backbone, 'model') or current_backbone.model is None:
            raise RuntimeError("Backbone model not available. Must run global_update first.")
        
        try:
            # Perform out-of-sample evaluation on the current backbone bar data
            architecture_dir = bb_dir / "backbone" / "architecture"
            test_loss = current_backbone.evaluate_out_of_sample(
                event_driven_data=current_bb_event_df,
                criterion=nn.MSELoss(reduction='sum'),
                batch_size=self.cfg.batch_size,
                quantile=self.cfg.quantile,
                save_dir=str(architecture_dir)
            )
            
            self.logger.info(f"✓ Backbone local adaptation completed (test_loss: {test_loss:.4f})")
            return {
                "status": "completed", 
                "test_loss": test_loss,
                "event_samples": len(current_bb_event_df),
                "artifacts_created": 1  # test_os.json
            }
            
        except Exception as e:
            self.logger.error(f"Backbone local adaptation failed: {str(e)}")
            
            # Create placeholder artifacts for testing
            artifacts_dir = bb_dir / "backbone" / "architecture"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "test_os.json").touch()
            
            return {
                "status": "failed", 
                "error": str(e),
                "event_samples": len(current_bb_event_df),
                "artifacts_created": 1
            }

    def _run_interface_local_adapt(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path) -> Dict[str, Any]:
        """
        Run interface local adaptation step using local_adaptation method.
        
        Uses the existing global hyperparameters but performs local variable selection
        for each adapter bar using Wald tests to identify structural breaks.
        """
        self.logger.info("Running interface local adaptation...")
        
        artifacts_dir = bb_dir / "interface"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Ensure interface has been globally trained
            if not hasattr(self.interface, 'selected_variables') or self.interface.selected_variables is None:
                raise RuntimeError("Interface not globally trained. Must run global_update first.")
            
            # Get the backbone bar data from physical time
            bb_start, bb_end = self._get_backbone_bar_bounds(self._bb_idx)
            bb_mask = (phys_df['backbone_bar'] == self._bb_idx)
            bb_data = phys_df[bb_mask].copy()
            
            if len(bb_data) == 0:
                raise RuntimeError(f"No data found for backbone bar {self._bb_idx}")
            
            self.logger.info(f"Processing backbone bar {self._bb_idx} with {len(bb_data)} samples")
            
            # Extract neural covariates for the backbone bar using interface
            self.logger.info("Extracting neural covariates from backbone...")
            bb_data_with_neural = self.interface._prepare_data(
                event_driven_backbone_time_bar_data=event_df,
                physical_time_driven_backbone_time_bar_data=bb_data,
                target_col_name=self.cfg.target_col_name,
                n_lags=self.cfg.n_lags
            )
            
            self.logger.info(f"Prepared data shape: {bb_data_with_neural.shape}")
            
            # Get unique adapter bars within this backbone bar
            adapter_bars = sorted(bb_data_with_neural['adapter_bar'].unique())
            self.logger.info(f"Found {len(adapter_bars)} adapter bars: {adapter_bars}")
            
            local_results = {}
            successful_bars = 0
            
            # Perform local adaptation for each adapter bar
            for adapter_bar in adapter_bars:
                try:
                    adapter_mask = (bb_data_with_neural['adapter_bar'] == adapter_bar)
                    adapter_data = bb_data_with_neural[adapter_mask].copy()
                    
                    if len(adapter_data) < 10:  # Minimum samples needed
                        self.logger.warning(f"Skipping adapter bar {adapter_bar}: insufficient data ({len(adapter_data)} samples)")
                        continue
                    
                    self.logger.info(f"Running local adaptation for adapter bar {adapter_bar} ({len(adapter_data)} samples)")
                    
                    # Call the interface local_adaptation method
                    local_result = self.interface.local_adaptation(
                        adapter_data, 
                        target_col_name=self.cfg.target_col_name,
                        significance_level=0.05
                    )
                    
                    local_results[f"adapter_bar_{adapter_bar:03d}"] = local_result
                    successful_bars += 1
                    
                    self.logger.info(f"✓ adapter_bar_{adapter_bar:03d} local adaptation completed")
                    
                except Exception as e:
                    self.logger.warning(f"Local adaptation failed for adapter bar {adapter_bar}: {str(e)}")
                    local_results[f"adapter_bar_{adapter_bar:03d}"] = {"error": str(e)}
            
            # Save local adaptation results
            results_file = artifacts_dir / "local_adaptation_results.json"
            import json
            with open(results_file, 'w') as f:
                # Convert numpy types and other non-serializable types to native Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, (bool, np.bool_)):
                        return bool(obj)
                    elif hasattr(obj, 'item'):  # numpy scalar
                        return obj.item()
                    # Skip non-serializable objects like model wrappers
                    elif hasattr(obj, '__class__') and 'Wrapper' in str(type(obj)):
                        return f"<{type(obj).__name__}>"
                    elif hasattr(obj, '__module__') and 'statsmodels' in str(obj.__module__):
                        return f"<{type(obj).__name__}>"
                    elif hasattr(obj, '__class__') and 'Vector' in str(type(obj)):
                        return f"<{type(obj).__name__}>"
                    elif hasattr(obj, '__module__') and 'rpy2' in str(obj.__module__):
                        return f"<{type(obj).__name__}>"
                    return obj
                
                serializable_results = {}
                for key, value in local_results.items():
                    if isinstance(value, dict):
                        def recursively_convert(d):
                            converted = {}
                            for k, v in d.items():
                                # Skip non-serializable adaptive lasso result
                                if k == 'adaptive_lasso_result' and hasattr(v, 'get') and 'ada_model' in str(v):
                                    converted[k] = "<AdaptiveLassoResult - contains non-serializable model>"
                                elif isinstance(v, dict):
                                    converted[k] = recursively_convert(v)
                                else:
                                    converted[k] = convert_numpy(v)
                            return converted
                        serializable_results[key] = recursively_convert(value)
                    else:
                        serializable_results[key] = convert_numpy(value)
                
                json.dump({
                    "backbone_bar": self._bb_idx,
                    "total_adapter_bars": len(adapter_bars),
                    "successful_bars": successful_bars,
                    "success_rate": successful_bars / len(adapter_bars) if adapter_bars else 0,
                    "results": serializable_results
                }, f, indent=2)
            
            self.logger.info(f"Interface local adaptation completed: {successful_bars}/{len(adapter_bars)} bars successful")
            return {
                "status": "completed", 
                "artifacts_created": 1,
                "total_adapter_bars": len(adapter_bars),
                "successful_bars": successful_bars,
                "success_rate": successful_bars / len(adapter_bars) if adapter_bars else 0
            }
            
        except Exception as e:
            self.logger.error(f"Interface local adaptation failed: {str(e)}")
            raise

    def _run_adapter_local_adapt(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path) -> Dict[str, Any]:
        """
        Run adapter local adaptation step.
        
        Updates adapter parameters in the same fashion as global_update method,
        iterating over adapter bars and updating ARMA-GARCH parameters.
        """
        self.logger.info("Running adapter local adaptation...")
        
        artifacts_dir = bb_dir / "adapter"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Ensure adapter has been globally trained
            if not hasattr(self.adapter, 'parameters') or self.adapter.parameters is None:
                raise RuntimeError("Adapter not globally trained. Must run global_update first.")
            
            # Get the backbone bar data from physical time
            bb_start, bb_end = self._get_backbone_bar_bounds(self._bb_idx)
            bb_mask = (phys_df['backbone_bar'] == self._bb_idx)
            bb_data = phys_df[bb_mask].copy()
            
            if len(bb_data) == 0:
                raise RuntimeError(f"No data found for backbone bar {self._bb_idx}")
            
            self.logger.info(f"Processing backbone bar {self._bb_idx} with {len(bb_data)} samples")
            
            # Extract neural covariates for the backbone bar using interface
            self.logger.info("Extracting neural covariates from backbone...")
            bb_data_with_neural = self.interface._prepare_data(
                event_driven_backbone_time_bar_data=event_df,
                physical_time_driven_backbone_time_bar_data=bb_data,
                target_col_name=self.cfg.target_col_name,
                n_lags=self.cfg.n_lags
            )
            
            # Get unique adapter bars within this backbone bar  
            adapter_bars = sorted(bb_data_with_neural['adapter_bar'].unique())
            self.logger.info(f"Found {len(adapter_bars)} adapter bars: {adapter_bars}")
            
            successful_bars = 0
            all_parameters = {}  # Track parameters for all adapter bars
            last_sampling_bar = None  # For lookback strategy
            
            # Process each adapter bar
            for i, adapter_bar in enumerate(adapter_bars):
                try:
                    self.logger.info(f"Preparing data for adapter bar {adapter_bar + 1}...")
                    
                    # Get adapter bar bounds and apply lookback strategy
                    adapter_start, adapter_end = self._get_adapter_bar_bounds(adapter_bar)
                    
                    # Lookback strategy: include last sampling bar from previous adapter bar
                    # This ensures temporal continuity and prevents NaN hidden activations
                    if last_sampling_bar is not None:
                        lookback_start = last_sampling_bar
                        self.logger.debug(f"Using lookback from {lookback_start} to {adapter_end}")
                    else:
                        lookback_start = adapter_start
                        self.logger.debug(f"No lookback available, using {adapter_start} to {adapter_end}")
                    
                    # Filter data for this adapter bar (with potential lookback)
                    adapter_mask = (
                        (bb_data_with_neural['timestamp'] >= lookback_start) & 
                        (bb_data_with_neural['timestamp'] <= adapter_end)
                    )
                    bar_edlob = bb_data_with_neural[adapter_mask].copy()
                    
                    if len(bar_edlob) == 0:
                        self.logger.warning(f"No data found for adapter bar {adapter_bar}")
                        continue
                    
                    # Remember last sampling bar for next iteration
                    if len(bar_edlob) > 0:
                        last_sampling_bar = bar_edlob['timestamp'].iloc[-1]
                    
                    # Filter to only the current adapter bar for training
                    # (lookback data was used for continuity but we train only on current bar)
                    current_bar_mask = (bar_edlob['adapter_bar'] == adapter_bar)
                    current_bar_data = bar_edlob[current_bar_mask].copy()
                    
                    if len(current_bar_data) < 10:  # Minimum samples needed
                        self.logger.warning(f"Skipping adapter bar {adapter_bar}: insufficient data ({len(current_bar_data)} samples)")
                        continue
                    
                    self.logger.info(f"Using all data: {len(current_bar_data)} observations")
                    
                    # Get selected variables from interface
                    if not hasattr(self.interface, 'selected_variables') or self.interface.selected_variables is None:
                        raise RuntimeError("Interface variable selection not available")
                    
                    selected_vars = self.interface.selected_variables
                    self.logger.info(f"Selected variables from interface: {selected_vars}")
                    
                    # Update adapter with this bar's data
                    self.logger.info(f"Fitting GARCH model with {len(current_bar_data)} observations and {len(selected_vars)} covariates...")
                    
                    garch_results = self.adapter.update_adapter(
                        event_driven_adapter_time_bar_data=bar_edlob,
                        physical_time_driven_adapter_time_bar_data=current_bar_data,
                        adapter_bar_number=adapter_bar + 1,  # 1-indexed like global update
                        target_col_name=self.cfg.target_col_name
                    )
                    
                    # Store parameter results for this adapter bar
                    all_parameters[f"adapter_bar_{adapter_bar:03d}"] = garch_results.get('coefficients', {})
                    
                    successful_bars += 1
                    self.logger.info(f"✓ adapter_bar_{adapter_bar:03d} completed: {len(garch_results['coefficients'])} parameters updated")
                    
                except Exception as e:
                    self.logger.warning(f"Adapter update failed for bar {adapter_bar}: {str(e)}")
                    continue
            
            # Save adapter artifacts manually (similar to global_update approach)
            # Create parameter summary files
            if all_parameters:
                # Save parameter data as parquet
                param_df = pd.DataFrame(all_parameters).T  # Transpose to have bars as rows
                param_file = artifacts_dir / "adapter_parameters.parquet"
                param_df.to_parquet(param_file)
            
            # Save performance metrics
            performance_data = {
                'total_bars': len(adapter_bars),
                'successful_bars': successful_bars,
                'failed_bars': len(adapter_bars) - successful_bars,
                'success_rate': successful_bars / len(adapter_bars) if adapter_bars else 0
            }
            performance_df = pd.DataFrame([performance_data])
            performance_file = artifacts_dir / "performance_local.parquet"
            performance_df.to_parquet(performance_file)
            
            success_rate = successful_bars / len(adapter_bars) if adapter_bars else 0
            self.logger.info(f"Adapter local adaptation completed: {successful_bars}/{len(adapter_bars)} bars successful ({success_rate:.1%} success rate)")
            
            return {
                "status": "completed", 
                "artifacts_created": 2,  # parameter and performance files
                "total_adapter_bars": len(adapter_bars),
                "successful_bars": successful_bars,
                "success_rate": success_rate
            }
            
        except Exception as e:
            self.logger.error(f"Adapter local adaptation failed: {str(e)}")
            raise

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
