"""
TTSTL Orchestrator - Coordinates Backbone, Interface, and Adapter modules.

This module provides the main orchestration logic for the Two Time-Scale Transfer Learning
framework, managing the execution of backbone training, interface variable selection,
and adapter parameter updates across trading sessions.
"""

import logging
import os
import secrets
import json
import yaml
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

import pandas as pd
import numpy as np
import torch

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

    # Backbone / Interface / Adapter knobs
    quantile: float = 0.005
    parsimonious_mode: bool = False
    batch_size: int = 32
    epochs: int = 12
    selection_threshold: float = 0.15
    tolerance_threshold: float = 1e-6
    ols_significance_level: float = 0.05

    # Train/validation split parameters
    validation_proportion: float = 0.1
    buffer_proportion: float = 0.05

    # Hyperparameter grids (explicitly included)
    alpha_enet_grid: Optional[List[float]] = None
    lambda_enet_grid: Optional[List[float]] = None
    tau_adalasso_grid: Optional[List[float]] = None
    lambda_adalasso_grid: Optional[List[float]] = None

    # Optional run-id override
    run_id: Optional[str] = None

    def __post_init__(self):
        """Set default grids if None provided."""
        if self.alpha_enet_grid is None:
            self.alpha_enet_grid = [0.1, 0.5, 0.9]
        if self.lambda_enet_grid is None:
            self.lambda_enet_grid = np.logspace(-4, 0, 20).tolist()
        if self.tau_adalasso_grid is None:
            self.tau_adalasso_grid = [0.5, 1.0, 1.5, 2.0]
        if self.lambda_adalasso_grid is None:
            self.lambda_adalasso_grid = np.logspace(-4, 0, 20).tolist()


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
        
        # Seed RNGs for reproducibility
        self._seed_rngs()
        
        # Create run directory and write initial artifacts
        self.run_slug = self._create_run_slug()
        self.run_dir = self.runs_root / self.run_slug
        self._create_run_directory()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info(f"Orchestrator initialized with run: {self.run_slug}")

    def global_update(self, event_df: pd.DataFrame, phys_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform global update: full backbone training, interface selection, adapter fitting.
        
        Args:
            event_df: Event-driven trading session data
            phys_df: Physical-time trading session data
            
        Returns:
            Dict containing key metrics, selected variables, and artifact paths
        """
        # Increment backbone index and create directory structure
        self._bb_idx += 1
        bb_dir = self._create_backbone_directories("global_update")
        
        self.logger.info(f"Starting global_update for backbone bar {self._bb_idx:03d}")
        
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

    def local_adapt(self, event_df: pd.DataFrame, phys_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform local adaptation: frozen backbone, interface local selection, adapter updates.
        
        Args:
            event_df: Event-driven trading session data
            phys_df: Physical-time trading session data
            
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
        """Seed all random number generators for reproducibility."""
        import random
        
        # Seed Python RNG
        random.seed(self.cfg.seed)
        
        # Seed NumPy
        np.random.seed(self.cfg.seed)
        
        # Seed PyTorch
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.cfg.seed)
            torch.cuda.manual_seed_all(self.cfg.seed)
            # For deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.logger.debug(f"Seeded RNGs with seed: {self.cfg.seed}")

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
        
        # Write config.yaml
        config_path = self.run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self.cfg), f, default_flow_style=False)
        
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
        """Initialize Backbone, Interface, and Adapter components."""
        # Initialize CNN-LSTM model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNNLSTM().to(device)
        
        # Initialize Backbone
        self.backbone = Backbone(
            model=model,
            delta_adapter=self.cfg.adelta,
            delta_sample=self.cfg.sdelta,
            parsimonious_mode=self.cfg.parsimonious_mode
        )
        
        # Initialize Interface with backbone module and hyperparameter grids
        self.interface = Interface(
            backbone_module=self.backbone,
            alpha_enet_grid=self.cfg.alpha_enet_grid,
            lambda_enet_grid=self.cfg.lambda_enet_grid,
            tau_adalasso_grid=self.cfg.tau_adalasso_grid,
            lambda_adalasso_grid=self.cfg.lambda_adalasso_grid
        )
        
        # Initialize Adapter with interface
        self.adapter = Adapter(
            interface=self.interface
        )
        
        self.logger.debug("Components initialized successfully")

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
        
        # Add backbone_bar column
        if 'backbone_bar' not in df.columns:
            df['backbone_bar'] = (df['timestamp'] // self.cfg.bdelta).astype(int)
        
        # Add adapter_bar column  
        if 'adapter_bar' not in df.columns:
            df['adapter_bar'] = (df['timestamp'] // self.cfg.adelta).astype(int)
        
        return df

    def _run_backbone_global_update(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path) -> Dict[str, Any]:
        """Run backbone global update step."""
        self.logger.info("Running backbone global update...")
        
        # Placeholder for backbone global update
        # TODO: Implement actual backbone training
        
        # Create placeholder artifacts
        artifacts_dir = bb_dir / "backbone"
        
        # Create placeholder files
        (artifacts_dir / "architecture" / "train_log.parquet").touch()
        (artifacts_dir / "architecture" / "best.pt").touch()
        (artifacts_dir / "stationarity" / "stationarity_raw.parquet").touch()
        (artifacts_dir / "stationarity" / "stationarity_summary.parquet").touch()
        (artifacts_dir / "correlation" / "corr_matrix.npz").touch()
        (artifacts_dir / "correlation" / "corr_heatmap.png").touch()
        (artifacts_dir / "extended_covariates.json").touch()
        (artifacts_dir / "nn_weights.npz").touch()
        
        return {"status": "completed", "artifacts_created": 8}

    def _run_interface_global_update(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path, backbone_results: Dict) -> Dict[str, Any]:
        """Run interface global update step."""
        self.logger.info("Running interface global update...")
        
        # Placeholder for interface global update
        # TODO: Implement actual interface global selection
        
        artifacts_dir = bb_dir / "interface"
        
        # Create placeholder files
        (artifacts_dir / "corridor_ada_lasso.png").touch()
        (artifacts_dir / "hyperparams_selected.json").touch()
        (artifacts_dir / "enet_hyperparams_grid_eval.parquet").touch()
        (artifacts_dir / "adalasso_hyperparams_grid_eval.parquet").touch()
        
        return {"status": "completed", "artifacts_created": 4}

    def _run_adapter_global_update(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path, interface_results: Dict) -> Dict[str, Any]:
        """Run adapter global update step."""
        self.logger.info("Running adapter global update...")
        
        # Placeholder for adapter global update
        # TODO: Implement actual adapter fitting
        
        artifacts_dir = bb_dir / "adapter"
        
        # Create placeholder files
        (artifacts_dir / "armax_garch_corridor.png").touch()
        (artifacts_dir / "param_series.parquet").touch()
        (artifacts_dir / "performance_is.parquet").touch()
        
        return {"status": "completed", "artifacts_created": 3}

    def _run_backbone_local_adapt(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path) -> Dict[str, Any]:
        """Run backbone local adaptation step."""
        self.logger.info("Running backbone local adaptation...")
        
        # Placeholder for backbone local adaptation
        # TODO: Implement actual backbone OOS testing
        
        artifacts_dir = bb_dir / "backbone"
        (artifacts_dir / "test_os.json").touch()
        
        return {"status": "completed", "artifacts_created": 1}

    def _run_interface_local_adapt(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path) -> Dict[str, Any]:
        """Run interface local adaptation step."""
        self.logger.info("Running interface local adaptation...")
        
        # Placeholder for interface local adaptation
        # TODO: Implement actual interface local selection
        
        artifacts_dir = bb_dir / "interface"
        (artifacts_dir / "corridor_ada_lasso.png").touch()
        (artifacts_dir / "structural_breaks.parquet").touch()
        
        return {"status": "completed", "artifacts_created": 2}

    def _run_adapter_local_adapt(self, event_df: pd.DataFrame, phys_df: pd.DataFrame, bb_dir: Path) -> Dict[str, Any]:
        """Run adapter local adaptation step."""
        self.logger.info("Running adapter local adaptation...")
        
        # Placeholder for adapter local adaptation
        # TODO: Implement actual adapter local updates
        
        artifacts_dir = bb_dir / "adapter"
        (artifacts_dir / "armax_garch_corridor.png").touch()
        (artifacts_dir / "param_series.parquet").touch()
        (artifacts_dir / "performance_os.parquet").touch()
        
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
        import torch
        import pandas
        import numpy
        
        return {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "torch": torch.__version__,
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
            # TODO: Add rpy2, R, glmnet versions
        }

    def _atomic_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Atomically write JSON data to file."""
        tmp_path = path.with_suffix(path.suffix + '.tmp')
        
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        os.replace(tmp_path, path)
