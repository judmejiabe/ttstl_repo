# Standard library imports
import copy
import json
import logging
import os
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

# R interface imports
from rpy2.rinterface_lib import callbacks
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Local application imports
from ttstl.utils.data_preprocessor import create_pretrained_scaler, get_dataset_cnn_lstm


# Suppress R console output to reduce noise from stationarity tests
def nullify_output(x):
    pass


callbacks.consolewrite_print = nullify_output
callbacks.consolewrite_warnerror = nullify_output

try:
    urca = importr("urca")
    fUnitRoots = importr("fUnitRoots")
except Exception as e:
    warnings.warn(f"Could not import R packages urca or fUnitRoots: {e}")
    urca = None
    fUnitRoots = None

pandas2ri.activate()


class Backbone:
    def __init__(
        self,
        model: nn.Module,
        delta_adapter: float = 0.0,
        delta_sample: float = 0.0,
        parsimonious_mode: bool = False,
        neural_covariates_only: bool = False,
        manual_covariates_only: bool = False,
        pretrained_scaler: Optional[StandardScaler] = None,
        lower_bounds: Optional[Dict[str, float]] = None,
        upper_bounds: Optional[Dict[str, float]] = None,
        past_returns: bool = False,
    ) -> None:
        """Initialize the Backbone with a model and delta adapter.
        
        Parameters
        ----------
        model : nn.Module
            The neural network model to be used.
        delta_adapter : float, optional
            Adapter time bar length in the same units of time as the data. 0.0 by default.
        delta_sample : float, optional
            Sample time bar length in the same units of time as the data. 0.0 by default.
        parsimonious_mode : bool, optional
            Whether to use parsimonious mode for neural covariate extraction. False by default.
        neural_covariates_only : bool, optional
            Whether to use only neural covariates (no manual/stationary features). False by default.
        manual_covariates_only : bool, optional
            Whether to use only manual/stationary features (no neural covariates). False by default.
        pretrained_scaler : StandardScaler, optional
            Pretrained scaler for consistent preprocessing. None by default.
        lower_bounds : Dict[str, float], optional
            Lower quantile bounds for winsorization. None by default.
        upper_bounds : Dict[str, float], optional
            Upper quantile bounds for winsorization. None by default.
        past_returns : bool, optional
            Whether to use past returns (concurrent returns) instead of future returns. Defaults to False.
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.delta_adapter = delta_adapter
        self.delta_sample = delta_sample
        self.parsimonious_mode = (
            parsimonious_mode if not manual_covariates_only else True
        )
        self.neural_covariates_only = neural_covariates_only
        self.manual_covariates_only = manual_covariates_only
        self.past_returns = past_returns
        self.full_covariate_list = None
        self.dropped_pairs = dict()

        # Preprocessing components for consistent data processing
        self.pretrained_scaler = pretrained_scaler
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Validate mutually exclusive modes
        if self.neural_covariates_only and self.manual_covariates_only:
            raise ValueError(
                "neural_covariates_only and manual_covariates_only cannot both be True"
            )

        # Correlation matrix attributes
        self.neural_covariates_correlation = None
        self.manually_extracted_covariates_correlation = None
        self.cross_correlation_matrix = None

        # Variable lists for heatmap plotting (stored for external plotting)
        self.neural_covariate_columns = None
        self.stationary_variables = None
        self.available_extra_covariates = None
        # Considers the pretrained model in the batch gradient descent
        self.pretrained = False

        self.logger = logging.getLogger("ttstl.backbone")

    def set_preprocessing_components(
        self,
        pretrained_scaler: StandardScaler,
        lower_bounds: Dict[str, float],
        upper_bounds: Dict[str, float],
    ) -> None:
        """
        Set or update the preprocessing components (scaler and bounds).
        
        This method allows setting preprocessing components after backbone initialization,
        which is useful when the scaler is created during training or loaded from disk.
        
        Parameters
        ----------
        pretrained_scaler : StandardScaler
            Fitted StandardScaler for feature preprocessing
        lower_bounds : Dict[str, float]
            Dictionary mapping feature names to lower quantile bounds
        upper_bounds : Dict[str, float]
            Dictionary mapping feature names to upper quantile bounds
        """
        self.pretrained_scaler = pretrained_scaler
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.logger.info("Updated backbone preprocessing components")
        self.logger.info(f"Scaler available: {self.pretrained_scaler is not None}")
        self.logger.info(
            f"Bounds available: {self.lower_bounds is not None and self.upper_bounds is not None}"
        )
        if self.lower_bounds is not None:
            self.logger.info(
                f"Bounds available for features: {list(self.lower_bounds.keys())}"
            )

    def has_preprocessing_components(self) -> bool:
        """
        Check if all preprocessing components are available.
        
        Returns
        -------
        bool
            True if scaler and bounds are available, False otherwise
        """
        return (
            self.pretrained_scaler is not None
            and self.lower_bounds is not None
            and self.upper_bounds is not None
        )

    def _one_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.modules.loss,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        """Performs batch gradient descent on the provided dataset.
        
        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training data.
        criterion : nn.modules.loss
            Loss function.
        optimizer : torch.optim.Optimizer, optional
            Optimizer for the model.
        """

        is_train = optimizer is not None
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        n_samples = 0

        with torch.set_grad_enabled(is_train):
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_size = inputs.size(0)
                n_samples += batch_size
                total_loss += loss.item()

        return total_loss / n_samples

    def batch_gradient_descent(
        self,
        train_loader: DataLoader,
        criterion: nn.modules.loss,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epochs: int = 10,
        validation_loader: Optional[DataLoader] = None,
        save_checkpoints: bool = False,
        checkpoint_dir: Optional[str] = None,
    ) -> float:
        """Train the model for specified epochs with optional checkpoint saving.
        
        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for the training data.
        criterion : nn.modules.loss
            Loss function.
        optimizer : torch.optim.Optimizer
            Optimizer for the model.
        epochs : int
            Number of training epochs.
        validation_loader : DataLoader, optional
            DataLoader for validation data.
        save_checkpoints : bool, optional
            Whether to save checkpoints every epoch. False by default.
        checkpoint_dir : str, optional
            Directory to save checkpoints.
            
        Returns
        -------
        float
            Best validation loss achieved.
        """
        self.logger.info("Starting training...")
        validation_loader = validation_loader or train_loader
        train_losses = []
        validation_losses = []
        best_model = copy.deepcopy(self.model)
        if self.pretrained:
            best_validation_loss = self._one_epoch(validation_loader, criterion)
            train_losses.append(-1)
            validation_losses.append(best_validation_loss)
        else:
            best_validation_loss = float("inf")
        if self.pretrained:
            self.logger.info(
                f"Initial Mean Validation Loss: {best_validation_loss:.4f}"
            )
        else:
            self.logger.info("No pretrained model detected, starting fresh training.")
        best_epoch = 0

        # Setup checkpoint directory if needed
        if save_checkpoints and checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Saving checkpoints to: {checkpoint_path}")

        for epoch in range(epochs):

            self.logger.info(f"Epoch {epoch + 1}/{epochs}")

            self._one_epoch(train_loader, criterion, optimizer)
            this_train_loss = self._one_epoch(train_loader, criterion)
            this_validation_loss = self._one_epoch(validation_loader, criterion)

            validation_losses.append(this_validation_loss)
            train_losses.append(this_train_loss)

            if this_validation_loss < best_validation_loss:
                best_validation_loss = this_validation_loss
                best_model = copy.deepcopy(
                    self.model
                )  # Deep copy to avoid reference issues
                best_epoch = epoch + 1

            self.logger.info(
                f"Mean Train Loss: {this_train_loss:.4f}, Mean Validation Loss: {this_validation_loss:.4f}"
            )

            # Save checkpoint every epoch if requested
            if save_checkpoints and checkpoint_dir:
                checkpoint_file = checkpoint_path / f"epoch_{epoch + 1:04d}.pt"
                checkpoint_data = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    if optimizer
                    else None,
                    "train_loss": this_train_loss,
                    "validation_loss": this_validation_loss,
                    "best_validation_loss": best_validation_loss,
                }
                torch.save(checkpoint_data, checkpoint_file)
                self.logger.info(f"Saved checkpoint: {checkpoint_file}")

        # Handle epochs numbering when pretrained model exists
        if self.pretrained:
            # Include epoch 0 for pretrained model validation loss
            epoch_numbers = [0] + list(range(1, epochs + 1))
        else:
            epoch_numbers = list(range(1, epochs + 1))

        # Create training history dataframe
        self.training_history_df = pd.DataFrame(
            {
                "epoch": epoch_numbers,
                "train_loss": train_losses,
                "validation_loss": validation_losses,
            }
        )

        if epochs > 0:
            self.pretrained = True

        self.logger.info(
            f"Training complete. Best Mean Validation Loss: {best_validation_loss:.4f} at epoch {best_epoch}"
        )

        self.model = best_model
        return best_validation_loss

    def _run_stationarity_tests(self, series: pd.Series) -> dict:
        """Conducts the Augmented Dickey-Fuller (ADF), Philips Perron (PP), and
        Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity tests on a given 
        Pandas series. The results are returned in dictionary with keys 'ADF',
        'PP', 'KPSS', and 'NA' where the value corresponds to 1 if rejection
        and zero otherwise. If any of the tests fail, the 'NA' key will be set
        to 1.
        
        Parameters
        ----------
        series : pd.Series
            Pandas Series to test for stationarity.
            
        Returns
        -------
        dict
            Dictionary with keys 'ADF', 'PP', 'KPSS', and 'NA' with 
            values indicating test results.
        """

        alpha = 0.05
        series_r = pandas2ri.py2rpy(series)
        try:
            adf = fUnitRoots.adfTest(series_r, lags=10, type="c")
            adf_rej = adf.slots["test"].rx2("p.value")[0] < alpha
            pp = urca.ur_pp(series_r, type="Z-tau", model="constant", lags="long")
            pp_rej = pp.slots["teststat"][0] < pp.slots["cval"][0, 1]
            kpss = urca.ur_kpss(series_r, type="mu", lags="long")
            kpss_rej = kpss.slots["teststat"][0] > kpss.slots["cval"][0, 1]
            return {
                "ADF": int(adf_rej),
                "PP": int(pp_rej),
                "KPSS": int(kpss_rej),
                "NA": 0,
            }
        except Exception as e:
            return {"ADF": 0, "PP": 0, "KPSS": 0, "NA": 1}

    def _run_multiple_stationarity_tests(
        self, data: pd.DataFrame, session_start: float = 0.0
    ) -> pd.DataFrame:

        """Perform multiple stationarity tests on the provided data. 
        The testing frequency is determined by the delta_adapter attribute.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing Phyisical Time Driven data.
        session_start : float, optional
            Session start time. 0.0 by default.
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the results of the stationarity tests.
        """

        columns = data.columns.tolist()
        if "timestamp" in columns:
            columns.remove("timestamp")

        try:
            data["adapter_bar"] = np.floor(
                (data["timestamp"] - session_start + 1e-6) / self.delta_adapter
            ).astype(int)
        except KeyError:
            data["adapter_bar"] = np.floor(
                (data.index.to_series() - session_start + 1e-6) / self.delta_adapter
            ).astype(int)

        records = []

        for ell, group in data.groupby("adapter_bar"):
            for column in columns:
                if column in {"adapter_bar", "backbone_bar"}:
                    continue
                series = group[column]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                tests = self._run_stationarity_tests(series)
                records.append(
                    {
                        "adapter_bar": ell,
                        "variable": column,
                        "ADF": tests["ADF"],
                        "PP": tests["PP"],
                        "KPSS": tests["KPSS"],
                        "NA": tests["NA"],
                    }
                )
                if tests["NA"] == 1:
                    self.logger.info(
                        f"At least one stationarity test failed for variable {column} in adapter bar {ell}. This could be due to lack of variation, excessive variation, or missingness."
                    )

        rejections_df = pd.DataFrame.from_records(records)

        return rejections_df

    def select_stationary_variables(
        self,
        data: pd.DataFrame,
        adf_rejection_rate_threshold: float = 0.8,
        kpss_verification_threshold: float = 0.2,
        na_verification_threshold: float = 0.1,
        kpss_drop: bool = True,
        target_col_name: str = "micro_return",
        save_dir: Optional[str] = None,
        session_start: float = 0.0,
    ) -> pd.DataFrame:
        """Select stationary variables from the DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the time series data.
        adf_rejection_rate_threshold : float, optional
            ADF rejection rate threshold for stationarity. 0.8 by default.
        kpss_verification_threshold : float, optional
            KPSS verification threshold. 0.2 by default.
        na_verification_threshold : float, optional
            NA verification threshold. 0.1 by default.
        kpss_drop : bool, optional
            Whether to use strict KPSS dropping. True by default.
        target_col_name : str, optional
            Name of the target variable column. 'micro_return' by default.
        save_dir : str, optional
            Directory to save stationarity test results.
        session_start : float, optional
            Session start time. 0.0 by default.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with only stationary variables.
        """
        self.logger.info(f"Running stationarity tests with kpss_drop:{kpss_drop}")

        rejections_df = self._run_multiple_stationarity_tests(
            data, session_start=session_start
        )

        grouped = rejections_df.groupby("variable")[["ADF", "KPSS", "NA"]].mean()

        # Save stationarity test results if save_dir provided
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save raw stationarity test results per variable × adapter_bar
            stationarity_raw_path = save_path / "stationarity_raw.parquet"
            rejections_df.to_parquet(stationarity_raw_path, index=False)
            self.logger.info(
                f"Saved raw stationarity test results to {stationarity_raw_path}"
            )

            # Save aggregated rates per variable
            stationarity_summary_path = save_path / "stationarity_summary.parquet"
            grouped.to_parquet(
                stationarity_summary_path, index=True
            )  # Keep variable as index
            self.logger.info(
                f"Summary of stationarity tests saved to to {stationarity_summary_path}"
            )

        # Define base condition for stationarity
        adf_stationary = grouped["ADF"] > adf_rejection_rate_threshold
        high_na = grouped["NA"] > na_verification_threshold

        if kpss_drop:
            # Strict stationarity: both ADF and KPSS must agree
            kpss_stationary = grouped["KPSS"] < kpss_verification_threshold
            stationary_variables = grouped.index[
                adf_stationary & kpss_stationary
            ].tolist()
            warning_variables = []
        else:
            # Lenient stationarity: only ADF required
            stationary_variables = grouped.index[adf_stationary].tolist()
            # Variables that pass ADF but fail KPSS need verification
            kpss_warning = grouped["KPSS"] > kpss_verification_threshold
            warning_variables = grouped.index[adf_stationary & kpss_warning].tolist()

        # Variables with high NA rates (among those passing ADF)
        high_na_vars = grouped.index[adf_stationary & high_na].tolist()

        if target_col_name in stationary_variables:
            self.logger.info(
                f"Target column '{target_col_name}' passed stationarity tests."
            )
        else:
            self.logger.warning(
                f"Target column '{target_col_name}' does not pass stationarity tests."
            )

        # Populate covariate list from stationary variables
        self.full_covariate_list = [
            var
            for var in stationary_variables
            if var not in {"timestamp", target_col_name}
        ]

        self.logger.info("=" * 80)
        self.logger.info("Stationary covariates found")
        self.logger.info("=" * 80)

        for i, var in enumerate(self.full_covariate_list):
            self.logger.info(f"{i + 1}. {var}")

        self.logger.info("=" * 80)

        # Issue warnings
        if warning_variables:
            self.logger.warning(
                f"Variables meeting ADF threshold but requiring verification for high KPSS rejections: {warning_variables}"
            )

        if high_na_vars:
            self.logger.warning(
                f"Variables passing stationarity criteria but with high NA rates: {high_na_vars}"
            )

        return stationary_variables

    def _add_covariate_set(self, extra_covariate_set: list) -> None:
        """Add extra covariate set to the full variable set.
        
        Parameters
        ----------
        extra_covariate_set : list
            List of additional covariates to add.
        """
        if self.full_covariate_list is None:
            self.full_covariate_list = extra_covariate_set
        else:
            self.full_covariate_list += extra_covariate_set
            # Remove duplicates
            seen = set()
            self.full_covariate_list = [
                x for x in self.full_covariate_list if not (x in seen or seen.add(x))
            ]

    def drop_highly_correlated_variables(
        self, data: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """Drops highly correlated variables from data at a random order.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the time series data.
        threshold : float, optional
            Correlation threshold. 0.95 by default.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with covariance matrix analysis results.
        """

        self.logger.info("Dropping highly correlated variables")

        self.logger.info(
            f"Initial covariate set size: {len(self.full_covariate_list):,}"
        )

        corr_matrix = data[self.full_covariate_list].corr(method="pearson")
        # Determine highly correlated pairs
        highly_correlated_pairs = corr_matrix.abs() > threshold
        adjacency_dict = {}
        for var in highly_correlated_pairs.columns:
            if sum(highly_correlated_pairs[var]) > 1:
                adjacency_dict[var] = set(
                    highly_correlated_pairs.index[highly_correlated_pairs[var]].tolist()
                ) - {var}
        self.logger.info(f"=" * 80)
        self.logger.info("Highly correlated variables")
        self.logger.info("=" * 80)

        for var, neighbors in adjacency_dict.items():
            self.logger.info(f"Variable {var} is highly correlated with : {neighbors}")

        self.logger.info("=" * 80)

        shuffled_variables = np.random.permutation(list(adjacency_dict.keys())).tolist()
        dropped_vars = set()

        for var in shuffled_variables:
            if var not in adjacency_dict:
                continue
            self.logger.info(f"Dropping variable {var}")
            for neighbor in list(adjacency_dict[var]):
                self.dropped_pairs[var] = neighbor
                self.dropped_pairs[neighbor] = var
                adjacency_dict[neighbor].remove(var)
                if not adjacency_dict[neighbor]:
                    del adjacency_dict[neighbor]
            del adjacency_dict[var]
            dropped_vars.add(var)

        self.full_covariate_list = [
            var for var in self.full_covariate_list if var not in dropped_vars
        ]

        self.logger.info("=" * 80)
        self.logger.info("Retained Variables")
        self.logger.info("=" * 80)
        for i, var in enumerate(self.full_covariate_list):
            self.logger.info(f"{i + 1}. {var}")

        self.logger.info("=" * 80)

        return corr_matrix

    def extract_neural_covariates(
        self,
        event_driven_data: pd.DataFrame,
        physical_time_data: pd.DataFrame,
        batch_size: int = 512,
        get_output: bool = False,
    ) -> pd.DataFrame:
        """Extract neural network covariates from event-driven data for physical time-driven data.
        
        For each timestamp in the physical time-driven data, find the last row in the event-driven
        data with a timestamp smaller than the physical timestamp, and extract neural network
        features (hidden activations or forward pass output) for that row.
        
        Parameters
        ----------
        event_driven_data : pd.DataFrame
            Event-driven data with timestamps.
        physical_time_data : pd.DataFrame
            Physical time-driven data with timestamps.
        batch_size : int, optional
            Batch size for processing. 512 by default.
        get_output : bool, optional
            If True, get model output instead of hidden activations. False by default.
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted neural covariates.
        """
        try:
            self.logger.info(
                f"Extracting neural covariates in {'parsimonious' if self.parsimonious_mode else 'full'} mode"
            )

            # Validate that preprocessing components are available
            if not self.has_preprocessing_components():
                raise ValueError(
                    "Preprocessing components (scaler and bounds) are required for neural covariate extraction. "
                    "Use set_preprocessing_components() to provide them or initialize backbone with these components."
                )

            # Create dataset from event-driven data (using train=False to keep all data)
            event_dataset = get_dataset_cnn_lstm(
                event_driven_data,
                delta=self.delta_sample,
                pretrained_scaler=self.pretrained_scaler,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                train=False,
                past_returns=self.past_returns,
            )

            # Create a DataLoader for batch processing with larger batch size for efficiency
            if self.device != torch.device("cpu"):
                event_loader = DataLoader(
                    event_dataset,
                    batch_size=batch_size,
                    num_workers=4,
                    pin_memory=True,
                    shuffle=False,
                )
            else:
                event_loader = DataLoader(
                    event_dataset, batch_size=batch_size, shuffle=False
                )

            self.logger.info(
                f"Processing {len(event_dataset)} samples in batches of {batch_size}"
            )

            # Extract features using the trained model
            self.model.eval()
            representations = []

            with torch.no_grad():
                for inputs, targets in event_loader:
                    inputs = inputs.to(self.device, dtype=torch.float32)

                    # Extract features based on mode
                    if self.parsimonious_mode or get_output:
                        # Use forward pass output (1-dimensional)
                        # Shape: (batch_size, 1)
                        rep = self.model(inputs)
                    else:
                        # Use hidden activations (higher dimensional)
                        rep = self.model.hidden_activations(inputs)

                    representations.append(rep.cpu())

            # Concatenate all predictions and convert to numpy
            representations = torch.cat(representations).numpy()

            reps = []

            # Get event-driven timestamps
            event_timestamps = event_driven_data["timestamp"].values

            # Get physical time timestamps
            if "timestamp" in physical_time_data.columns:
                physical_timestamps = physical_time_data["timestamp"].values
            else:
                physical_timestamps = physical_time_data.index.values

            # Vectorized searchsorted operation
            indices = np.searchsorted(
                event_timestamps, physical_timestamps, side="right"
            )

            # Handle edge cases: set indices to 0 where no prior event exists
            valid_mask = indices > 0
            indices = np.maximum(indices - 1, 0)
            # Vectorized representation extraction
            reps = representations[indices]
            reps[~valid_mask] = np.nan

            # Create a DataFrame with timestamps and representations,
            # set the representations as columns. Name the columns
            # as hidden_activations_i if not parsimonious_mode else forward_pass_i
            covariates_df = pd.DataFrame(
                np.hstack([np.array(physical_timestamps).reshape(-1, 1), reps]),
                columns=None,
            )

            # Name the columns according to the model's output layer
            if self.parsimonious_mode or get_output:
                prefix = "forward_pass_"
            else:
                prefix = "hidden_activations_"

            covariates_df.columns = ["timestamp"] + [
                f"{prefix}{i}" for i in range(reps.shape[1])
            ]

            self.logger.info(
                f"Created {covariates_df.shape[1] - 1:,} new neural covariates. These will appear with {prefix} prefix."
            )

            return covariates_df

        except ValueError as e:
            # Re-raise ValueError as-is (these are usually validation errors with useful messages)
            self.logger.error(
                f"Validation error in neural covariate extraction: {str(e)}"
            )
            raise e
        except Exception as e:
            self.logger.error(f"Error extracting neural covariates: {str(e)}")
            traceback.print_exc()
            raise RuntimeError("Neural covariate extraction failed")
            return {}

    def _compute_correlation_matrices(
        self,
        data: pd.DataFrame,
        neural_covariates_names: List,
        microstructural_covariates_names: List,
        target_col_name,
        save_dir: Path,
    ) -> None:
        """
        Compute and store correlation matrices for neural and manually extracted covariates.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        neural_covariates_names : List
            List of names of neural covariates.
        microstructural_covariates_names : List
            List of names of microstructural covariates.
        target_col_name : str
            Name of the target column.
        save_dir : Path
            Directory to save results.
        """

        self.logger.info(
            f"Computing correlation matrices for {len(neural_covariates_names)} neural covariates and {len(microstructural_covariates_names)} microstructural covariates."
        )

        # Prepare correlation data including extra covariates
        correlation_variables = (
            [target_col_name]
            + neural_covariates_names
            + microstructural_covariates_names
        )
        correlation_data = data[correlation_variables].copy()

        # Count and log rows with at least one NA value
        na_rows_count = correlation_data.isnull().any(axis=1).sum()
        total_rows = len(correlation_data)
        self.logger.info(
            f"Found {na_rows_count} rows with at least one NA value out of {total_rows} total rows ({na_rows_count/total_rows*100:.2f}%)"
        )

        # Identify which variables have missing values
        na_by_variable = correlation_data.isnull().sum()
        variables_with_na = na_by_variable[na_by_variable > 0]
        if len(variables_with_na) > 0:
            self.logger.info(
                f"Variables with missing values: {dict(variables_with_na)}"
            )

        if na_rows_count > 0:
            self.logger.info("Dropping rows with NaN values for correlation analysis")
            correlation_data = correlation_data.dropna()

        if self.parsimonious_mode:
            # Parsimonious mode: Create single correlation matrix with neural covariates below extra covariates
            self.logger.info(
                "Using parsimonious mode: creating single correlation matrix"
            )

            self.neural_covariates_correlation = correlation_data.corr(method="pearson")
            self.manually_extracted_covariates_correlation = None
            self.cross_correlation_matrix = None

            self.logger.info(
                f"Computed a single correlation matrix with all variables, shape: {self.neural_covariates_correlation.shape}."
            )
            self.logger.info(
                "The correlation matrix can be accessed via the 'neural_covariates_correlation' attribute."
            )

        else:
            # Non-parsimonious mode: Create separate correlation matrices as before
            self.logger.info(
                "Using full mode: creating separate correlation matrices for the microstructural features and neural features"
            )

            # Initialize all to None
            self.manually_extracted_covariates_correlation = None
            self.neural_covariates_correlation = None
            self.cross_correlation_matrix = None

            # 1. Microstructural covariates correlation (only if we have microstructural covariates)
            if len(microstructural_covariates_names) > 0:
                microstructural_with_target = [
                    target_col_name
                ] + microstructural_covariates_names
                microstructural_data = correlation_data[microstructural_with_target]
                self.manually_extracted_covariates_correlation = microstructural_data.corr(
                    method="pearson"
                )
                self.logger.info(
                    f"Computed microstructural covariates correlation matrix with shape: {self.manually_extracted_covariates_correlation.shape}."
                )
                self.logger.info(
                    "The correlation matrix can be accessed via the 'manually_extracted_covariates_correlation' attribute."
                )
            else:
                self.logger.info(
                    "No microstructural covariates available - skipping microstructural correlation matrix"
                )

            # 2. Neural covariates correlation (only if we have neural covariates)
            if len(neural_covariates_names) > 0:
                neural_with_target = [target_col_name] + neural_covariates_names
                neural_data = correlation_data[neural_with_target]
                self.neural_covariates_correlation = neural_data.corr(method="pearson")
                self.logger.info(
                    f"Computed neural covariates correlation matrix with shape: {self.neural_covariates_correlation.shape}."
                )
                self.logger.info(
                    "The correlation matrix can be accessed via the 'neural_covariates_correlation' attribute."
                )
            else:
                self.logger.info(
                    "No neural covariates available - skipping neural correlation matrix"
                )

            # 3. Cross-correlation between neural and microstructural variables (only if BOTH exist)
            if (
                len(neural_covariates_names) > 0
                and len(microstructural_covariates_names) > 0
            ):
                cross_correlation_data = correlation_data[
                    neural_covariates_names + microstructural_covariates_names
                ]
                full_correlation = cross_correlation_data.corr(method="pearson")
                # Extract cross-correlation submatrix (neural vs microstructural only)
                self.cross_correlation_matrix = full_correlation.loc[
                    neural_covariates_names, microstructural_covariates_names
                ]
                self.logger.info(
                    f"Computed cross-correlation matrix with shape: {self.cross_correlation_matrix.shape}."
                )
                self.logger.info(
                    "The correlation matrix can be accessed via the 'cross_correlation_matrix' attribute."
                )
            else:
                self.logger.info(
                    "Cannot compute cross-correlation matrix - need both neural and microstructural covariates"
                )

        # Save correlation matrices if save_dir provided
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Prepare dictionary for correlation matrices
            correlation_matrices = dict()

            if self.parsimonious_mode:
                correlation_matrices[
                    "all_covariates_correlation"
                ] = self.neural_covariates_correlation.values
                correlation_matrices[
                    "all_covariates_columns"
                ] = self.neural_covariates_correlation.columns.tolist()
                correlation_matrices[
                    "all_covariates_index"
                ] = self.neural_covariates_correlation.index.tolist()

            else:

                # Add neural covariates correlation matrix if not None
                if self.neural_covariates_correlation is not None:
                    correlation_matrices[
                        "neural_covariates_correlation"
                    ] = self.neural_covariates_correlation.values
                    correlation_matrices[
                        "neural_covariates_columns"
                    ] = self.neural_covariates_correlation.columns.tolist()
                    correlation_matrices[
                        "neural_covariates_index"
                    ] = self.neural_covariates_correlation.index.tolist()

                # Add manually extracted covariates correlation matrix if not None
                if self.manually_extracted_covariates_correlation is not None:
                    correlation_matrices[
                        "microstructural_covariates_correlation"
                    ] = self.manually_extracted_covariates_correlation.values
                    correlation_matrices[
                        "microstructural_covariates_columns"
                    ] = self.manually_extracted_covariates_correlation.columns.tolist()
                    correlation_matrices[
                        "microstructural_covariates_index"
                    ] = self.manually_extracted_covariates_correlation.index.tolist()

                # Add cross correlation matrix if not None
                if self.cross_correlation_matrix is not None:
                    correlation_matrices[
                        "cross_correlation_matrix"
                    ] = self.cross_correlation_matrix.values
                    correlation_matrices[
                        "cross_correlation_columns"
                    ] = self.cross_correlation_matrix.columns.tolist()
                    correlation_matrices[
                        "cross_correlation_index"
                    ] = self.cross_correlation_matrix.index.tolist()

            # Save all correlation matrices to npz file
            if correlation_matrices:
                correlation_path = save_path / "corr_matrices.npz"
                np.savez_compressed(correlation_path, **correlation_matrices)
                self.logger.info(f"Saved correlation matrices to {correlation_path}")

    def _create_correlation_heatmap(
        self, correlation_matrix: pd.DataFrame, main_title: str, save_path: Path
    ) -> None:
        """Create and save correlation heatmap.

        Parameters
        ----------
        correlation_matrix : pd.DataFrame
            Correlation matrix to visualize.
        main_title : str
            Title for the heatmap.
        save_path : Path
            Path to save the plot.
        """

        self.logger.info(f"Creating correlation heatmap with title {main_title}")
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            correlation_matrix, annot=False, cmap="coolwarm", ax=ax, vmin=-1, vmax=1
        )

        # Set tick positions and labels for the combined matrix
        ax.set_xticks(np.arange(len(correlation_matrix.columns.tolist())) + 0.5)
        ax.set_xticklabels(
            correlation_matrix.columns.tolist(), rotation=90, ha="center", fontsize=6
        )
        ax.set_yticks(np.arange(len(correlation_matrix.index.tolist())) + 0.5)
        ax.set_yticklabels(correlation_matrix.index.tolist(), rotation=0, fontsize=6)

        ax.set_title(main_title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Saved correlation heatmap to {save_path}")

    def _save_correlation_heatmaps(self, save_dir: Path) -> None:
        """Save correlation heatmaps to disk. This function will handle parsimonious mode and full mode automatically.
        
        Parameters
        ----------
        save_dir : Path
            Directory to save plots.
        """

        if self.parsimonious_mode:
            if self.neural_covariates_correlation is not None:
                self._create_correlation_heatmap(
                    self.neural_covariates_correlation,
                    "Correlation Heatmap Full Covariate Set",
                    save_dir / "corr_heatmap.png",
                )
            else:
                self.logger.warning(
                    "No correlation matrix available in parsimonious mode - skipping heatmap creation"
                )
        else:
            # Full mode: create heatmaps only for matrices that exist
            if self.neural_covariates_correlation is not None:
                self._create_correlation_heatmap(
                    self.neural_covariates_correlation,
                    "Correlation Heatmap Neural Covariates",
                    save_dir / "corr_heatmap_neural.png",
                )
            else:
                self.logger.info(
                    "No neural covariates correlation matrix - skipping neural heatmap"
                )

            if self.manually_extracted_covariates_correlation is not None:
                self._create_correlation_heatmap(
                    self.manually_extracted_covariates_correlation,
                    "Correlation Heatmap Microstructural Measures",
                    save_dir / "corr_heatmap_microstructural.png",
                )
            else:
                self.logger.info(
                    "No microstructural covariates correlation matrix - skipping microstructural heatmap"
                )

            if self.cross_correlation_matrix is not None:
                self._create_correlation_heatmap(
                    self.cross_correlation_matrix,
                    "Cross Correlation Heatmap: Neural Covariates and Microstructural Measures",
                    save_dir / "corr_heatmap_cross_covariates.png",
                )
            else:
                self.logger.info(
                    "No cross-correlation matrix - skipping cross-correlation heatmap"
                )

    def save_training_history(self, save_dir: str) -> None:
        """Save training history to parquet file.
        
        Parameters
        ----------
        save_dir : str
            Directory to save the training history file.
        """
        if not hasattr(self, "training_history_df") or self.training_history_df is None:
            self.logger.error("No training history available for saving.")
            raise ValueError("No training history available for saving.")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save training history as parquet file
        train_log_path = save_path / "train_log.parquet"
        self.training_history_df.to_parquet(train_log_path, index=False)
        self.logger.info(f"Saved training history to {train_log_path}")

    def save_model_weights(self, save_dir: str) -> None:
        """Save the best model weights to disk.
        
        Parameters
        ----------
        save_dir : str
            Directory to save the model weights.
        """
        if self.model is None:
            self.logger.error("No model available for saving.")
            return

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state dict as PyTorch checkpoint
        model_path = save_path / "best.pt"
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Saved model weights in .pt format to {model_path}")

        # Also save model architecture info and training metadata
        metadata = {
            "model_class": self.model.__class__.__name__,
            "device": str(self.device),
            "delta_adapter": self.delta_adapter,
            "delta_sample": self.delta_sample,
            "parsimonious_mode": self.parsimonious_mode,
        }

        # Add training history info if available
        if (
            hasattr(self, "training_history_df")
            and self.training_history_df is not None
            and len(self.training_history_df) > 0
        ):
            metadata["final_train_loss"] = float(
                self.training_history_df["train_loss"].iloc[-1]
            )
            metadata["final_validation_loss"] = float(
                self.training_history_df["validation_loss"].iloc[-1]
            )
            metadata["best_validation_loss"] = float(
                self.training_history_df["validation_loss"].min()
            )
            metadata["epochs_trained"] = len(self.training_history_df)
        else:
            # No training occurred (epochs=0)
            metadata["final_train_loss"] = None
            metadata["final_validation_loss"] = None
            metadata["best_validation_loss"] = None
            metadata["epochs_trained"] = 0

        # Save metadata as JSON
        metadata_path = save_path / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved model metadata to {metadata_path}")

    def save_nn_weights(self, save_dir: str) -> None:  # Checked
        """Save neural network weights in portable numpy format.
        
        Parameters
        ----------
        save_dir : str
            Directory to save the weights.
        """
        if self.model is None:
            print("Warning: No model available to save.")
            return

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Convert PyTorch state dict to numpy arrays
        state_dict = self.model.state_dict()
        numpy_weights = {}

        for name, tensor in state_dict.items():
            # Convert to numpy and store
            numpy_weights[name] = tensor.cpu().detach().numpy()

        # Save as compressed numpy format
        weights_path = save_path / "nn_weights.npz"
        np.savez_compressed(weights_path, **numpy_weights)
        self.logger.info(
            f"Saved neural network weights in .npz format to {weights_path}"
        )

    def save_test_results(self, save_dir: str, test_loss: float) -> None:
        """Save test results for out-of-sample evaluation.
        
        Parameters
        ----------
        save_dir : str
            Directory to save test results.
        test_loss : float
            Test loss value.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Create test results metadata
        test_results = {
            "model_class": self.model.__class__.__name__,
            "parsimonious_mode": self.parsimonious_mode,
            "test_loss": test_loss,
        }

        # Save as JSON
        test_path = save_path / "test_os.json"
        with open(test_path, "w") as f:
            json.dump(test_results, f, indent=2)
        self.logger.info(f"Saved test results to {test_path}")

    def evaluate_out_of_sample(
        self,
        event_driven_data: pd.DataFrame,
        criterion: nn.modules.loss = nn.MSELoss(reduction="sum"),
        batch_size: int = 32,
        save_dir: Optional[str] = None,
    ) -> float:
        """Evaluate the trained model on out-of-sample data.
        
        Parameters
        ----------
        event_driven_data : pd.DataFrame
            Event-driven data for evaluation.
        criterion : nn.modules.loss, optional
            Loss function. MSELoss by default.
        batch_size : int, optional
            Batch size for evaluation. 32 by default.
        save_dir : str, optional
            Directory to save test results. None by default.
            
        Returns
        -------
        float
            Test loss.
        """
        if self.model is None:
            raise ValueError(
                "No model available for evaluation. Run global_update first."
            )

        # Skip evaluation if running in manual_covariates_only mode
        if self.manual_covariates_only:
            self.logger.info("=" * 80)
            self.logger.info(
                "SKIPPING NEURAL NETWORK EVALUATION (manual_covariates_only=True)"
            )
            self.logger.info(
                "Returning test_loss = -1.0 to indicate no neural evaluation occurred"
            )
            self.logger.info("=" * 80)

            # Save test results indicating no evaluation occurred
            if save_dir:
                test_results = {
                    "model_class": self.model.__class__.__name__,
                    "parsimonious_mode": self.parsimonious_mode,
                    "manual_covariates_only": True,
                    "neural_network_evaluated": False,
                    "test_loss": -1.0,
                    "note": "Neural network evaluation skipped in manual_covariates_only mode",
                }

                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                test_path = save_path / "test_os.json"
                with open(test_path, "w") as f:
                    json.dump(test_results, f, indent=2)
                self.logger.info(f"Saved test results (no evaluation) to {test_path}")

            return -1.0  # Special value indicating no neural evaluation occurred

        self.logger.info("Starting neural network evaluation in local adapt mode")

        # Validate that preprocessing components are available
        if not self.has_preprocessing_components():
            raise ValueError(
                "Preprocessing components (scaler and bounds) are required for dataset creation. "
                "Use set_preprocessing_components() to provide them or initialize backbone with these components."
            )

        # Create dataset using the CNN-LSTM preprocessor
        # Use train=True to filter out NaN targets for proper loss evaluation
        test_dataset = get_dataset_cnn_lstm(
            event_driven_data,
            delta=self.delta_sample,
            pretrained_scaler=self.pretrained_scaler,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            train=True,  # Filter NaN targets for evaluation
            past_returns=self.past_returns,
        )

        # Create data loader
        if self.device != torch.device("cpu"):
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=False,
            )
        else:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.logger.info(f"Evaluating on {len(test_dataset)} events")

        # Evaluate using existing _one_epoch method (without optimizer for evaluation)
        test_loss = self._one_epoch(test_loader, criterion, optimizer=None)

        self.logger.info(
            f"Out-of-sample evaluation complete. Test loss: {test_loss:.6f}"
        )

        # Save test results if save_dir provided
        if save_dir:
            self.save_test_results(save_dir, test_loss)

        return test_loss

    def global_update(
        self,
        event_driven_backbone_time_bar_data: pd.DataFrame,
        physical_time_driven_backbone_time_bar_data: pd.DataFrame,
        criterion: nn.modules.loss = nn.MSELoss(reduction="sum"),
        optimizer: Optional[torch.optim.Optimizer] = None,
        epochs: int = 10,
        validation_proportion: float = 0.1,
        buffer_proportion: float = 0.05,
        batch_size: int = 32,
        adf_rejection_rate_threshold: float = 0.8,
        kpss_verification_threshold: float = 0.2,
        na_verification_threshold: float = 0.1,
        kpss_drop: bool = True,
        correlation_threshold: float = 0.95,
        target_col_name: str = "micro_return",
        save_dir: str = None,
        session_start: float = 0.0,
    ) -> float:
        """Perform global update of the model with time-based train/validation split.
        
        Creates a 90/10% (configurable) partition of the backbone interval event driven data.
        The validation interval is selected uniformly at random from the start of the bar to 
        train_proportion of the bar. Buffer zones are applied to avoid data leakage.
        
        Parameters
        ----------
        event_driven_backbone_time_bar_data : pd.DataFrame
            Event-driven data for the current time bar.
        physical_time_driven_backbone_time_bar_data : pd.DataFrame
            Physical time-driven data for stationarity analysis.
        criterion : nn.modules.loss, optional
            Loss function. Defaults to MSELoss with sum reduction.
        optimizer : torch.optim.Optimizer, optional
            Optimizer for the model. Defaults to Adam with lr=0.0001.
        epochs : int, optional
            Number of epochs to train. Defaults to 10.
        validation_proportion : float, optional
            Proportion of data for validation. Defaults to 0.1.
        buffer_proportion : float, optional
            Buffer proportion to avoid data leakage. Defaults to 0.05.
        batch_size : int, optional
            Batch size for data loaders. Defaults to 32.
        adf_rejection_rate_threshold : float, optional
            ADF rejection rate threshold for stationarity. Defaults to 0.8.
        kpss_verification_threshold : float, optional
            KPSS verification threshold. Defaults to 0.2.
        na_verification_threshold : float, optional
            NA verification threshold. Defaults to 0.1.
        kpss_drop : bool, optional
            Whether to use strict KPSS dropping. Defaults to True.
        correlation_threshold : float, optional
            Threshold for dropping highly correlated variables. Defaults to 0.95.
        target_col_name : str, optional
            Name of the target variable column. Defaults to 'micro_return'.
        save_dir : str, optional
            Directory to save results of the global_update
            
        Returns
        -------
        float
            Best validation loss achieved during training.
        """

        if save_dir is None:
            self.logger.error(
                "No save directory specified. Please specify a save directory through the 'save_dir' argument."
            )
            raise ValueError(
                "No save directory specified. Please specify a save directory through the 'save_dir' argument."
            )

        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("Running Backbone module global_update")
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)

        train_proportion = 1.0 - validation_proportion

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        # Ensure we have timestamp column for partitioning
        if "timestamp" not in event_driven_backbone_time_bar_data.columns:
            self.logger.error(
                "event_driven_backbone_time_bar_data must contain 'timestamp' column"
            )
            raise ValueError(
                "event_driven_backbone_time_bar_data must contain 'timestamp' column"
            )

        # Sort data by timestamp to ensure proper ordering
        data = event_driven_backbone_time_bar_data.sort_values("timestamp").reset_index(
            drop=True
        )

        # Get time range of the data
        start_time = data["timestamp"].min()
        end_time = data["timestamp"].max()
        time_span = end_time - start_time

        # Calculate validation interval length
        validation_length = validation_proportion * time_span
        buffer_length = buffer_proportion * time_span

        # Calculate the maximum start position for validation interval
        # (train_proportion of the time span minus validation length)
        max_validation_start = end_time - validation_length

        # Randomly select validation interval start time
        validation_start = np.random.uniform(start_time, max_validation_start)
        validation_end = validation_start + validation_length

        # Define buffer zones around validation interval
        validation_buffer_start = validation_start - buffer_length
        validation_buffer_end = validation_end + buffer_length

        # Create masks for train and validation data
        # Training data: everything except validation interval + buffers
        train_mask = ~(
            (data["timestamp"] >= validation_buffer_start)
            & (data["timestamp"] <= validation_buffer_end)
        )

        # Validation data: only the validation interval (no buffers)
        validation_mask = (data["timestamp"] >= validation_start) & (
            data["timestamp"] <= validation_end
        )

        # Split the data
        train_data = data[train_mask].reset_index(drop=True)
        validation_data = data[validation_mask].reset_index(drop=True)

        # Check if we have enough data
        if len(train_data) == 0:
            self.logger.error("No training data available after partitioning")
            raise ValueError("No training data available after partitioning")
        if len(validation_data) == 0:
            self.logger.error("No validation data available after partitioning")
            raise ValueError("No validation data available after partitioning")

        self.logger.info(
            f"Training with {len(train_data)} samples, validating with {len(validation_data)} samples"
        )
        self.logger.info(
            f"Backbone bar events interval: [{start_time:.2f}, {end_time:.2f}]"
        )
        self.logger.info(
            f"Validation interval: [{validation_start:.2f}, {validation_end:.2f}]"
        )
        self.logger.info(
            f"Buffer intervals: [{validation_buffer_start:.2f}, {validation_start:.2f}] and [{validation_end:.2f}, {validation_buffer_end:.2f}]"
        )

        # Validate that preprocessing components are available
        if not self.has_preprocessing_components():
            raise ValueError(
                "Preprocessing components (scaler and bounds) are required for dataset creation. "
                "Use set_preprocessing_components() to provide them or initialize backbone with these components."
            )

        # Create datasets using the CNN-LSTM preprocessor
        train_dataset = get_dataset_cnn_lstm(
            train_data,
            delta=self.delta_sample,
            pretrained_scaler=self.pretrained_scaler,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            train=True,
            past_returns=self.past_returns,
        )

        validation_dataset = get_dataset_cnn_lstm(
            validation_data,
            delta=self.delta_sample,
            pretrained_scaler=self.pretrained_scaler,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            train=True,
            past_returns=self.past_returns,
        )

        # Create data loaders

        # Train
        if self.device != torch.device("cpu"):
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False
            )

        # Validation
        if self.device != torch.device("cpu"):
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
                shuffle=False,
            )
        else:
            validation_loader = DataLoader(
                validation_dataset, batch_size=batch_size, shuffle=False
            )

        # Train the model using existing batch_gradient_descent method (skip if manual_covariates_only)
        if self.manual_covariates_only:
            self.logger.info("=" * 80)
            self.logger.info(
                "SKIPPING NEURAL NETWORK TRAINING (manual_covariates_only=True)"
            )
            self.logger.info("=" * 80)
            best_validation_loss = 0.0  # No training occurred

            # Create architecture directory but don't save model artifacts
            if save_dir:
                architecture_dir = Path(save_dir) / "architecture"
                architecture_dir.mkdir(parents=True, exist_ok=True)

                # Save minimal metadata indicating no training occurred
                metadata = {
                    "model_class": self.model.__class__.__name__,
                    "device": str(self.device),
                    "delta_adapter": self.delta_adapter,
                    "delta_sample": self.delta_sample,
                    "parsimonious_mode": self.parsimonious_mode,
                    "manual_covariates_only": self.manual_covariates_only,
                    "neural_network_trained": False,
                    "epochs_trained": 0,
                }

                metadata_path = architecture_dir / "model_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                self.logger.info(
                    f"Saved model metadata (no training) to {metadata_path}"
                )
        else:
            checkpoint_dir = None
            if save_dir:
                checkpoint_dir = str(Path(save_dir) / "architecture" / "checkpoints")

            best_validation_loss = self.batch_gradient_descent(
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=epochs,
                validation_loader=validation_loader,
                save_checkpoints=True,
                checkpoint_dir=checkpoint_dir,
            )

            architecture_dir = None
            if save_dir:
                architecture_dir = Path(save_dir) / "architecture"
                self.save_training_history(save_dir=str(architecture_dir))
                self.save_model_weights(save_dir=str(architecture_dir))
                self.save_nn_weights(save_dir=str(architecture_dir))

        # ========================================================================
        # STEP 1: Extract neural network covariates FIRST (before stationarity tests)
        # ========================================================================

        # Extract neural network covariates from the trained model (skip if manual_covariates_only)
        if self.manual_covariates_only:
            self.logger.info("=" * 80)
            self.logger.info(
                "SKIPPING NEURAL COVARIATE EXTRACTION (manual_covariates_only=True)"
            )
            self.logger.info("=" * 80)
            # Use the original physical data without neural covariates
            augmented_physical_data = physical_time_driven_backbone_time_bar_data.copy()
            neural_covariate_columns = []  # No neural covariates
        elif (
            physical_time_driven_backbone_time_bar_data is not None
            and not physical_time_driven_backbone_time_bar_data.empty
        ):

            self.logger.info("=" * 80)
            self.logger.info("EXTRACTING NEURAL COVARIATES")
            self.logger.info("=" * 80)

            # Extract features from event-driven data using the trained model
            neural_covariates = self.extract_neural_covariates(
                event_driven_data=data,
                physical_time_data=physical_time_driven_backbone_time_bar_data,
            )

            # Get neural covariate column names
            neural_covariate_columns = [
                col for col in neural_covariates.columns if col != "timestamp"
            ]

            # Verify neural covariates extraction
            if neural_covariates is not None and len(neural_covariates) > 0:
                # Set timestamp columns as float to avoid incompatible types during merging
                neural_covariates["timestamp"] = neural_covariates["timestamp"].astype(
                    float
                )

                if self.neural_covariates_only:
                    # Neural covariates only mode - use neural covariates but include target variable
                    self.logger.info(
                        "Neural covariates only mode - using neural covariates without manual features"
                    )
                    # Need to merge target variable from physical data for stationarity testing
                    physical_time_driven_backbone_time_bar_data[
                        "timestamp"
                    ] = physical_time_driven_backbone_time_bar_data["timestamp"].astype(
                        float
                    )
                    augmented_physical_data = pd.merge_asof(
                        neural_covariates,
                        physical_time_driven_backbone_time_bar_data[
                            ["timestamp", target_col_name]
                        ],
                        on="timestamp",
                        direction="nearest",
                        tolerance=1e-6,
                    )
                    self.logger.info(
                        f"Merged target variable {target_col_name} with neural covariates"
                    )
                else:
                    # Standard mode - merge neural covariates with physical data
                    physical_time_driven_backbone_time_bar_data[
                        "timestamp"
                    ] = physical_time_driven_backbone_time_bar_data["timestamp"].astype(
                        float
                    )
                    self.logger.info(
                        "Merging neural covariates with physical time data"
                    )
                    augmented_physical_data = pd.merge_asof(
                        physical_time_driven_backbone_time_bar_data,
                        neural_covariates,
                        on="timestamp",
                        direction="nearest",
                        tolerance=1e-6,
                    )

                self.logger.info(
                    f"Augmented data: {augmented_physical_data.shape[0]:,} rows, {augmented_physical_data.shape[1]:,} columns"
                )

            else:
                self.logger.error("No neural covariates extracted from the model")
                raise ValueError("There was an error extracting neural covariates.")
        else:
            self.logger.error(
                "No physical time-driven data provided for neural covariate extraction"
            )
            raise ValueError(
                "No physical time-driven data provided for neural covariate extraction"
            )

        # ========================================================================
        # STEP 2: Run stationarity tests on AUGMENTED data (includes neural covariates)
        # ========================================================================

        # Apply stationarity analysis to the augmented physical time-driven data
        if augmented_physical_data is not None and not augmented_physical_data.empty:

            # Construct save directory for stationarity results
            stationarity_save_dir = None
            if save_dir:
                stationarity_save_dir = Path(save_dir) / "stationarity"

            self.logger.info("=" * 80)
            self.logger.info("RUNNING STATIONARITY TESTS ON AUGMENTED DATA")
            if not self.manual_covariates_only and neural_covariate_columns:
                self.logger.info("Testing BOTH manual covariates AND neural covariates")
            else:
                self.logger.info("Testing ONLY manual covariates")
            self.logger.info("=" * 80)

            # Run stationarity tests on the augmented physical time-driven data
            # This now includes both manual and neural covariates
            stationary_vars = self.select_stationary_variables(
                data=augmented_physical_data,
                adf_rejection_rate_threshold=adf_rejection_rate_threshold,
                kpss_verification_threshold=kpss_verification_threshold,
                na_verification_threshold=na_verification_threshold,
                kpss_drop=kpss_drop,
                target_col_name=target_col_name,
                save_dir=stationarity_save_dir,
                session_start=session_start,
            )

        else:
            self.logger.error("No data available for stationarity analysis")
            raise ValueError("No data available for stationarity analysis")

        if self.full_covariate_list and len(self.full_covariate_list) > 1:
            # Separate neural covariates from manual covariates
            # stationary_vars contains all variables that passed stationarity tests (both neural and manual)

            if self.neural_covariates_only:
                # In neural-only mode, only neural covariates (no manual covariates)
                stationary_neural_covariates = [
                    col for col in stationary_vars if col in neural_covariate_columns
                ]
                stationary_manual_covariates = []
            elif self.manual_covariates_only:
                # In manual-only mode, only manual covariates (no neural covariates)
                stationary_neural_covariates = []
                stationary_manual_covariates = [
                    col
                    for col in stationary_vars
                    if col not in {"timestamp", target_col_name}
                ]
            else:
                # In standard mode, separate neural and manual covariates
                # Neural covariates: those that passed stationarity AND are in neural_covariate_columns
                stationary_neural_covariates = [
                    col for col in stationary_vars if col in neural_covariate_columns
                ]
                # Manual covariates: those that passed stationarity AND are NOT in neural_covariate_columns
                stationary_manual_covariates = [
                    col
                    for col in stationary_vars
                    if col not in {"timestamp", target_col_name}
                    and col not in neural_covariate_columns
                ]

            save_dir_correlation = None
            if save_dir:
                save_dir_correlation = Path(save_dir) / "correlation"

                self._compute_correlation_matrices(
                    augmented_physical_data,
                    stationary_neural_covariates,
                    stationary_manual_covariates,
                    target_col_name,
                    save_dir_correlation,
                )
                self._save_correlation_heatmaps(save_dir_correlation)

        # Drop highly correlated variables from the augmented dataset
        if self.full_covariate_list and len(self.full_covariate_list) > 1:

            self.drop_highly_correlated_variables(
                data=augmented_physical_data, threshold=correlation_threshold
            )

        elif len(self.full_covariate_list) == 1:
            # Handle case with exactly one covariate - continue execution but warn
            if self.neural_covariates_only:
                self.logger.info(
                    f"Neural covariates only mode: Using single neural covariate '{self.full_covariate_list[0]}'"
                )
            elif self.manual_covariates_only:
                self.logger.info(
                    f"Manual covariates only mode: Using single manual covariate '{self.full_covariate_list[0]}'"
                )
            else:
                self.logger.warning(
                    f"Only one covariate available: '{self.full_covariate_list[0]}'. Consider adjusting selection criteria."
                )
        else:
            self.logger.warning(
                "No covariates available for correlation analysis. Continuing with empty covariate set."
            )

        # Final cleanup
        if self.full_covariate_list:
            self.full_covariate_list = [
                var
                for var in self.full_covariate_list
                if var not in {"timestamp", target_col_name}
            ]

        # Save final covariate list if save_dir provided
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save covariate list as JSON
            covariate_list_path = save_path / "final_covariate_list.json"
            with open(covariate_list_path, "w") as f:
                json.dump(
                    {"final_covariate_list": self.full_covariate_list}, f, indent=2
                )
            self.logger.info(
                f"Saved final covariate list ({len(self.full_covariate_list)} variables) to {covariate_list_path}"
            )

        self.logger.info("=" * 80)
        self.logger.info("=" * 80)
        self.logger.info("Backbone module global_update completed")
        self.logger.info("=" * 80)
        self.logger.info("=" * 80)

        return {
            "backbone_model": self.model,
            "full_covariate_set": self.full_covariate_list,
        }
