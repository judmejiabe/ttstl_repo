import warnings
from typing import Optional
import traceback

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import numpy as np

from ttstl.utils.data_preprocessor import get_dataset_cnn_lstm

urca = importr('urca')
fUnitRoots = importr('fUnitRoots')

pandas2ri.activate()

class Backbone():
    def __init__(self, model : nn.Module, delta_adapter : float = 0.0) -> None:
        """Initialize the Backbone with a model and delta adapter.
        Args:
            model (nn.Module): The neural network model to be used.
            delta_adapter (float, optional): Adapter time bar length in the same units of time as the data.
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.delta_adapter = delta_adapter
        self.full_covariate_list = None
        self.dropped_pairs = dict()

    def _one_epoch(self, 
                   train_loader : DataLoader,
                   criterion : nn.modules.loss,
                   optimizer : Optional[torch.optim.Optimizer] = None) -> float:
        """Performs batch gradient descent on the provided dataset.
        Args:
            train_loader (DataLoader): DataLoader for the training data.
            criterion (nn.modules.loss, optional): Loss function.
            optimizer (torch.optim.Optimizer, optional): Optimizer for the model.
            batch_size (int, optional): Size of each batch. Defaults to 32.
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
        validation_loader: Optional[DataLoader] = None) -> float:
        """Train the model for one epoch.
        Args:
            train_loader (DataLoader): DataLoader for the training data.
            criterion (nn.modules.loss): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for the model.
        Returns:
            float: Average loss over the epoch.
        """
        print("Starting training...")
        validation_loader = validation_loader or train_loader
        train_losses = []
        validation_losses = []
        best_model = self.model
        best_validation_loss = float('inf')
        for epoch in range(epochs):

            print(f"Epoch {epoch + 1}/{epochs}")

            self._one_epoch(train_loader, criterion, optimizer)
            this_train_loss = self._one_epoch(train_loader, criterion)
            this_validation_loss = self._one_epoch(validation_loader, 
                                                   criterion)

            validation_losses.append(this_validation_loss)
            train_losses.append(this_train_loss)

            if this_validation_loss < best_validation_loss:
                best_validation_loss = this_validation_loss
                best_model = self.model

            print(f"Train Loss: {this_train_loss:.4f}, Validation Loss: {this_validation_loss:.4f}")

        print(f"Training complete. Best validation loss: {best_validation_loss:.4f}")

        self.model = best_model
        return best_validation_loss
    
    def _run_stationarity_tests(self,
                                series: pd.Series) -> dict:
        """Conducts the Augmented Dickey-Fuller (ADF), Philips Perron (PP), and
        Kwiatkowski-Phillips-Schmidt-Shin (KPSS) stationarity tests on a given 
        Pandas series. The results are returned in dictionary with keys 'ADF',
        'PP', 'KPSS', and 'NA' where the value corresponds to 1 if rejection
        and zero otherwise. If any of the tests fail, the 'NA' key will be set
        to 1.
        Args:
            series (pd.Series): Pandas Series to test for stationarity.
        Returns:
            dict: Dictionary with keys 'ADF', 'PP', 'KPSS', and 'NA' with 
                  values indicating test results.
        """

        alpha = 0.05
        series_r = pandas2ri.py2rpy(series)
        try:
            adf = fUnitRoots.adfTest(series_r, lags = 10, type = 'c')
            adf_rej = adf.slots['test'].rx2('p.value')[0]  < alpha
            pp = urca.ur_pp(series_r, type = 'Z-tau', model = 'constant', lags = 'long')
            pp_rej = pp.slots['teststat'][0] < pp.slots['cval'][0, 1]
            kpss = urca.ur_kpss(series_r, type = 'mu', lags = 'long')
            kpss_rej = kpss.slots['teststat'][0] > kpss.slots['cval'][0, 1]
            return {'ADF' : int(adf_rej),
                    'PP' : int(pp_rej),
                    'KPSS' : int(kpss_rej),
                    'NA' : 0} 
        except Exception as e:
            print(f"Error running stationarity tests: {e}")
            return {'ADF': 0, 'PP': 0, 'KPSS': 0, 'NA': 1}

    def _run_multiple_stationarity_tests(self,
                           data : pd.DataFrame) -> None:

        """Perform multiple stationarity tests on the provided data. 
        The testing frequency is determined by the delta_adapter attribute.
        Args:
            data (pd.DataFrame): Dataframe containing Phyisical Time Driven data.
        Returns:
            pd.DataFrame: DataFrame containing the results of the stationarity tests.
        """
        
        columns = data.columns.tolist()
        if 'timestamp' in columns:
            columns.remove('timestamp')


        try:
            data['adapter_bar'] = (
            np.floor((data['timestamp'] + 1e-3) / self.delta_adapter)
            .astype(int)
            )
        except KeyError:
            data['adapter_bar'] = (
            np.floor((data.index.to_series() + 1e-3) / self.delta_adapter)
            .astype(int)
            )

        records = []

        for ell, group in data.groupby('adapter_bar'):
            for column in columns:
                series = group[column]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                tests = self._run_stationarity_tests(series)
                records.append({
                    'adapter_bar': ell,
                    'variable' : column,
                    'ADF': tests['ADF'],
                    'PP': tests['PP'],
                    'KPSS': tests['KPSS'],
                    'NA': tests['NA']
                })
        
        rejections_df = pd.DataFrame.from_records(records)

        return rejections_df

    def select_stationary_variables(self,
                                     data: pd.DataFrame,
                                     adf_rejection_rate_threshold : float = 0.8,
                                     kpss_verification_threshold : float = 0.2,
                                     na_verification_threshold : float = 0.1,
                                     kpss_drop : bool = True) -> pd.DataFrame:
        """Select stationary variables from the DataFrame.
        Args:
            data (pd.DataFrame): DataFrame containing the time series data.
        Returns:
            pd.DataFrame: DataFrame with only stationary variables.
        """
        
        rejections_df = self._run_multiple_stationarity_tests(data)
        grouped = rejections_df.groupby('variable')[['ADF', 'KPSS', 'NA']]\
            .mean()

        # Define base condition for stationarity
        adf_stationary = grouped['ADF'] > adf_rejection_rate_threshold
        high_na = grouped['NA'] > na_verification_threshold

        if kpss_drop:
            # Strict stationarity: both ADF and KPSS must agree
            kpss_stationary = grouped['KPSS'] < kpss_verification_threshold
            stationary_variables = grouped.index[adf_stationary & \
                                                 kpss_stationary].tolist()
            warning_variables = []
        else:
            # Lenient stationarity: only ADF required
            stationary_variables = grouped.index[adf_stationary].tolist()
            # Variables that pass ADF but fail KPSS need verification
            kpss_warning = grouped['KPSS'] > kpss_verification_threshold
            warning_variables = grouped.index[adf_stationary & \
                                              kpss_warning].tolist()

        # Variables with high NA rates (among those passing ADF)
        high_na_vars = grouped.index[adf_stationary & \
                                     high_na].tolist()
        
        

        self.full_covariate_list = [var for var in stationary_variables \
                                    if var not in {'timestamp', 'micro_return'}]

        # Issue warnings
        if warning_variables:
            warnings.warn(f"Variables meeting ADF threshold but requiring verification for high KPSS rejections: {warning_variables}", UserWarning)

        if high_na_vars:
            warnings.warn(f"Variables passing stationarity criteria but with high NA rates:: {high_na_vars}", UserWarning)

        return stationary_variables
    
    def _add_covariate_set(self, 
                          extra_covariate_set: list) -> None:
        """Add extra covariate set to the full variable set.
        Args:
            extra_covariate_set (list): List of additional covariates to add.
        """
        if self.full_covariate_list is None:
            self.full_covariate_list = extra_covariate_set
        else:
            self.full_covariate_list += extra_covariate_set
            # Remove duplicates
            seen = set()
            self.full_covariate_list = [x for x in self.full_covariate_list if not (x in seen or seen.add(x))]

    def drop_highly_correlated_variables(self, 
                                    data: pd.DataFrame, 
                                    threshold : float = 0.95) -> pd.DataFrame:
        """Perform covariance matrix analysis on the DataFrame.
        Args:
            data (pd.DataFrame): DataFrame containing the time series data.
        Returns:
            pd.DataFrame: DataFrame with covariance matrix analysis results.
        """

        cov_matrix = data[self.full_covariate_list].corr(method='pearson')
        # Determine highly correlated pairs
        highly_correlated_pairs = cov_matrix.abs() > threshold
        adjacency_dict = {}
        for var in highly_correlated_pairs.columns:
            if sum(highly_correlated_pairs[var]) > 1:
                adjacency_dict[var] = set(highly_correlated_pairs.\
                                          index[highly_correlated_pairs[var]]\
                                            .tolist()) - {var}

        shuffled_variables = np.random.permutation(list(adjacency_dict.\
                                                        keys())).tolist()
        dropped_vars = set()

        for var in shuffled_variables:
            if var not in adjacency_dict:
                continue
            for neighbor in list(adjacency_dict[var]):
                self.dropped_pairs[var] = neighbor
                self.dropped_pairs[neighbor] = var
                adjacency_dict[neighbor].remove(var)
                if not adjacency_dict[neighbor]:
                    del adjacency_dict[neighbor]
            del adjacency_dict[var]
            dropped_vars.add(var)

        self.full_covariate_list = [var for var in self.full_covariate_list if var not in dropped_vars]

        return cov_matrix
    
    def _extract_neural_covariates(self,
                                   event_driven_data: pd.DataFrame,
                                   physical_time_data: pd.DataFrame,
                                   delta: float = 100.0,
                                   quantile: Optional[float] = None,
                                   parsimonious_mode: bool = False) -> dict:
        """Extract neural network covariates from event-driven data for physical time-driven data.
        
        For each timestamp in the physical time-driven data, find the last row in the event-driven
        data with a timestamp smaller than the physical timestamp, and extract neural network
        features (hidden activations or forward pass output) for that row.
        
        Args:
            event_driven_data (pd.DataFrame): Event-driven data with timestamps.
            physical_time_data (pd.DataFrame): Physical time-driven data with timestamps.
            delta (float): Sampling time bar length
            quantile (Optional[float]): Winsorization quantile.
            parsimonious_mode (bool): If True, use forward pass output; if False, use hidden activations.
            
        Returns:
            dict: Dictionary with covariate names as keys and pandas Series as values.
        """
        try:
            print(f"Extracting neural covariates in {'parsimonious' if parsimonious_mode else 'full'} mode...")
            
            # Create dataset from event-driven data (using train=False to keep all data)
            event_dataset = get_dataset_cnn_lstm(
                event_driven_data, delta=delta, quantile=quantile,
                train=False)
            
            # Create a DataLoader for batch processing with larger batch size for efficiency
            batch_size = min(256, len(event_dataset))  # Use larger batch size
            if self.device != torch.device('cpu'):
                event_loader = DataLoader(
                    event_dataset, batch_size=batch_size, 
                    num_workers=4, pin_memory=True, shuffle=False
                )
            else:
                event_loader = DataLoader(event_dataset, batch_size=batch_size, shuffle=False)
            
            print(f"Processing {len(event_dataset)} samples in batches of {batch_size}")
            
            # Extract features using the trained model
            self.model.eval()
            representations = []
            
            with torch.no_grad():
                for inputs, targets in event_loader:
                    inputs = inputs.to(self.device, dtype=torch.float32)
                    
                    # Extract features based on mode
                    if parsimonious_mode:
                        # Use forward pass output (1-dimensional)
                        # Shape: (batch_size, 1)
                        rep = self.model(inputs)  
                    else:
                        # Use hidden activations (higher dimensional)
                        # Shape: (batch_size, 64)
                        rep = self.model.hidden_activations(inputs)  

                    representations.append(rep.cpu())

            # Concatenate all predictions and convert to numpy
            representations = torch.cat(representations).numpy()
            
            print(f"Extracted features shape: {representations.shape}")

            reps = []

            # Get event-driven timestamps
            event_timestamps = event_driven_data['timestamp'].values
            
            # Get physical time timestamps
            if 'timestamp' in physical_time_data.columns:
                physical_timestamps = physical_time_data['timestamp'].values
            else:
                physical_timestamps = physical_time_data.index.values
            
            # Vectorized searchsorted operation
            indices = np.searchsorted(event_timestamps, physical_timestamps, 
                                      side='right')
            
            # Handle edge cases: set indices to 0 where no prior event exists
            valid_mask = indices > 0
            indices = np.maximum(indices - 1, 0)
            # Vectorized representation extraction
            if representations.ndim == 1:
                reps = representations[indices]
                reps[~valid_mask] = np.nan
            else:
                reps = representations[indices]
                reps[~valid_mask] = np.nan

            # Create a DataFrame with timestamps and representations,
            # set the representations as columns. Name the columns
            # as hidden_activations_i if not parsimonious_mode else forward_pass_i
            covariates_df = pd.DataFrame(
                np.hstack([np.array(physical_timestamps).reshape(-1, 1), reps]),
                columns=None
            )

            # Name the columns according to the model's output layer
            if parsimonious_mode:
                covariates_df.columns = ['timestamp'] + \
                    [f'forward_pass_{i}' for i in range(reps.shape[1])]
            else:
                covariates_df.columns = ['timestamp'] + \
                    [f'hidden_activations_{i}' for i in range(reps.shape[1])]


            print(covariates_df.info()) # Debugging output
            print(covariates_df.head()) # Debugging output

            return covariates_df

            
        except Exception as e:
            print(f"Error extracting neural covariates: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    
    def global_update(self,
                      event_driven_backbone_time_bar_data : pd.DataFrame,
                      physical_time_driven_backbone_time_bar_data : pd.DataFrame,
                      criterion: nn.modules.loss = nn.MSELoss(reduction = 'sum'),
                      optimizer: Optional[torch.optim.Optimizer] = None,
                      epochs: int = 10,
                      validation_proportion: float = 0.1,
                      buffer_proportion: float = 0.05,
                      batch_size: int = 32,
                      delta: float = 100.0,
                      quantile: Optional[float] = 0.005,
                      adf_rejection_rate_threshold: float = 0.8,
                      kpss_verification_threshold: float = 0.2,
                      na_verification_threshold: float = 0.1,
                      kpss_drop: bool = True,
                      parsimonious_mode: bool = False,
                      correlation_threshold: float = 0.95) -> float:
        """Perform global update of the model with time-based train/validation split.
        
        Creates a 90/10% (configurable) partition of the backbone interval event driven data.
        The validation interval is selected uniformly at random from the start of the bar to 
        train_proportion of the bar. Buffer zones are applied to avoid data leakage.
        
        Args:
            event_driven_backbone_time_bar_data (pd.DataFrame): Event-driven data for the current time bar.
            physical_time_driven_backbone_time_bar_data (pd.DataFrame): Physical time-driven data for stationarity analysis.
            criterion (nn.modules.loss): Loss function. Defaults to MSELoss with sum reduction.
            optimizer (torch.optim.Optimizer): Optimizer for the model. Defaults to Adam with lr=0.001.
            epochs (int, optional): Number of epochs to train. Defaults to 10.
            validation_proportion (float, optional): Proportion of data for validation. Defaults to 0.1.
            buffer_proportion (float, optional): Buffer proportion to avoid data leakage. Defaults to 0.05.
            batch_size (int, optional): Batch size for data loaders. Defaults to 32.
            delta (float, optional): Time horizon for micro return computation. Defaults to 1.0.
            quantile (Optional[float], optional): Winsorization quantile. Defaults to 0.005.
            adf_rejection_rate_threshold (float, optional): ADF rejection rate threshold for stationarity. Defaults to 0.8.
            kpss_verification_threshold (float, optional): KPSS verification threshold. Defaults to 0.2.
            na_verification_threshold (float, optional): NA verification threshold. Defaults to 0.1.
            kpss_drop (bool, optional): Whether to use strict KPSS dropping. Defaults to True.
            parsimonious_mode (bool, optional): If True, use forward pass output; if False, use hidden activations. Defaults to False.
            correlation_threshold (float, optional): Threshold for dropping highly correlated variables. Defaults to 0.95.
        Returns:
            float: Best validation loss achieved during training.
        """

        print("\n--- Starting Global Update ---")

        print(f"Event-driven data shape: {event_driven_backbone_time_bar_data.shape}") # Debugging output
        print(f"Physical time-driven data shape: {physical_time_driven_backbone_time_bar_data.shape}") # Debugging output


        train_proportion = 1.0 - validation_proportion

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Ensure we have timestamp column for partitioning
        if 'timestamp' not in event_driven_backbone_time_bar_data.columns:
            raise ValueError("event_driven_backbone_time_bar_data must contain 'timestamp' column")
        
        # Sort data by timestamp to ensure proper ordering
        data = event_driven_backbone_time_bar_data.sort_values('timestamp').reset_index(drop=True)
        
        # Get time range of the data
        start_time = data['timestamp'].min()
        end_time = data['timestamp'].max()
        time_span = end_time - start_time
        
        # Calculate validation interval length
        validation_length = validation_proportion * time_span
        buffer_length = buffer_proportion * time_span
        
        # Calculate the maximum start position for validation interval
        # (train_proportion of the time span minus validation length)
        max_validation_start = end_time - validation_length

        # Randomly select validation interval start time
        np.random.seed()  # Ensure randomness
        validation_start = np.random.uniform(start_time, max_validation_start)
        validation_end = validation_start + validation_length
        
        # Define buffer zones around validation interval
        validation_buffer_start = validation_start - buffer_length
        validation_buffer_end = validation_end + buffer_length
        
        # Create masks for train and validation data
        # Training data: everything except validation interval + buffers
        train_mask = ~((data['timestamp'] >= validation_buffer_start) & 
                      (data['timestamp'] <= validation_buffer_end))
        
        # Validation data: only the validation interval (no buffers)
        validation_mask = ((data['timestamp'] >= validation_start) & 
                          (data['timestamp'] <= validation_end))
        
        # Split the data
        train_data = data[train_mask].reset_index(drop=True)
        validation_data = data[validation_mask].reset_index(drop=True)
        
        # Check if we have enough data
        if len(train_data) == 0:
            raise ValueError("No training data available after partitioning")
        if len(validation_data) == 0:
            raise ValueError("No validation data available after partitioning")

        # Create datasets using the CNN-LSTM preprocessor
        train_dataset = get_dataset_cnn_lstm(
            train_data, delta=delta, quantile=quantile, 
            train=True)
        
        validation_dataset = get_dataset_cnn_lstm(
            validation_data, delta=delta, quantile=quantile, 
            train=True)
        
        # Create data loaders

        # Train
        if self.device != torch.device('cpu'):
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, 
                num_workers=4, pin_memory=True, shuffle=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Validation
        if self.device != torch.device('cpu'):
            validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, 
            num_workers=4, pin_memory=True, shuffle=False
            )
        else:
            validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False
            )
        
        print(f"Training with {len(train_data)} samples, validating with {len(validation_data)} samples")
        print(f"Validation interval: [{validation_start:.2f}, {validation_end:.2f}]")
        print(f"Buffer intervals: [{validation_buffer_start:.2f}, {validation_start:.2f}] and [{validation_end:.2f}, {validation_buffer_end:.2f}]")
        
        # Train the model using existing batch_gradient_descent method
        best_validation_loss = self.batch_gradient_descent(
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=epochs,
            validation_loader=validation_loader
        )

        # Apply stationarity analysis to the physical time-driven data
        print("\n--- Running Stationarity Analysis ---")
        if physical_time_driven_backbone_time_bar_data is not None and not physical_time_driven_backbone_time_bar_data.empty:
            print("Selecting stationary variables from augmented physical time-driven data...")
            
            # Run stationarity tests on the augmented physical time-driven data
            stationary_vars = self.select_stationary_variables(
                data=physical_time_driven_backbone_time_bar_data,
                adf_rejection_rate_threshold=adf_rejection_rate_threshold,
                kpss_verification_threshold=kpss_verification_threshold,
                na_verification_threshold=na_verification_threshold,
                kpss_drop=kpss_drop
            )
        
        else:
            raise ValueError("No physical time-driven data provided for stationarity analysis")
            
        print("--- Stationarity Analysis Complete ---\n")

        # Extract neural network covariates from the trained model
        print("\n--- Constructing Neural Network Covariates ---")
        if physical_time_driven_backbone_time_bar_data is not None and not physical_time_driven_backbone_time_bar_data.empty:
            print("Extracting neural network features for physical time-driven data...")
            
            # Extract features from event-driven data using the trained model
            neural_covariates = self._extract_neural_covariates(
                event_driven_data=data,
                physical_time_data=physical_time_driven_backbone_time_bar_data,
                delta=delta,
                quantile=quantile,
                parsimonious_mode=parsimonious_mode
            )

            print(f"Neural covariates extracted with shape: {neural_covariates.shape}") # Debugging output

            # Add neural covariate columns to the full covariate set
            neural_covariate_columns = [col for col in \
                                        neural_covariates.columns\
                                              if col != 'timestamp']
            self._add_covariate_set(neural_covariate_columns)

            
            # Add neural network covariates to physical data
            if neural_covariates is not None and len(neural_covariates) > 0:
                # Set timestamp columns as float to avoid incompatible types during merging
                physical_time_driven_backbone_time_bar_data['timestamp'] = \
                    physical_time_driven_backbone_time_bar_data['timestamp']\
                        .astype(float)
                neural_covariates['timestamp'] = \
                    neural_covariates['timestamp'].astype(float)
                # Merge the neural covariates with physical data
                augmented_physical_data = pd.merge_asof(physical_time_driven_backbone_time_bar_data,
                                                        neural_covariates,
                                                        on='timestamp',
                                                        direction = 'nearest',
                                                        tolerance = 1e-6)
                print(augmented_physical_data.info()) # Debugging output
                print(f"Augmented physical data shape: {augmented_physical_data.shape}") # Debugging output
                
            else:
                raise ValueError("There was an error extracting neural covariates.")
        else:
            raise ValueError("No physical time-driven data provided for neural covariate extraction")

        print("--- Neural Network Covariate Construction Complete ---\n")

        # Drop highly correlated variables from the augmented dataset
        print("\n--- Dropping Highly Correlated Variables ---")
        if self.full_covariate_list and len(self.full_covariate_list) > 1:
            init_size = len(self.full_covariate_list)
            print(f"Initial covariate set size: {init_size}")
            
            self.drop_highly_correlated_variables(
                data=augmented_physical_data,
                threshold=correlation_threshold
            )
            
            print(f"Dropped pairs: {self.dropped_pairs}")
            print(f"Final covariate set size: {len(self.full_covariate_list)}")
            print(f"Dropped {init_size - len(self.full_covariate_list)} highly correlated variables")
        else:
            raise ValueError("No covariates available for correlation analysis or only one covariate present.")

        print("--- Highly Correlated Variable Removal Complete ---\n")

        print("--- Global Update Complete ---\n")
        print(f"Best validation loss achieved: {best_validation_loss:.4f}")
        print(f"Final covariate set:")
        for i, var in enumerate(self.full_covariate_list):
            print(f"{i + 1}. {var}")

        return {'backbone_model': self.model,
                'full_covariate_set': self.full_covariate_list}
