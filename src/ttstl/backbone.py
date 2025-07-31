import warnings
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import numpy as np

urca = importr('urca')
fUnitRoots = importr('fUnitRoots')

pandas2ri.activate()

class Backbone():
    def __init__(self, model : nn.Module, delta_adapter : float = 0.0) -> None:
        self.model = model
        self.device = next(self.model.parameters()).device
        self.delta_adapter = delta_adapter
        self.full_variable_set = None

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

    def train(self, 
              train_loader : DataLoader,
              criterion : nn.modules.loss,
              optimizer : torch.optim.Optimizer,
              epochs : int = 10,
              validation_loader : Optional[DataLoader] = None) -> float:
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
            print(f"Processing adapter_bar: {ell}")
            print(f'Group shape: {group.shape}')
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

    def _select_stationary_variables(self,
                                     data: pd.DataFrame,
                                     adf_rejection_rate_threshold : float = 0.8,
                                     kpss_verification_threshold : float = 0.2,
                                     na_verification_threshold : float = 0.1,
                                     kpss_drop : bool = False) -> pd.DataFrame:
        """Select stationary variables from the DataFrame.
        Args:
            data (pd.DataFrame): DataFrame containing the time series data.
        Returns:
            pd.DataFrame: DataFrame with only stationary variables.
        """
        
        rejections_df = self._run_multiple_stationarity_tests(data)
        grouped = rejections_df.groupby('variable')[['ADF', 'KPSS', 'NA']].mean()

        # Define base condition for stationarity
        adf_stationary = grouped['ADF'] > adf_rejection_rate_threshold
        high_na = grouped['NA'] > na_verification_threshold

        if kpss_drop:
            # Strict stationarity: both ADF and KPSS must agree
            kpss_stationary = grouped['KPSS'] < kpss_verification_threshold
            stationary_variables = grouped.index[adf_stationary & kpss_stationary].tolist()
            warning_variables = []
        else:
            # Lenient stationarity: only ADF required
            stationary_variables = grouped.index[adf_stationary].tolist()
            # Variables that pass ADF but fail KPSS need verification
            kpss_warning = grouped['KPSS'] > kpss_verification_threshold
            warning_variables = grouped.index[adf_stationary & kpss_warning].tolist()

        # Variables with high NA rates (among those passing ADF)
        high_na_vars = grouped.index[adf_stationary & high_na].tolist()

        self.full_variable_set = stationary_variables

        # Issue warnings
        if warning_variables:
            warnings.warn(f"Variables meeting ADF threshold but requiring verification for high KPSS rejections: {warning_variables}", UserWarning)

        if high_na_vars:
            warnings.warn(f"Variables passing stationarity criteria but with high NA rates:: {high_na_vars}", UserWarning)

        return stationary_variables

        
        
        
