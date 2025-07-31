__author__ = 'Juan Diego Mejia Becerra'
__email__ = 'juanmejia.7565@gmail.com'

import os
from typing import List, Optional, Tuple
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ..datasets.cnn_lstm_dataset import CNNLSTMDataset

def preprocess_mbo_data_cnn_lstm(data : pd.DataFrame, 
                                 delta : float, 
                                 quantile: Optional[float] = 0.005) \
                                    -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses MBO data for a CNN-LSTM model.

    Args:
        data (pd.DataFrame): Input DataFrame containing
            - 'timestamp' (numeric): time of each observation,
            - 'micro' (numeric): micro price values,
            - order flow features (any column whose name contains 'of').
        delta (float): Time horizon (same unit as 'timestamp') used to compute future micro returns.
        quantile (Optional[float]): If provided, winsorization is applied to each standardized
            order flow feature by clipping values below the lower quantile and above the upper
            quantile. If None, no winsorization is applied.

    Returns:
        standardized_data (pd.DataFrame): Standardized (and optionally winsorized) order flow features.
        target (pd.DataFrame): Contains two columns:
            - 'timestamp': original timestamps,
            - 'micro_return': percentage change in 'micro' over the horizon delta,
              computed as (future_micro / current_micro - 1) * 1e3. Observations without a
              future timestamp are assigned NaN.

    Notes:
        - Future micro values are selected as the last observation with timestamp <= timestamp + delta.
        - StandardScaler is used to standardize order flow features.
        - Winsorization (quantile-based clipping) is applied per feature if quantile is not None.
    """

    df = copy.deepcopy(data)
    
    # Convert to numpy arrays for faster operations
    timestamps = df['timestamp'].values
    micro_values = df['micro'].values
    
    # Calculate future timestamps vectorized
    future_timestamps = timestamps + delta
    max_ts = timestamps[-1]  
    
    # Use searchsorted for all lookups at once (vectorized)
    indices = np.searchsorted(timestamps, future_timestamps, side='right')
    
    # Handle edge cases vectorized
    valid_mask = future_timestamps <= max_ts
    indices = np.where(valid_mask, indices - 1, -1)
    
    # Get future micro values vectorized
    future_micro = np.full(len(micro_values), np.nan)
    valid_indices = indices[valid_mask]
    future_micro[valid_mask] = micro_values[valid_indices]
    
    # Calculate micro returns vectorized
    micro_returns = np.where(
        ~np.isnan(future_micro) & (micro_values != 0),
        (future_micro / micro_values - 1) * 1e3,
        np.nan
    )
    
    # Create target DataFrame
    target = pd.DataFrame({
        'timestamp': timestamps,
        'micro_return': micro_returns
    })
    
    # Standardization and quantile winsorization

    # Select order flow columns for standardization
    columns_to_standardize = [col for col in df.columns if 'of' in col]
    
    if not columns_to_standardize:
        raise ValueError("No order flow columns were provided.")
    
    # Standardize the order flow columns
    df_to_standardize = df[columns_to_standardize]
    scaler = StandardScaler()
    standardized_data = pd.DataFrame(
        scaler.fit_transform(df_to_standardize),
        index=df_to_standardize.index,
        columns=df_to_standardize.columns
    )

    if quantile is not None:
        for col in standardized_data.columns:
            lower_bound = standardized_data[col].quantile(quantile)
            upper_bound = standardized_data[col].quantile(1 - quantile)
            standardized_data[col] = np.clip(standardized_data[col], 
                                             lower_bound, upper_bound)

    return standardized_data, target


def get_dataset_cnn_lstm(data : pd.DataFrame,
                 delta : float, 
                 quantile: Optional[float] = 0.005,
                 train : bool = True, 
                 window_size : Optional[int] = 100) \
                    -> CNNLSTMDataset:
        """
        Build a CNN-LSTM PyTorch dataset from preprocessed Market-by-Order (MBO) data.

        This function applies preprocessing to compute micro price returns and standardize
        features, then packages the results into sliding-window samples suitable for
        training or inference with a CNN-LSTM model.

        Args:
            data (pd.DataFrame): 
                Source MBO data. Must include the columns:
                'timestamp', 'micro', 'BIDofk', 'ASKofk'.
            delta (float): 
                Time horizon (same unit as 'timestamp') over which to compute micro price returns.
            quantile (Optional[float], default=0.005): 
                Winsorization quantile used during feature preprocessing.
            train (Optional[bool], default=True): 
                If True, truncate data at the first occurrence of NaN in targets to remove
                incomplete samples. If False, keep all samples (useful for inference).
            window_size (Optional[int], default=100): 
                Number of consecutive time steps in each input sequence.

        Returns:
            CNNLSTMDataset:
                A PyTorch Dataset object containing:
                - inputs: Tensor of shape (n_samples, window_size, n_features)
                - targets: Tensor of shape (n_samples, n_target_outputs)

        Raises:
            ValueError:
                If required columns are missing from the input DataFrame.
        """

        standardized_data, target = preprocess_mbo_data_cnn_lstm(data, delta, quantile)


        max_index = target[target.isna().any(axis=1)].index.min()
        target.drop('timestamp', axis=1, inplace=True)

        if train:
            inputs = torch.from_numpy(np.array(standardized_data.iloc[:max_index])).float()
            targets = torch.from_numpy(np.array(target.iloc[:max_index])).float()
        else:
            inputs = torch.from_numpy(np.array(standardized_data)).float()
            targets = torch.from_numpy(np.array(target)).float()

        dataset = CNNLSTMDataset(inputs, targets, window_size)

        return dataset