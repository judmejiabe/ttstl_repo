__author__ = 'Juan Diego Mejia Becerra'
__email__ = 'juanmejia.7565@gmail.com'

import os
from typing import List, Optional, Tuple, Dict
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ..datasets.cnn_lstm_dataset import CNNLSTMDataset

def create_pretrained_scaler(data: pd.DataFrame, 
                           quantile: Optional[float] = 0.005) \
                              -> Tuple[StandardScaler, Optional[dict], Optional[dict]]:
    """
    Create a pretrained scaler on historical data to avoid lookahead bias.
    
    This method fits a StandardScaler on order flow features from historical data
    and optionally computes quantile-based clipping bounds. The returned scaler
    and bounds can be reused for consistent preprocessing of future data.
    
    Args:
        data (pd.DataFrame): Historical event-driven input DataFrame containing
            - 'timestamp' (numeric): time of each observation,
            - 'micro' (numeric): micro price values,
            - order flow features (any column whose name contains 'of').
        quantile (Optional[float]): If provided, computes winsorization bounds
            by calculating lower and upper quantiles for each standardized feature.
            If None, returns None for both bound dictionaries.
    
    Returns:
        Tuple containing:
        - scaler (StandardScaler): Fitted scaler on order flow features
        - lower_bounds (Optional[dict]): Dictionary mapping feature names to lower 
          quantile bounds for clipping. None if quantile is None.
        - upper_bounds (Optional[dict]): Dictionary mapping feature names to upper 
          quantile bounds for clipping. None if quantile is None.
    
    Raises:
        ValueError: If no order flow columns are found in the data.
    
    Notes:
        - Only fits on order flow features (columns containing 'of')
        - Quantile bounds are computed on standardized features
        - This method should be called once on historical data, then reused
    """
    df = copy.deepcopy(data)
    
    # Select order flow columns for standardization
    columns_to_standardize = [col for col in df.columns if 'of' in col]
    
    if not columns_to_standardize:
        raise ValueError("No order flow columns were provided.")
    
    # Fit scaler on order flow columns
    df_to_standardize = df[columns_to_standardize]
    scaler = StandardScaler()
    scaler.fit(df_to_standardize)
    
    # Compute quantile bounds if requested
    lower_bounds = None
    upper_bounds = None
    
    if quantile is not None:
        # Transform data to get standardized features for quantile computation
        standardized_data = pd.DataFrame(
            scaler.transform(df_to_standardize),
            index=df_to_standardize.index,
            columns=df_to_standardize.columns
        )
        
        # Compute quantile bounds for each feature
        lower_bounds = {}
        upper_bounds = {}
        
        for col in standardized_data.columns:
            lower_bounds[col] = standardized_data[col].quantile(quantile)
            upper_bounds[col] = standardized_data[col].quantile(1 - quantile)
    
    return scaler, lower_bounds, upper_bounds


def preprocess_mbo_data_cnn_lstm(data : pd.DataFrame, 
                                 delta : float, 
                                 quantile: Optional[float] = 0.005,
                                 pretrained_scaler: Optional[StandardScaler] = None,
                                 lower_bounds: Optional[dict] = None,
                                 upper_bounds: Optional[dict] = None) \
                                    -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses MBO data for a CNN-LSTM model.

    Args:
        data (pd.DataFrame): Event-driven input DataFrame containing
            - 'timestamp' (numeric): time of each observation,
            - 'micro' (numeric): micro price values,
            - order flow features (any column whose name contains 'of').
        delta (float): Time horizon (same unit as 'timestamp') used to compute future micro returns.
        quantile (Optional[float]): If provided, winsorization is applied to each standardized
            order flow feature by clipping values below the lower quantile and above the upper
            quantile. If None, no winsorization is applied.
        pretrained_scaler (Optional[StandardScaler]): Pre-fitted scaler for order flow features.
            REQUIRED to avoid lookahead bias. Use create_pretrained_scaler() to create one
            from historical data.
        lower_bounds (Optional[dict]): Dictionary mapping feature names to lower quantile bounds
            for winsorization. REQUIRED when quantile is specified to avoid lookahead bias.
        upper_bounds (Optional[dict]): Dictionary mapping feature names to upper quantile bounds
            for winsorization. REQUIRED when quantile is specified to avoid lookahead bias.

    Returns:
        standardized_data (pd.DataFrame): Standardized (and optionally winsorized) order flow features.
        target (pd.DataFrame): Contains two columns:
            - 'timestamp': original timestamps,
            - 'micro_return': percentage change in 'micro' over the horizon delta,
              computed as (future_micro / current_micro - 1) * 1e3. Observations without a
              future timestamp are assigned NaN.

    Notes:
        - Future micro values are selected as the last observation with timestamp <= timestamp + delta.
        - pretrained_scaler is REQUIRED to prevent lookahead bias in feature standardization.
        - lower_bounds and upper_bounds are REQUIRED when quantile is specified to prevent
          lookahead bias in winsorization.
        - Use create_pretrained_scaler() to generate these components from historical data.
    
    Raises:
        ValueError: If pretrained_scaler is None, or if quantile is specified but lower_bounds
                   or upper_bounds are None.
        - If pretrained_scaler is provided, uses it for consistent scaling across time periods.
        - Winsorization bounds from pretrained data are preferred when available to avoid lookahead bias.
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
    
    if pretrained_scaler is not None:
        # Use pretrained scaler to avoid lookahead bias
        standardized_data = pd.DataFrame(
            pretrained_scaler.transform(df_to_standardize),
            index=df_to_standardize.index,
            columns=df_to_standardize.columns
        )
    else:
        # Require pretrained scaler to avoid lookahead bias
        raise ValueError("pretrained_scaler must be provided to avoid lookahead bias. "
                        "Use create_pretrained_scaler() to create one from historical data.")

    if quantile is not None:
        if pretrained_scaler is not None and lower_bounds is not None and upper_bounds is not None:
            # Use pretrained quantile bounds to avoid lookahead bias
            for col in standardized_data.columns:
                if col in lower_bounds and col in upper_bounds:
                    standardized_data[col] = np.clip(standardized_data[col], 
                                                   lower_bounds[col], upper_bounds[col])
        else:
            # Require pretrained bounds to avoid lookahead bias
            raise ValueError("lower_bounds and upper_bounds must be provided when quantile is specified "
                           "to avoid lookahead bias. Use create_pretrained_scaler() to create them from historical data.")

    return standardized_data, target


def get_dataset_cnn_lstm(data : pd.DataFrame,
                 delta : float, 
                 quantile: Optional[float] = 0.005,
                 train : bool = True, 
                 window_size : Optional[int] = 100,
                 pretrained_scaler: Optional[StandardScaler] = None,
                 lower_bounds: Optional[Dict[str, float]] = None,
                 upper_bounds: Optional[Dict[str, float]] = None) \
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
            pretrained_scaler (Optional[StandardScaler], default=None):
                Pretrained StandardScaler to use for feature standardization. If provided,
                avoids lookahead bias by using historical scaling parameters.
            lower_bounds (Optional[Dict[str, float]], default=None):
                Precomputed lower quantile bounds for winsorization. Used with pretrained_scaler
                to avoid lookahead bias.
            upper_bounds (Optional[Dict[str, float]], default=None):
                Precomputed upper quantile bounds for winsorization. Used with pretrained_scaler
                to avoid lookahead bias.

        Returns:
            CNNLSTMDataset:
                A PyTorch Dataset object containing:
                - inputs: Tensor of shape (n_samples, window_size, n_features)
                - targets: Tensor of shape (n_samples, n_target_outputs)

        Raises:
            ValueError:
                If required columns are missing from the input DataFrame.
        """

        standardized_data, target = preprocess_mbo_data_cnn_lstm(
            data, delta, quantile, pretrained_scaler, lower_bounds, upper_bounds
        )


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