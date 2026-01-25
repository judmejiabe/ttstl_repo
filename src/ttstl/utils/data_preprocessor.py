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
    
    Parameters
    ----------
    data : pd.DataFrame
        Historical event-driven input DataFrame containing:

        - 'timestamp' (numeric): time of each observation,
        - 'micro' (numeric): micro price values,
        - order flow features (any column whose name contains 'of').

    quantile : float, optional
        If provided, computes winsorization bounds by calculating lower and upper 
        quantiles for each standardized feature. If None, returns None for both bound dictionaries.
        Default is 0.005.

    Returns
    -------
    tuple
        Tuple containing:

        - scaler (StandardScaler): Fitted scaler on order flow features.
        - lower_bounds (dict or None): Dictionary mapping feature names to lower 
          quantile bounds for clipping. None if quantile is None.
        - upper_bounds (dict or None): Dictionary mapping feature names to upper 
          quantile bounds for clipping. None if quantile is None.

    Raises
    ------
    ValueError
        If no order flow columns are found in the data.

    Notes
    -----
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
                                 pretrained_scaler: StandardScaler,
                                 lower_bounds: Dict[str, float],
                                 upper_bounds: Dict[str, float],
                                 past_returns: bool = False) \
                                    -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses MBO data for a CNN-LSTM model.

    Parameters
    ----------
    data : pd.DataFrame
        Event-driven input DataFrame containing:

        - 'timestamp' (numeric): time of each observation,
        - 'micro' (numeric): micro price values,
        - order flow features (any column whose name contains 'of').

    delta : float
        Time horizon (same unit as 'timestamp') used to compute micro returns.
        For future returns: future timestamp = current + delta.
        For past returns: past timestamp = current - delta.
    pretrained_scaler : StandardScaler
        Pre-fitted scaler for order flow features.
        REQUIRED to avoid lookahead bias. Use create_pretrained_scaler() to create one
        from historical data.
    lower_bounds : Dict[str, float]
        Dictionary mapping feature names to lower quantile bounds for winsorization.
        REQUIRED for consistent preprocessing.
    upper_bounds : Dict[str, float]
        Dictionary mapping feature names to upper quantile bounds for winsorization.
        REQUIRED for consistent preprocessing.
    past_returns : bool, optional
        If True, compute returns from past micro values (concurrent/observed returns).
        If False, compute returns from future micro values (forecasting returns).
        Default is False.

    Returns
    -------
    standardized_data : pd.DataFrame
        Standardized and winsorized order flow features.
    target : pd.DataFrame
        DataFrame containing two columns:

        - 'timestamp': original timestamps
        - 'micro_return': percentage change in 'micro' over the horizon delta.
          If past_returns=True: (current_micro / past_micro - 1) * 1e3
          If past_returns=False: (future_micro / current_micro - 1) * 1e3
          Observations without a corresponding past/future timestamp are assigned NaN.

    Raises
    ------
    ValueError
        If any required components are missing or if bounds dictionaries are incomplete.

    Notes
    -----
    - For future returns: Future micro values are selected as the last observation with timestamp <= timestamp + delta.
    - For past returns: Past micro values are selected as the last observation with timestamp <= timestamp - delta.
    - pretrained_scaler, lower_bounds, and upper_bounds are REQUIRED to prevent lookahead bias.
    - Use create_pretrained_scaler() to generate these components from historical data.
    """

    df = copy.deepcopy(data)
    
    # Convert to numpy arrays for faster operations
    timestamps = df['timestamp'].values
    micro_values = df['micro'].values
    
    if past_returns:
        # Calculate past returns: (current_micro / past_micro - 1) * 1e3
        # Calculate past timestamps vectorized
        past_timestamps = timestamps - delta
        min_ts = timestamps[0]
        
        # Use searchsorted for all lookups at once (vectorized)
        indices = np.searchsorted(timestamps, past_timestamps, side='right')
        
        # Handle edge cases vectorized - for past returns, valid when past_timestamps >= min_ts
        valid_mask = past_timestamps >= min_ts
        indices = np.where(valid_mask, indices - 1, 0)
        
        # Get past micro values vectorized
        past_micro = np.full(len(micro_values), np.nan)
        valid_indices = indices[valid_mask]
        past_micro[valid_mask] = micro_values[valid_indices]
        
        # Calculate micro returns vectorized: current / past - 1
        micro_returns = np.where(
            ~np.isnan(past_micro) & (past_micro != 0),
            (micro_values / past_micro - 1) * 1e3,
            np.nan
        )
        
    else:
        # Calculate future returns: (future_micro / current_micro - 1) * 1e3
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
        
        # Calculate micro returns vectorized: future / current - 1
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
    
    # Standardization and winsorization

    # Select order flow columns for standardization
    columns_to_standardize = [col for col in df.columns if 'of' in col]
    
    if not columns_to_standardize:
        raise ValueError("No order flow columns were provided.")
    
    # Standardize the order flow columns using pretrained scaler
    df_to_standardize = df[columns_to_standardize]
    
    # Use pretrained scaler (now required)
    standardized_data = pd.DataFrame(
        pretrained_scaler.transform(df_to_standardize),
        index=df_to_standardize.index,
        columns=df_to_standardize.columns
    )

    # Apply winsorization using required bounds
    # Validate that all required features have bounds
    missing_lower = [col for col in standardized_data.columns if col not in lower_bounds]
    missing_upper = [col for col in standardized_data.columns if col not in upper_bounds]
    
    if missing_lower:
        raise ValueError(f"Missing lower bounds for features: {missing_lower}")
    if missing_upper:
        raise ValueError(f"Missing upper bounds for features: {missing_upper}")
    
    # Apply winsorization to all features
    for col in standardized_data.columns:
        standardized_data[col] = np.clip(standardized_data[col], 
                                       lower_bounds[col], upper_bounds[col])

    return standardized_data, target


def get_dataset_cnn_lstm(data : pd.DataFrame,
                 delta : float, 
                 pretrained_scaler: StandardScaler,
                 lower_bounds: Dict[str, float],
                 upper_bounds: Dict[str, float],
                 train : bool = True, 
                 window_size : Optional[int] = 100,
                 past_returns : bool = False) \
                    -> CNNLSTMDataset:
    """
    Build a CNN-LSTM PyTorch dataset from preprocessed Market-by-Order (MBO) data.

    This function applies preprocessing to compute micro price returns and standardize
    features, then packages the results into sliding-window samples suitable for
    training or inference with a CNN-LSTM model.

    Parameters
    ----------
    data : pd.DataFrame
        Source MBO data. Must include the columns:
        'timestamp', 'micro', 'BIDofk', 'ASKofk'.
    delta : float
        Time horizon (same unit as 'timestamp') over which to compute micro price returns.
    pretrained_scaler : StandardScaler
        Pretrained StandardScaler to use for feature standardization. REQUIRED
        to avoid lookahead bias by using historical scaling parameters.
    lower_bounds : Dict[str, float]
        Precomputed lower quantile bounds for winsorization. REQUIRED for consistent
        preprocessing and to avoid lookahead bias.
    upper_bounds : Dict[str, float]
        Precomputed upper quantile bounds for winsorization. REQUIRED for consistent
        preprocessing and to avoid lookahead bias.
    train : bool, optional
        If True, truncate data to remove incomplete samples with NaN targets.
        For future returns, truncates at the end. For past returns, truncates at the beginning.
        If False, keep all samples (useful for inference). Default is True.
    window_size : int, optional
        Number of consecutive time steps in each input sequence. Default is 100.
    past_returns : bool, optional
        If True, compute returns as (current_micro / past_micro - 1) * 1e3.
        If False, compute returns as (future_micro / current_micro - 1) * 1e3.
        Affects where NaN values appear and how truncation is handled in train mode.
        Default is False.

    Returns
    -------
    CNNLSTMDataset
        A PyTorch Dataset object containing:

        - inputs: Tensor of shape (n_samples, window_size, n_features)
        - targets: Tensor of shape (n_samples, n_target_outputs)

    Raises
    ------
    ValueError
        If required columns are missing from the input DataFrame or if required
        preprocessing components are missing.
    """

    standardized_data, target = preprocess_mbo_data_cnn_lstm(
        data, delta, pretrained_scaler, lower_bounds, upper_bounds, past_returns
    )

    target.drop('timestamp', axis=1, inplace=True)

    if train:
        if past_returns:
            # For past returns, NaN values appear at the beginning of the dataset
            # Find the first valid (non-NaN) index
            min_index = target.dropna().index.min()
            if pd.isna(min_index):
                raise ValueError("No valid targets found in the dataset")
            inputs = torch.from_numpy(np.array(standardized_data.iloc[min_index:])).float()
            targets = torch.from_numpy(np.array(target.iloc[min_index:])).float()
        else:
            # For future returns, NaN values appear at the end of the dataset
            # Find the first NaN index to truncate
            max_index = target[target.isna().any(axis=1)].index.min()
            if pd.isna(max_index):
                # No NaN values found, use all data
                max_index = len(target)
            inputs = torch.from_numpy(np.array(standardized_data.iloc[:max_index])).float()
            targets = torch.from_numpy(np.array(target.iloc[:max_index])).float()
    else:
        inputs = torch.from_numpy(np.array(standardized_data)).float()
        targets = torch.from_numpy(np.array(target)).float()

    dataset = CNNLSTMDataset(inputs, targets, window_size)

    return dataset