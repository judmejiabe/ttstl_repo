__author__ = 'Juan Diego Mejia Becerra'
__email__ = 'juanmejia.7565@gmail.com'

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from ..backbone import CNNLSTMDataset

def preprocess_mbo_data_cnn_lstm(path : str, 
                                 delta : float, 
                                 quantile: Optional[float] = 0.005) \
                                    -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Preprocesses the input data to prepare it for the CNN-LSTM model.
        This function reads a Parquet file containing MBO data and 
        generates features and a target variable to feed the backbone model.
        It calculates the percentage change in 'micro' values over specified horizons 
        and standardizes selected columns containing 'of'.
        Args:
            path (str): The path to the Parquet file containing the data.
                The file should contain at least a 'timestamp' and a 'micro' column.
            delta (float): Length of Sampling time bars (same unit as 'timestamp').
            quantile (Optional[float]): If provided, the function conducts winsorization 
                                        on the inputs columns based on the specified quantile.
        Returns:
            tuple: A tuple containing two pandas DataFrames:
                - standardized_data: DataFrame containing the standardized features.
                - target: DataFrame containing the target variable, which represents 
                  the percentage change in 'micro' values over the specified horizons.
                  Each column in the target DataFrame represents the percentage change
                  over a specific horizon.
        Notes:
            - The function assumes that the input Parquet file contains 'timestamp' and 'micro' columns.
            - The target variable is calculated as the percentage change in 'micro' values
              over the specified time horizons, relative to the previous 'micro' value, multiplied by 1e3.
            - Columns containing 'of' are selected for standardization using StandardScaler.
            - Missing values in the 'micro' column after the horizon are filled with NaN.
        """

    df = pd.read_parquet(path)
    
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


def get_datasets_cnn_lstm(folder_path : str, 
                 file_names : List[str], 
                 delta : float, 
                 train : Optional[bool] = True, 
                 window_size : Optional[int] = 100) \
                    -> List[CNNLSTMDataset]:
    """
    Creates a list of CNNLSTMDataset objects from multiple MBO data files for CNN-LSTM training.

    This function processes multiple parquet files containing Market-by-Order (MBO) data,
    preprocesses each file to generate features and targets, and creates PyTorch datasets
    suitable for CNN-LSTM model training. The function handles both training and inference
    modes with different data filtering strategies.
    
    Args:
        folder_path (str): Path to the directory containing the parquet files.
                          Should be a valid directory path or pathlib.Path object.
        file_names (List[str]): List of parquet file names to process.
                               Each file should contain MBO data with 'timestamp' and 'micro' columns.
        delta (float): Time horizon for calculating micro price returns.
                      Same unit as 'timestamp' column in the data files.
        train (Optional[bool], default=True): Processing mode flag.
                                            - True: Training mode - filters out rows with NaN targets
                                            - False: Inference mode - keeps all data including NaN targets
        window_size (Optional[int], default=100): Sequence length for the CNN-LSTM model.
                                                 Number of consecutive observations to include in each sample.
    
    Returns:
        List[CNNLSTMDataset]: List of PyTorch datasets, one for each input file.
                             Each dataset contains:
                             - Windowed input sequences (features)
                             - Corresponding target values (micro returns)
                             - Proper indexing for batch processing
    
    Raises:
        FileNotFoundError: If any of the specified files don't exist in folder_path.
        ValueError: If preprocessing fails due to missing required columns or invalid data.
        
    Notes:
        - In training mode, the function automatically removes incomplete samples by finding
          the first row where target values become NaN and truncating the dataset there.
        - The timestamp column is removed from targets as it's not needed for model training.
        - Input features are standardized and optionally winsorized during preprocessing.
        - Each dataset is independent and can be used for separate model training or
          combined for multi-file training scenarios.
        - The function assumes that folder_path supports the '/' operator (pathlib.Path or similar).
    """

    datasets = []

    for file_name in file_names:

        path = folder_path / file_name
        inputs_df, target_df = preprocess_mbo_data_cnn_lstm(path, delta)

        max_index = target_df[target_df.isna().any(axis=1)].index.min()
        target_df.drop('timestamp', axis=1, inplace=True)

        if train:
            inputs = torch.from_numpy(np.array(inputs_df.iloc[:max_index])).float()
            targets = torch.from_numpy(np.array(target_df.iloc[:max_index])).float()
        else:
            inputs = torch.from_numpy(np.array(inputs_df)).float()
            targets = torch.from_numpy(np.array(target_df)).float()

        dataset = CNNLSTMDataset(inputs, targets, window_size)
        
        datasets.append(dataset)

    return datasets