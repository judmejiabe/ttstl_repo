#!/usr/bin/env python3
"""
Quick debug script to examine the hyperparameter transfer issue.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ttstl.interface import Interface
from ttstl.utils.data_downloader import import_to_pandas

def debug_hyperparameters():
    """Debug hyperparameter transfer and coefficient values."""
    print("="*60)
    print("DEBUGGING HYPERPARAMETER TRANSFER")
    print("="*60)
    
    # Load data (similar to demo)
    print("Loading data from GCS...")
    credentials_path = 'credentials.json'
    bucket_name = 'evolving-microstructural-tk-data'
    
    # Load event-driven and physical time data
    blob_name_edlob = 'data.nosync/Recorded_Synthetic_Data/01102025_long_event_driven_recorded_lob_0.parquet'
    blob_name_ptlob = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_driven_recorded_lob_0.parquet'
    blob_name_ptme = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_market_event_data_0.parquet'

    df_edlob = import_to_pandas(blob_name=blob_name_edlob, bucket_name=bucket_name, credentials_path=credentials_path)
    df_ptlob = import_to_pandas(blob_name=blob_name_ptlob, bucket_name=bucket_name, credentials_path=credentials_path)
    df_ptme = import_to_pandas(blob_name=blob_name_ptme, bucket_name=bucket_name, credentials_path=credentials_path)
    
    print(f"Event data columns: {df_edlob.columns.tolist()}")
    print(f"Physical data columns: {df_ptme.columns.tolist()}")
    
    # Take a small subset for debugging - just first 50k rows for event, 1k for physical
    event_data = df_edlob.iloc[:50000].copy()
    physical_data = df_ptme.iloc[:1000].copy()
    
    print(f"Loaded data shapes: event={event_data.shape}, physical={physical_data.shape}")
    
    # Create interface with smaller grids for quick testing
    alpha_grid = [0.5, 0.75, 1.0]
    lambda_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    tau_grid = [0.5, 1.0, 1.5]
    lambda_adalasso_grid = [0.1, 1.0, 10.0, 100.0]
    
    interface = Interface(
        alpha_enet_grid=alpha_grid,
        lambda_enet_grid=lambda_grid,
        tau_adalasso_grid=tau_grid,
        lambda_adalasso_grid=lambda_adalasso_grid
    )
    
    # Run hyperparameter tuning only
    print("\n1. Running hyperparameter tuning...")
    hyperparameter_results = interface._update_hyperparameters(
        event_driven_backbone_time_bar_data=event_data,
        physical_time_driven_backbone_time_bar_data=physical_data,
        quantile=0.005,
        target_col_name='micro_return',
        n_lags=3,
        validation_proportion=0.3
    )
    
    print(f"\nOptimal hyperparameters found:")
    print(f"  Elastic Net: α={interface.best_alpha_enet:.3f}, λ={interface.best_lambda_enet:.6f}")
    print(f"  Adaptive Lasso: τ={interface.best_tau_adalasso:.3f}, λ={interface.best_lambda_adalasso:.6f}")
    
    # Test one adapter bar with these parameters
    print(f"\n2. Testing optimal parameters on one adapter bar...")
    
    # Prepare data
    augmented_data = interface._prepare_data(
        event_driven_backbone_time_bar_data=event_data,
        physical_time_driven_backbone_time_bar_data=physical_data,
        quantile=0.005,
        target_col_name='micro_return',
        n_lags=3
    )
    
    # Get first adapter bar
    first_bar = augmented_data['adapter_bar'].iloc[0]
    bar_data = augmented_data[augmented_data['adapter_bar'] == first_bar].copy()
    
    print(f"Testing on adapter bar {first_bar} with {len(bar_data)} samples")
    
    # Get features and target
    X_bar = bar_data[interface.extended_covariate_list].values
    y_bar = bar_data['micro_return'].values
    
    print(f"Feature matrix shape: {X_bar.shape}")
    print(f"Target vector shape: {y_bar.shape}")
    
    # Apply adaptive lasso with optimal hyperparameters
    ada_result = interface._fit_flexible_adalasso(
        X=X_bar,
        y=y_bar,
        alpha_enet=interface.best_alpha_enet,
        lambda_enet=interface.best_lambda_enet,
        tau_adalasso=interface.best_tau_adalasso,
        lambda_adalasso=interface.best_lambda_adalasso
    )
    
    # Examine coefficients
    ada_coefficients = ada_result['ada_coefficients']
    enet_coefficients = ada_result['enet_coefficients']
    penalty_weights = ada_result['penalty_weights']
    
    print(f"\n3. Coefficient Analysis:")
    print(f"  Elastic Net coefficients shape: {enet_coefficients.shape}")
    print(f"  Adaptive Lasso coefficients shape: {ada_coefficients.shape}")
    print(f"  Penalty weights shape: {penalty_weights.shape}")
    
    # Remove intercept for analysis
    if len(ada_coefficients) == len(interface.extended_covariate_list) + 1:
        ada_coef_no_intercept = ada_coefficients[1:]
        enet_coef_no_intercept = enet_coefficients[1:]
        print(f"  Intercept detected and removed")
    else:
        ada_coef_no_intercept = ada_coefficients
        enet_coef_no_intercept = enet_coefficients
    
    # Count variables above different thresholds
    thresholds = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    print(f"\n4. Variable count by tolerance threshold:")
    print(f"  Threshold    | ElasticNet | AdaptiveLasso")
    print(f"  -------------|------------|-------------")
    for thresh in thresholds:
        enet_count = np.sum(np.abs(enet_coef_no_intercept) > thresh)
        ada_count = np.sum(np.abs(ada_coef_no_intercept) > thresh)
        print(f"  {thresh:>10.0e} | {enet_count:>10d} | {ada_count:>12d}")
    
    # Show top 10 variables by absolute coefficient value
    print(f"\n5. Top 10 variables by absolute Adaptive Lasso coefficient:")
    coef_abs = np.abs(ada_coef_no_intercept)
    top_indices = np.argsort(coef_abs)[::-1][:10]
    
    for i, idx in enumerate(top_indices, 1):
        var_name = interface.extended_covariate_list[idx]
        ada_coef = ada_coef_no_intercept[idx]
        enet_coef = enet_coef_no_intercept[idx]
        weight = penalty_weights[idx]
        print(f"  {i:2d}. {var_name:<25} | Ada: {ada_coef:>10.6f} | ENet: {enet_coef:>10.6f} | Weight: {weight:>8.2f}")
    
    print(f"\n6. Hyperparameter sensitivity test:")
    print(f"Testing different lambda values with same other parameters...")
    
    test_lambdas = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    print(f"  Lambda       | Variables > 1e-6")
    print(f"  -------------|----------------")
    
    for test_lambda in test_lambdas:
        try:
            test_result = interface._fit_flexible_adalasso(
                X=X_bar,
                y=y_bar,
                alpha_enet=interface.best_alpha_enet,
                lambda_enet=interface.best_lambda_enet,
                tau_adalasso=interface.best_tau_adalasso,
                lambda_adalasso=test_lambda
            )
            
            test_coefficients = test_result['ada_coefficients']
            if len(test_coefficients) == len(interface.extended_covariate_list) + 1:
                test_coef_no_intercept = test_coefficients[1:]
            else:
                test_coef_no_intercept = test_coefficients
            
            var_count = np.sum(np.abs(test_coef_no_intercept) > 1e-6)
            print(f"  {test_lambda:>10.1f} | {var_count:>14d}")
            
        except Exception as e:
            print(f"  {test_lambda:>10.1f} | Error: {str(e)[:20]}...")
    
    print(f"\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    debug_hyperparameters()
