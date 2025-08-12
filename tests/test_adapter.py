#!/usr/bin/env python3
"""
Simple test for TTSTL Adapter update_adapter method.
Tests update_adapter function iterating over the first 5 adapter bars.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from ttstl.backbone import Backbone
from ttstl.interface import Interface
from ttstl.adapter import Adapter
from ttstl.architectures.cnn_lstm import CNNLSTM
from ttstl.utils.data_downloader import import_to_pandas
from ttstl.utils.data_preprocessor import get_dataset_cnn_lstm

# # Suppress R output globally
# import rpy2.rinterface_lib.callbacks
# def nullify_output(x):
#     pass
# rpy2.rinterface_lib.callbacks.consolewrite_print = nullify_output
# rpy2.rinterface_lib.callbacks.consolewrite_warnerror = nullify_output


def main():
    """Test the update_adapter method iterating over first 5 adapter bars."""
    
    print("TTSTL Adapter Simple Test - First 5 Adapter Bars")
    print("="*60)
    
    # Initialize the CNN-LSTM model - use GPU if available
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = CNNLSTM().to(device)

    # Specify credentials and data
    credentials_path = 'credentials.json'
    bucket_name = 'evolving-microstructural-tk-data'
    blob_name_train = "data.nosync/Recorded_Synthetic_Data/01102025_short_event_driven_recorded_lob_1.parquet"

    # Load minimal training data for backbone
    print("Loading minimal training data...")
    df_train = import_to_pandas(
        blob_name=blob_name_train,
        bucket_name=bucket_name,
        credentials_path=credentials_path)
    
    # Quick backbone training
    print("Quick backbone training...")
    dataset_train = get_dataset_cnn_lstm(df_train, delta=100, quantile=None)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=4)
    
    backbone = Backbone(model=model, delta_adapter=100000.0, delta_sample=100.0)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    backbone.batch_gradient_descent(train_loader, criterion, optimizer, epochs=1)
    
    # Load test data
    print("Loading test data...")
    blob_name_edlob = 'data.nosync/Recorded_Synthetic_Data/01102025_long_event_driven_recorded_lob_0.parquet'
    blob_name_ptlob = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_driven_recorded_lob_0.parquet'
    blob_name_ptme = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_market_event_data_0.parquet'

    df_edlob = import_to_pandas(blob_name=blob_name_edlob, bucket_name=bucket_name, credentials_path=credentials_path)
    df_ptlob = import_to_pandas(blob_name=blob_name_ptlob, bucket_name=bucket_name, credentials_path=credentials_path)
    df_ptme = import_to_pandas(blob_name=blob_name_ptme, bucket_name=bucket_name, credentials_path=credentials_path)

    # Reduce dataset for quick testing
    reduction_factor = 0.1
    print(f"Reducing dataset size to first {int(reduction_factor*100)}% of timestamps for faster testing...")
    edlob_min_ts, edlob_max_ts = df_edlob['timestamp'].min(), df_edlob['timestamp'].max()
    edlob_cutoff = edlob_min_ts + reduction_factor * (edlob_max_ts - edlob_min_ts)

    pt_min_ts, pt_max_ts = df_ptlob['timestamp'].min(), df_ptlob['timestamp'].max()
    pt_cutoff = pt_min_ts + reduction_factor * (pt_max_ts - pt_min_ts)

    df_edlob = df_edlob[df_edlob['timestamp'] <= edlob_cutoff].copy()
    df_ptlob = df_ptlob[df_ptlob['timestamp'] <= pt_cutoff].copy()
    df_ptme = df_ptme[df_ptme['timestamp'] <= pt_cutoff].copy()
    
    print(f"Reduced event-driven data: {len(df_edlob):,} samples")
    print(f"Reduced physical time data: {len(df_ptlob):,} samples")

    # Process data for testing
    delta = 100
    df_edlob['sampling_time_bar'] = (np.rint((df_edlob['timestamp']) / delta).astype(int) + 1) * delta

    for k in range(10):
        df_edlob[f'OFI{k}'] = df_edlob[f'BIDof{k}'] - df_edlob[f'ASKof{k}']

    ofi_columns = [f'OFI{k}' for k in range(10)]
    ofi_df = df_edlob.groupby('sampling_time_bar')[ofi_columns].sum().reset_index()
    ofi_df = ofi_df.rename(columns={'sampling_time_bar': 'timestamp'})

    df_pt = df_ptlob.merge(df_ptme, on='timestamp', how='inner')
    df_pt = df_pt.merge(ofi_df, on='timestamp', how='inner')
    
    # Create micro_return for target variable
    df_pt['micro_return'] = df_pt['micro'].pct_change() * 1e3
    df_pt.dropna(subset=['micro_return'], inplace=True)
    
    print("Running quick backbone global_update...")
    
    # Quick backbone global_update to get stationary variables
    try:
        backbone.global_update(
            event_driven_backbone_time_bar_data=df_edlob,
            physical_time_driven_backbone_time_bar_data=df_pt,
            criterion=criterion,
            optimizer=optimizer,
            epochs=0,  # Skip training
            validation_proportion=0.1,
            buffer_proportion=0.05,
            parsimonious_mode=False,
            correlation_threshold=0.95
        )
        print(f"✓ Backbone found {len(backbone.full_covariate_list)} stationary variables")
    except Exception as e:
        print(f"Error in backbone global_update: {e}")
        return False
    
    # Setup interface with simple variable selection
    print("Setting up interface with simple variable selection...")
    
    interface = Interface(backbone_module=backbone,
                         alpha_enet_grid=[0.0, 1.0],
                         lambda_enet_grid=[0.1, 1.0],
                         tau_adalasso_grid=[1.0],
                         lambda_adalasso_grid=[1.0])
    
    try:
        interface.global_update(
            event_driven_backbone_time_bar_data=df_edlob,
            physical_time_driven_backbone_time_bar_data=df_pt,
            quantile=0.005,
            target_col_name='micro_return',
            n_lags=2,
            validation_proportion=0.5,
            selection_threshold=0.5
        )
        print(f"✓ Interface selected {len(interface.selected_variables)} variables")
        print(f"✓ Selected variables: {interface.selected_variables}")
    except Exception as e:
        print(f"Error in interface global_update: {e}")
        return False
    
    # Initialize adapter
    print("\n" + "="*40)
    print("TESTING ADAPTER UPDATE_ADAPTER METHOD")
    print("="*40)
    
    adapter = Adapter(interface=interface, update_method='myopic')
    print("✓ Adapter initialized successfully")
    
    # Create adapter bars from physical time data
    df_pt['adapter_bar'] = (np.floor((df_pt['timestamp'] + 1e-6) / backbone.delta_adapter).astype(int))
    adapter_bars = sorted(df_pt['adapter_bar'].unique())

    df_edlob['adapter_bar'] = (np.floor((df_edlob['timestamp'] + 1e-6) / backbone.delta_adapter).astype(int))

    
    print(f"Found {len(adapter_bars)} adapter bars in dataset")
    print(f"Testing first 5 adapter bars: {adapter_bars[:5]}")
    
    # Test update_adapter on first 5 adapter bars
    successful_updates = 0
    
    for i, adapter_bar in enumerate(adapter_bars[:5]):
        print(f"\n--- Testing Adapter Bar {adapter_bar} ({i+1}/5) ---")
        
        # Filter data for this adapter bar
        bar_ptdata = df_pt[df_pt['adapter_bar'] == adapter_bar].copy()
        bar_edlob = df_edlob[df_edlob['adapter_bar'] == adapter_bar].copy()
        print(f"Adapter bar {adapter_bar}: {len(bar_ptdata)} observations")
        
        if len(bar_ptdata) < 10:
            print(f"Skipping bar {adapter_bar}: insufficient data ({len(bar_ptdata)} < 10)")
            continue
        
        # try:
        # Test update_adapter method
        print("Testing update_adapter method...")
        result = adapter.update_adapter(
            event_driven_adapter_time_bar_data=bar_edlob,
            physical_time_driven_adapter_time_bar_data=bar_ptdata,
            adapter_bar_number=i+1,
            target_col_name='micro_return'
        )
        
        print(f"✓ update_adapter successful!")
        print(f"  Updated parameters: {len(result)} parameters")
        print(f"  Parameter names: {list(result.keys())}")
        
        successful_updates += 1
            
        # except Exception as e:
        #     print(f"✗ update_adapter failed: {e}")
        #     continue
    
    print("\n" + "="*50)
    if successful_updates > 0:
        print("✓ TESTS PASSED")
        print(f"Successfully updated adapter on {successful_updates}/5 adapter bars")
        print("✓ Adapter update_adapter method working correctly")
        return True
    else:
        print("✗ TESTS FAILED")
        print("No successful adapter updates completed")
        return False


if __name__ == "__main__":
    success = main()
    print("\n" + "="*50)
    if success:
        print("ADAPTER TEST COMPLETED SUCCESSFULLY!")
    else:
        print("ADAPTER TEST FAILED!")
    print("="*50)
