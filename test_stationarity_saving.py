#!/usr/bin/env python3
"""
Test script to verify stationarity test result saving functionality.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

# Import the backbone module
from src.ttstl.backbone import Backbone

class DummyModel(nn.Module):
    """Minimal dummy model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

def create_test_data():
    """Create test time series data for stationarity testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create timestamp column
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # Create some stationary and non-stationary variables
    data = {
        'timestamp': timestamps,
        'micro_return': np.random.normal(0, 0.01, n_samples),  # Target variable
        'stationary_var1': np.random.normal(0, 1, n_samples),  # Stationary
        'stationary_var2': np.random.normal(0, 0.5, n_samples),  # Stationary  
        'trend_var': np.cumsum(np.random.normal(0, 0.1, n_samples)),  # Non-stationary (trending)
        'random_walk': np.cumsum(np.random.normal(0, 1, n_samples)),  # Non-stationary (random walk)
    }
    
    return pd.DataFrame(data)

def test_stationarity_saving():
    """Test the stationarity test saving functionality."""
    
    # Create test data
    test_data = create_test_data()
    print(f"Created test data with shape: {test_data.shape}")
    print(f"Columns: {list(test_data.columns)}")
    
    # Create temporary directory for saving results
    with tempfile.TemporaryDirectory() as temp_dir:
        save_dir = Path(temp_dir) / "test_stationarity"
        
        try:
            # Initialize backbone with a dummy model
            dummy_model = DummyModel()
            backbone = Backbone(
                model=dummy_model,
                delta_adapter=1.0,
                delta_sample=1.0,
                parsimonious_mode=False
            )
            
            # Test the select_stationary_variables method with saving
            print(f"\nTesting stationarity analysis with save_dir: {save_dir}")
            
            stationary_vars = backbone.select_stationary_variables(
                data=test_data,
                adf_rejection_rate_threshold=0.8,
                kpss_verification_threshold=0.2,
                na_verification_threshold=0.1,
                kpss_drop=True,
                target_col_name='micro_return',
                save_dir=str(save_dir)
            )
            
            print(f"Selected stationary variables: {stationary_vars}")
            
            # Check if files were created
            raw_file = save_dir / "stationarity_raw.parquet"
            summary_file = save_dir / "stationarity_summary.parquet"
            
            if raw_file.exists():
                print(f"✓ Raw stationarity results saved to: {raw_file}")
                raw_df = pd.read_parquet(raw_file)
                print(f"  Raw data shape: {raw_df.shape}")
                print(f"  Raw data columns: {list(raw_df.columns)}")
                print(f"  Sample of raw data:\n{raw_df.head()}")
            else:
                print(f"✗ Raw stationarity file not found: {raw_file}")
            
            if summary_file.exists():
                print(f"✓ Stationarity summary saved to: {summary_file}")
                summary_df = pd.read_parquet(summary_file)
                print(f"  Summary data shape: {summary_df.shape}")
                print(f"  Summary data columns: {list(summary_df.columns)}")
                print(f"  Summary data:\n{summary_df}")
            else:
                print(f"✗ Summary stationarity file not found: {summary_file}")
                
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_stationarity_saving()
