#!/usr/bin/env python3
"""
Simple demonstration of the _update_hyperparameters method.

This script shows the key functionality:
1. Data preparation with lagged variables and scaling
2. Train/validation split by adapter bars  
3. Elastic Net hyperparameter tuning (MSE-based)
4. Adaptive Lasso hyperparameter tuning (BIC-based, with fixed Elastic Net parameters)
"""

import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ttstl.interface import Interface

def demo_hyperparameter_tuning():
    """Demonstrate the complete hyperparameter tuning workflow."""
    print("=== TTSTL Interface Hyperparameter Tuning Demo ===\n")
    
    # 1. Setup mock backbone
    backbone = Mock()
    backbone.full_covariate_list = ['volume', 'spread', 'volatility', 'neural_1', 'neural_2']
    backbone.delta_adapter = 100.0  # 100-second adapter bars
    
    # Mock neural covariate extraction
    def mock_extract_neural_covariates(event_driven_data, physical_time_data, **kwargs):
        timestamps = physical_time_data['timestamp'].values
        return pd.DataFrame({
            'timestamp': timestamps,
            'neural_1': np.random.normal(0, 1, len(timestamps)),
            'neural_2': np.random.normal(0, 1, len(timestamps))
        })
    backbone._extract_neural_covariates = mock_extract_neural_covariates
    
    # 2. Create interface with hyperparameter grids
    interface = Interface(
        backbone_module=backbone,
        alpha_enet_grid=[0.3, 0.7],           # Elastic net mixing
        lambda_enet_grid=[0.01, 0.1],         # Elastic net regularization
        tau_adalasso_grid=[1.0, 2.0],         # Adaptive lasso weight power
        lambda_adalasso_grid=[0.01, 0.05]     # Adaptive lasso regularization
    )
    
    # 3. Generate synthetic financial data
    np.random.seed(123)
    timestamps = np.linspace(0, 1000, 200)
    
    # Traditional financial features
    data = {
        'timestamp': timestamps,
        'volume': np.random.lognormal(8, 0.5, 200),
        'spread': np.random.gamma(2, 0.001, 200),
        'volatility': np.random.gamma(1, 0.01, 200),
        'micro_return': np.random.normal(0, 0.001, 200)  # Target variable
    }
    
    event_data = pd.DataFrame(data)
    physical_data = pd.DataFrame(data)
    
    print(f"Input data: {len(physical_data)} samples")
    print(f"Backbone features: {len(backbone.full_covariate_list)}")
    print(f"Hyperparameter grids: {len(interface.alpha_enet_grid)}×{len(interface.lambda_enet_grid)} for Elastic Net")
    print(f"                     {len(interface.tau_adalasso_grid)}×{len(interface.lambda_adalasso_grid)} for Adaptive Lasso")
    
    # 4. Run complete hyperparameter tuning
    print("\n--- Running Hyperparameter Tuning ---")
    
    results = interface._update_hyperparameters(
        event_driven_backbone_time_bar_data=event_data,
        physical_time_driven_backbone_time_bar_data=physical_data,
        n_lags=2,  # Include 2 lagged target values
        validation_proportion=0.25
    )
    
    # 5. Display results
    print("\n--- Results Summary ---")
    
    enet = results['elastic_net']
    ada = results['adaptive_lasso']
    data_info = results['data_info']
    
    print(f"\n🎯 BEST HYPERPARAMETERS:")
    print(f"   Elastic Net:     α={enet['best_alpha']:.2f}, λ={enet['best_lambda']:.3f}")
    print(f"   Adaptive Lasso:  τ={ada['best_tau']:.2f}, λ={ada['best_lambda']:.3f} (BIC={ada['best_bic']:.2f})")
    
    print(f"\n📊 DATA PROCESSING:")
    print(f"   Original samples:    {data_info['total_samples']}")
    print(f"   Training samples:    {data_info['training_samples']}")
    print(f"   Validation samples:  {data_info['validation_samples']}")
    print(f"   Features (with lags): {data_info['n_features']}")
    print(f"   Lagged variables:    {data_info['n_lags']}")
    
    print(f"\n⚙️ WORKFLOW COMPLETED:")
    print("   ✅ Data preparation (neural covariates + adapter bars + lags + scaling)")
    print("   ✅ Train/validation split by adapter bars")
    print("   ✅ Elastic Net hyperparameter tuning (MSE criterion)")
    print("   ✅ Adaptive Lasso hyperparameter tuning (BIC criterion)")
    
    print(f"\n🔧 STORED IN INTERFACE:")
    print(f"   best_alpha_enet: {interface.best_alpha_enet}")
    print(f"   best_lambda_enet: {interface.best_lambda_enet}")
    print(f"   best_tau_adalasso: {interface.best_tau_adalasso}")
    print(f"   best_lambda_adalasso: {interface.best_lambda_adalasso}")
    
    print("\n✨ Hyperparameter tuning complete! Ready for variable selection and adaptation.")

if __name__ == "__main__":
    demo_hyperparameter_tuning()
