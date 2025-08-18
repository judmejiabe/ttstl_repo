#!/usr/bin/env python3
"""
Simple script to demonstrate the adapter parameters object being returned.
"""
import sys
import os
sys.path.append('src')

from pathlib import Path
from ttstl.orchestrator import Orchestrator, OrchestratorConfig
import pandas as pd
import numpy as np

def check_adapter_parameters():
    """Create a simple adapter and show the parameters object structure."""
    
    # Import adapter directly
    from ttstl.adapter import Adapter
    from ttstl.interface import Interface
    from ttstl.backbone import BackboneWrapper
    
    # Create a mock backbone and interface (adapter needs one)
    backbone = BackboneWrapper()
    interface = Interface(backbone_module=backbone)
    
    # Create adapter with simple_average update method (this is the default in orchestrator)
    adapter = Adapter(
        interface=interface,
        update_method='simple_average',  # Explicitly set to simple_average
        arma_order=(1, 1),
        garch_order=(1, 1),
        dist='norm',
        model='sGARCH',
        archm=False,
        parameters={}  # Start with empty parameters
    )
    
    print("🔍 ADAPTER PARAMETERS OBJECT INSPECTION")
    print("="*50)
    print(f"Update method: {adapter.update_method}")
    print(f"Current parameters type: {type(adapter.parameters)}")
    print(f"Current parameters content: {adapter.parameters}")
    print(f"Number of parameters: {len(adapter.parameters) if adapter.parameters else 0}")
    
    # Mock some parameters to show the structure
    mock_parameters = {
        'mu': 0.0001,
        'ar1': 0.1234,
        'ma1': -0.5678,
        'mxreg1': 0.0123,
        'mxreg2': -0.0456,
        'mxreg3': 0.0789,
        'omega': 0.0001,
        'alpha1': 0.1500,
        'beta1': 0.8000
    }
    
    print("\n📋 EXAMPLE PARAMETERS STRUCTURE:")
    print("="*50)
    for param, value in mock_parameters.items():
        print(f"  {param:>10s}: {value:>10.6f}")
    
    # Show simple average update method
    print(f"\n⚙️  UPDATE METHOD: {adapter.update_method}")
    print("="*50)
    print("The simple_average method updates parameters as:")
    print("  new_param = (ell-1)/ell * old_param + 1/ell * fitted_param")
    print("  where ell is the adapter bar number (1-indexed)")
    print("\nThis gives more weight to recent observations while")
    print("maintaining a running average of all previous estimates.")
    
    return adapter

if __name__ == "__main__":
    check_adapter_parameters()
