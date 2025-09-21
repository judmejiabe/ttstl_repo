#!/usr/bin/env python
"""
Test script to verify the frozen environment is working correctly.

This script tests all key packages and R integration to ensure
the environment is properly set up after freezing versions.
"""

import sys
import traceback

def test_basic_imports():
    """Test that all essential packages can be imported."""
    print("🔍 Testing Basic Package Imports...")
    print("=" * 50)
    
    packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('torch', None),
        ('torchvision', None),
        ('torchaudio', None),
        ('sklearn', None),
        ('statsmodels', None),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('scipy', None),
        ('pyarrow', None),
        ('google.cloud.storage', None),
        ('gcsfs', None),
        ('pytest', None),
    ]
    
    results = []
    for package, alias in packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✅ {package}")
            results.append((package, True, None))
        except Exception as e:
            print(f"❌ {package}: {e}")
            results.append((package, False, str(e)))
    
    return results

def test_r_integration():
    """Test R integration through rpy2."""
    print("\n🔍 Testing R Integration...")
    print("=" * 50)
    
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        print("✅ rpy2 import successful")
        
        # Test basic R execution
        ro.r('x <- 1:10')
        result = ro.r('sum(x)')
        print(f"✅ Basic R execution: sum(1:10) = {result[0]}")
        
        # Test R package loading
        r_packages = ['rugarch', 'readr', 'dplyr', 'ggplot2', 'urca']
        for pkg in r_packages:
            try:
                ro.r(f'library({pkg})')
                print(f"✅ R package {pkg} loaded")
            except Exception as e:
                print(f"❌ R package {pkg}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ R integration failed: {e}")
        traceback.print_exc()
        return False

def test_version_consistency():
    """Test that key package versions are as expected."""
    print("\n🔍 Testing Version Consistency...")
    print("=" * 50)
    
    try:
        import numpy as np
        import pandas as pd
        import torch
        import sklearn
        import matplotlib
        import rpy2
        
        versions = {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'torch': torch.__version__,
            'scikit-learn': sklearn.__version__,
            'matplotlib': matplotlib.__version__,
            'rpy2': rpy2.__version__,
        }
        
        expected = {
            'numpy': '2.0.2',
            'pandas': '2.3.1', 
            'torch': '2.5.0+cu118',
            'scikit-learn': '1.6.1',
            'matplotlib': '3.9.4',
            'rpy2': '3.5.11',
        }
        
        all_match = True
        for pkg, actual in versions.items():
            expected_ver = expected.get(pkg, 'unknown')
            if actual == expected_ver:
                print(f"✅ {pkg}: {actual}")
            else:
                print(f"⚠️  {pkg}: expected {expected_ver}, got {actual}")
                all_match = False
        
        return all_match
        
    except Exception as e:
        print(f"❌ Version check failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key packages."""
    print("\n🔍 Testing Basic Functionality...")
    print("=" * 50)
    
    try:
        # Test numpy
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        print("✅ NumPy basic operations")
        
        # Test pandas
        import pandas as pd
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        assert len(df) == 3
        print("✅ Pandas DataFrame creation")
        
        # Test torch
        import torch
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert tensor.sum().item() == 6.0
        print("✅ PyTorch tensor operations")
        
        # Test matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        plt.close(fig)
        print("✅ Matplotlib plotting")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Frozen Environment Setup")
    print("=" * 60)
    
    # Run all tests
    import_results = test_basic_imports()
    r_success = test_r_integration()
    version_success = test_version_consistency()
    func_success = test_basic_functionality()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    import_success = all(result[1] for result in import_results)
    failed_imports = [result[0] for result in import_results if not result[1]]
    
    print(f"📦 Package Imports: {'✅ PASS' if import_success else '❌ FAIL'}")
    if failed_imports:
        print(f"   Failed: {', '.join(failed_imports)}")
    
    print(f"🔗 R Integration: {'✅ PASS' if r_success else '❌ FAIL'}")
    print(f"📋 Version Consistency: {'✅ PASS' if version_success else '⚠️ PARTIAL'}")
    print(f"⚡ Basic Functionality: {'✅ PASS' if func_success else '❌ FAIL'}")
    
    overall_success = import_success and r_success and func_success
    print(f"\n🎯 Overall: {'✅ ENVIRONMENT READY' if overall_success else '❌ ISSUES DETECTED'}")
    
    if overall_success:
        print("\n🎉 The frozen environment is working correctly!")
        print("   All key packages are installed and functional.")
        print("   R integration is working properly.")
        print("   Ready for GARCH analysis and other tasks.")
    else:
        print("\n⚠️  Some issues were detected in the environment.")
        print("   Please review the failed tests above.")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())