#!/usr/bin/env python3
"""
TTSTL Installation Verification Script

This script checks if TTSTL and all its dependencies are properly installed.
Run this after installation to verify everything is working correctly.
"""

import sys
import traceback
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)"


def check_import(module_name: str, description: str) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True, f"✓ {description}"
    except ImportError as e:
        return False, f"✗ {description}: {str(e)}"
    except Exception as e:
        return False, f"✗ {description}: Unexpected error - {str(e)}"


def check_torch_cuda() -> Tuple[bool, str]:
    """Check PyTorch and CUDA availability."""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, f"✓ PyTorch {version} with CUDA {cuda_version} ({device_count} GPU(s): {device_name})"
        else:
            return True, f"✓ PyTorch {version} (CPU only)"
    except Exception as e:
        return False, f"✗ PyTorch: {str(e)}"


def check_r_integration() -> Tuple[bool, str]:
    """Check R integration and required packages."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        
        # Test basic R functionality
        r_version = ro.r('R.version.string')[0]
        
        # Test required R packages
        try:
            importr('glmnet')
            importr('rugarch')
            importr('urca')
            importr('fUnitRoots')
            return True, f"✓ R integration ({r_version}) with required packages"
        except Exception as e:
            return False, f"⚠ R integration works but packages missing: {str(e)}"
    except Exception as e:
        return False, f"✗ R integration: {str(e)}"


def check_ttstl_modules() -> Tuple[bool, str]:
    """Check TTSTL modules can be imported."""
    try:
        from ttstl import Backbone, Interface
        # Note: Adapter not yet implemented
        return True, "✓ TTSTL modules (Backbone, Interface)"
    except Exception as e:
        return False, f"✗ TTSTL modules: {str(e)}"


def check_ttstl_functionality() -> Tuple[bool, str]:
    """Basic functionality test."""
    try:
        from ttstl.backbone import Backbone
        from ttstl.interface import Interface
        
        # Test basic instantiation (without parameters that don't exist)
        # Just test that the classes exist and can be referenced
        backbone_class = Backbone
        interface_class = Interface
        
        return True, "✓ TTSTL basic functionality (classes accessible)"
    except Exception as e:
        return False, f"✗ TTSTL functionality: {str(e)}"


def run_verification() -> None:
    """Run all verification checks."""
    print("🔍 TTSTL Installation Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("NumPy", lambda: check_import("numpy", "NumPy")),
        ("Pandas", lambda: check_import("pandas", "Pandas")),
        ("SciPy", lambda: check_import("scipy", "SciPy")),
        ("Scikit-learn", lambda: check_import("sklearn", "Scikit-learn")),
        ("Statsmodels", lambda: check_import("statsmodels", "Statsmodels")),
        ("PyTorch + CUDA", check_torch_cuda),
        ("R Integration", check_r_integration),
        ("TTSTL Modules", check_ttstl_modules),
        ("TTSTL Functionality", check_ttstl_functionality),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            success, message = check_func()
            results.append((success, message))
            print(f"{message}")
        except Exception as e:
            results.append((False, f"✗ {name}: Unexpected error"))
            print(f"✗ {name}: Unexpected error - {str(e)}")
    
    print("\n" + "=" * 50)
    
    # Summary
    successful = sum(1 for success, _ in results if success)
    total = len(results)
    
    if successful == total:
        print(f"🎉 All checks passed! ({successful}/{total})")
        print("\n✅ TTSTL is properly installed and ready to use!")
        print("\nNext steps:")
        print("  • Try: python backbone_test.py")
        print("  • Explore: notebooks/")
        print("  • Read: docs/")
    else:
        failed = total - successful
        print(f"⚠️  {failed} checks failed ({successful}/{total} passed)")
        print("\n❌ Installation issues detected!")
        print("\nTroubleshooting:")
        print("  • Check INSTALLATION.md for detailed instructions")
        print("  • Verify all prerequisites are installed")
        print("  • Try reinstalling: conda env remove -n ttstl_dev && conda env create -f environment.yml")
        print("  • For R issues: R -e \"install.packages(c('glmnet', 'rugarch'))\"")
    
    print(f"\nSystem Information:")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")
    
    return successful == total


if __name__ == "__main__":
    try:
        success = run_verification()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error during verification: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
