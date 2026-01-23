
import pytest
import ttstl

@pytest.mark.lightweight
def test_smoke():
    """Basic smoke test to ensure package import works."""
    print(f"TTSTL version: {ttstl.__version__ if hasattr(ttstl, '__version__') else 'unknown'}")
    assert True
