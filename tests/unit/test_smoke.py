import pytest

import ttstl


@pytest.mark.lightweight
def test_smoke():
    """Basic smoke test to ensure package import works."""
    print(
        f"TTSTL version: {ttstl.__version__ if hasattr(ttstl, '__version__') else 'unknown'}"
    )

    # Verify new top-level imports
    assert hasattr(ttstl, "Adapter"), "Adapter not exposed in ttstl package"
    assert hasattr(ttstl, "Orchestrator"), "Orchestrator not exposed in ttstl package"
    assert hasattr(
        ttstl, "OrchestratorConfig"
    ), "OrchestratorConfig not exposed in ttstl package"

    assert True
