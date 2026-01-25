# Contributing to TTSTL

Thank you for your interest in contributing to the Two-Time-Scale Transfer Learning (TTSTL) framework! This document provides guidelines for developers and reviewers to ensure a smooth workflow.

## 🧪 Testing

We use `pytest` for testing. The test suite is divided into two tiers to optimize your development cycle.

### Test Markers

We have two primary markers for tests:

1.  **Lightweight** (`@pytest.mark.lightweight`)
    *   **Purpose:** Fast sanity checks, logic verification, and basic integration tests.
    *   **Time:** Should run in seconds.
    *   **When to run:** Before every commit.

2.  **Comprehensive** (`@pytest.mark.comprehensive`)
    *   **Purpose:** Full-pipeline experiments that replicate the paper's results. These verify the entire system end-to-end.
    *   **Time:** Can take 30+ minutes.
    *   **When to run:** Before merging a PR or when modifying core logic in the backbone, adapter, or orchestrator.

### Running Tests

**Run lightweight tests (Recommended for active dev):**
```bash
pytest -m lightweight
```

**Run comprehensive suite (Full verification):**
```bash
pytest -m comprehensive
```

**Run all tests::**
```bash
pytest
```

## 🔧 R Dependency Handling in CI/CD

The "Quality Check" CI/CD workflow runs in a Python-only environment to be fast and lightweight. Because of this, it does **not** have the R environment or R packages (`urca`, `fUnitRoots`, `rugarch`) installed.

To ensure the package can be imported and linted in this environment:
1.  **Top-level R imports are wrapped in `try-except` blocks** in modules like `src/ttstl/backbone.py` and `src/ttstl/adapter.py`.
2.  If R packages are missing, the import variable (e.g., `urca`) is set to `None`.
3.  **Smoke tests** verify that the package imports successfully even without R.
4.  **Comprehensive tests** require the full Conda environment (with R) and will fail if run in the lightweight CI environment (which is expected).

**For Contributors:**
- Do **not** remove the `try-except` blocks around `rpy2` imports.
- When adding new R dependencies, follow this pattern to keep the CI pipeline green.

## 📂 Project Structure

* `manuscript/`: Contains the research paper and supplementary materials in LaTeX format.
* `outputs/`: Directory for storing experiment outputs and post-processing results.
    * `analysis_market_data_experiment_results/`: Original analysis of market data experiments results.
    * `analysis_synthetic_data_experiment_results/`: Original analysis of synthetic data experiments results.
* `src/ttstl/`: Core package source code.
    * `adapter.py`: Adapter module implementation (Volatility modeling & Probabilistic forecasting, see image in the Architecture section).
    * `backbone.py`: Backbone module implementation (Deep learning feature extractor, see image in the Architecture section).
    * `interface.py`: Interface module implementation (Variable selection & R-Python bridging, see image in the Architecture section).
    * `orchestrator.py`: Main API matching backbone, interface, and adapter modules.
    * `architectures/`: Neural network architecture definitions (e.g., CNN-LSTM).
    * `datasets/`: PyTorch dataset implementations for LOB data.
    * `utils/`: Utility functions for data processing and metrics.
    * `r_scripts/`: R scripts wrapped by the modules for statistical methodologies.
* `scripts/`: Experiments and analysis.
    * `analyze_market_data_experiment_results.py`: Post-processing and visualization for market experiments.
    * `analyze_synthetic_data_experiment_results.py`: Post-processing and visualization for synthetic experiments.
    * `market_data_cnn_lstm_pretraining.py`: Script for pretraining the CNN-LSTM backbone on market data.
    * `market_data_experiments.py`: Main script for Intel (INTC) market data experiments.
    * `synthetic_study_analysis.py`: Analysis script for synthetic study results (RelMAE plots, rankings).
    * `synthetic_study_experiment.py`: Main script for synthetic data experiments.
    * `utils_experiment_analysis.py`: Unified framework for analyzing experiment results.
* `docs/`: Additional documentation and guides.
* `tests/`: Test suite for the project.
    * `integration/`: Integration tests and full reproduction scripts.
    * `unit/`: Unit tests for package import and basic functionality.
* `environment.yml`: Conda environment specification file.
* `Dockerfile`: Docker container build configuration.
* `CITATION.cff`: Citation metadata.
* `CONTRIBUTING.md`: Development and testing guidelines.
* `credentials.example.json`: Template for GCP credentials.
* `LICENSE`: License text.
* `pyproject.toml`: Build system requirements and dependencies.
* `README.md`: The file you are reading.
* `setup.py`: Package installation script.

## 📝 Code Style

*   Follow standard Python PEP 8 guidelines.
*   Ensure all new tests are properly marked as `lightweight` or `comprehensive`.
## 📖 Documentation Standard

We strictly use **NumPy-style docstrings** for all code documentation. This ensures compatibility with our automated documentation system (Sphinx).

### Key Rules
1.  **Format:** Use [NumPy Style](https://numpydoc.readthedocs.io/en/latest/format.html).
2.  **Type Hints:** Place type hints in the function signature (Python 3.9+ standard), NOT in the docstring.
3.  **Verbosity:** better to be verbose than vague. Explain *why*, not just *what*.

### 📋 Templates (Copy & Paste)

#### Function Template
```python
def function_name(param1: int, param2: str = "default") -> bool:
    """
    Short summary of what the function does.

    Extended description including implementation details, edge cases,
    or mathematical formulas if relevant.

    Parameters
    ----------
    param1
        Description of the first parameter.
    param2
        Description of the second parameter. 

    Returns
    -------
    bool
        True if successful, False otherwise.

    Raises
    ------
    ValueError
        If param1 is negative.
    """
    pass
```

#### Class Template
```python
class ClassName:
    """
    Short summary of the class purpose.

    Attributes
    ----------
    attr1 : int
        Description of attr1.
    attr2 : str
        Description of attr2.

    Methods
    -------
    method_name(param1)
        Description of what the method does.
    """

    def __init__(self, attr1: int):
        self.attr1 = attr1
```
