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

**Run everything:**
```bash
pytest
```

## 📂 Project Structure

*   `src/ttstl/`: Core package code.
*   `tests/`: Test suite.
    *   `unit/`: Unit tests and fast checks.
    *   `integration/`: Long-running integration tests and paper reproduction scripts.
*   `scripts/`: High-level entry points for experiments.

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
