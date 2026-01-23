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
