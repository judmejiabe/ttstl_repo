import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "lightweight: mark test as lightweight (fast)")
    config.addinivalue_line(
        "markers", "comprehensive: mark test as comprehensive (slow)"
    )
