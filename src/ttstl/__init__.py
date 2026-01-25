"""
Two Time Scale Transfer Learning (TTSTL) package
"""

__author__ = "Juan Diego Mejia Becerra"
__email__ = "juanmejia.7565@gmail.com"
__version__ = "0.1.0"

from .adapter import Adapter
from .architectures import CNNLSTM

# Import main classes and functions
from .backbone import Backbone
from .datasets import CNNLSTMDataset
from .interface import Interface

# Import core TTSTL components
from .orchestrator import Orchestrator, OrchestratorConfig
from .utils import (
    download_from_bucket,
    get_dataset_cnn_lstm,
    get_storage_client,
    import_to_pandas,
    preprocess_mbo_data_cnn_lstm,
)

# Define what gets imported with "from ttstl import *"
__all__ = [
    # TTSTL core components
    "Backbone",
    "Interface",
    "Adapter",
    "Orchestrator",
    "OrchestratorConfig",
    # Backbone components
    "CNNLSTM",
    "CNNLSTMDataset",
    # Utility functions
    "download_from_bucket",
    "import_to_pandas",
    "get_storage_client",
    "preprocess_mbo_data_cnn_lstm",
    "get_dataset_cnn_lstm",
]
