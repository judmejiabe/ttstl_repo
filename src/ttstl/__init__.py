"""
Two Time Scale Transfer Learning (TTSTL) package
"""

__author__ = 'Juan Diego Mejia Becerra'
__email__ = 'juanmejia.7565@gmail.com'
__version__ = "0.1.0"

# Import main classes and functions
from .backbone import Backbone
from .interface import Interface
# from .adapter import Adapter  # Not implemented yet
from .architectures import CNNLSTM
from .datasets import CNNLSTMDataset
from .utils import (
    download_from_bucket,
    import_to_pandas,
    get_storage_client,
    preprocess_mbo_data_cnn_lstm,
    get_dataset_cnn_lstm
)
# Import core TTSTL components when implemented
# from .orchestrator import *  # Add when you implement this

# Define what gets imported with "from ttstl import *"
__all__ = [
    # TTSTL core components
    'Backbone',
    'Interface', 
    # 'Adapter',  # Not implemented yet
    
    # Backbone components
    'CNNLSTM',
    'CNNLSTMDataset',
    
    # Utility functions
    'download_from_bucket',
    'import_to_pandas',
    'get_storage_client',
    'preprocess_mbo_data_cnn_lstm',
    'get_dataset_cnn_lstm',
    
    # Future components (add as you implement)
    # 'TTSTLModel',
    # 'Orchestrator',
]