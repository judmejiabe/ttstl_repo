"""
Two Time Scale Transfer Learning (TTSTL) package
"""

__author__ = 'Juan Diego Mejia Becerra'
__email__ = 'juanmejia.7565@gmail.com'
__version__ = "0.1.0"

# Import main classes and functions
from .backbone import CNNLSTM, CNNLSTMDataset, Backbone
from .utils import (
    download_from_bucket,
    import_to_pandas,
    preprocess_mbo_data_cnn_lstm,
    get_datasets_cnn_lstm
)
# Import core TTSTL components when implemented
# from .adapter import *  # Add when you implement this
# from .interface import *  # Add when you implement this
# from .orchestrator import *  # Add when you implement this

# Define what gets imported with "from ttstl import *"
__all__ = [
    # Backbone components
    'CNNLSTM',
    'CNNLSTMDataset', 
    'Backbone',
    
    # Utility functions
    'download_from_bucket',
    'import_to_pandas',
    'preprocess_mbo_data_cnn_lstm',
    'get_datasets_cnn_lstm',
    
    # TTSTL core components (add as you implement)
    # 'TTSTLModel',
    # 'Adapter',
    # 'Interface',
    # 'Orchestrator',
]