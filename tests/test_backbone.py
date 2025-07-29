"""
Unit tests for TTSTL Backbone components
"""

import unittest
import numpy as np
import torch
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ttstl.backbone import CNNLSTM, CNNLSTMDataset, Backbone

class TestBackbone(unittest.TestCase):
    """Test the abstract Backbone class"""
    
    def test_backbone_abstract(self):
        """Test that Backbone cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            Backbone()

class TestCNNLSTM(unittest.TestCase):
    """Test the CNNLSTM implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 10
        self.hidden_dim = 32
        self.output_dim = 3
        self.model = CNNLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.output_dim, self.output_dim)
        
    def test_forward_pass(self):
        """Test forward pass with dummy data"""
        batch_size = 8
        seq_length = 50
        
        # Create dummy input
        x = torch.randn(batch_size, seq_length, self.input_dim)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        expected_shape = (batch_size, self.output_dim)
        self.assertEqual(output.shape, expected_shape)
        
    def test_model_parameters(self):
        """Test that model has trainable parameters"""
        params = list(self.model.parameters())
        self.assertGreater(len(params), 0)
        
        # Check that parameters require gradients
        for param in params:
            self.assertTrue(param.requires_grad)

class TestCNNLSTMDataset(unittest.TestCase):
    """Test the CNNLSTMDataset class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create dummy data
        n_samples = 1000
        n_features = 5
        n_targets = 3
        
        self.features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.targets = pd.DataFrame(
            np.random.randn(n_samples, n_targets),
            columns=[f'target_{i}' for i in range(n_targets)]
        )
        self.sequence_length = 50
        
    def test_dataset_creation(self):
        """Test dataset creation"""
        dataset = CNNLSTMDataset(
            features=self.features,
            targets=self.targets,
            sequence_length=self.sequence_length
        )
        
        # Check dataset length
        expected_length = len(self.features) - self.sequence_length
        self.assertEqual(len(dataset), expected_length)
        
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method"""
        dataset = CNNLSTMDataset(
            features=self.features,
            targets=self.targets,
            sequence_length=self.sequence_length
        )
        
        # Get first item
        sequence, target = dataset[0]
        
        # Check shapes
        self.assertEqual(sequence.shape, (self.sequence_length, len(self.features.columns)))
        self.assertEqual(target.shape, (len(self.targets.columns),))
        
        # Check types
        self.assertIsInstance(sequence, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
        
    def test_dataset_edge_cases(self):
        """Test dataset edge cases"""
        # Test with minimal data
        small_features = self.features[:self.sequence_length + 1]
        small_targets = self.targets[:self.sequence_length + 1]
        
        dataset = CNNLSTMDataset(
            features=small_features,
            targets=small_targets,
            sequence_length=self.sequence_length
        )
        
        self.assertEqual(len(dataset), 1)

class TestModelTraining(unittest.TestCase):
    """Test model training functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a small model for testing
        self.model = CNNLSTM(
            input_dim=5,
            hidden_dim=16,
            output_dim=1,
            num_layers=1
        )
        
        # Create dummy dataset
        n_samples = 200
        features = pd.DataFrame(np.random.randn(n_samples, 5))
        targets = pd.DataFrame(np.random.randn(n_samples, 1))
        
        self.dataset = CNNLSTMDataset(
            features=features,
            targets=targets,
            sequence_length=20
        )
        
    def test_model_fit_method_exists(self):
        """Test that model has a fit method"""
        self.assertTrue(hasattr(self.model, 'fit'))
        
    def test_model_predict_method_exists(self):
        """Test that model has a predict method"""
        self.assertTrue(hasattr(self.model, 'predict'))
        
    def test_model_save_load_methods_exist(self):
        """Test that model has save and load methods"""
        self.assertTrue(hasattr(self.model, 'save'))
        self.assertTrue(hasattr(CNNLSTM, 'load'))

class TestModelIO(unittest.TestCase):
    """Test model save/load functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = CNNLSTM(
            input_dim=10,
            hidden_dim=32,
            output_dim=3
        )
        self.test_model_path = "/tmp/test_model.pth"
        
    def tearDown(self):
        """Clean up test files"""
        import os
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)
            
    def test_save_method_exists(self):
        """Test that save method exists and is callable"""
        self.assertTrue(hasattr(self.model, 'save'))
        self.assertTrue(callable(getattr(self.model, 'save')))
        
    def test_load_method_exists(self):
        """Test that load class method exists"""
        self.assertTrue(hasattr(CNNLSTM, 'load'))
        self.assertTrue(callable(getattr(CNNLSTM, 'load')))

if __name__ == '__main__':
    unittest.main()
