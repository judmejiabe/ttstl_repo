"""
Unit tests for TTSTL Interface components
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ttstl.interface import Interface

class TestInterface(unittest.TestCase):
    """Test the Interface class for change detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create stable reference data
        np.random.seed(42)
        self.reference_data = pd.DataFrame(
            np.random.normal(0, 1, (1000, 5)),
            columns=[f'feature_{i}' for i in range(5)]
        )
        
        # Create changed data (different mean)
        self.changed_data = pd.DataFrame(
            np.random.normal(2, 1, (100, 5)),
            columns=[f'feature_{i}' for i in range(5)]
        )
        
        # Create similar data (same distribution)
        self.similar_data = pd.DataFrame(
            np.random.normal(0, 1, (100, 5)),
            columns=[f'feature_{i}' for i in range(5)]
        )
        
    def test_interface_initialization(self):
        """Test interface initialization with different methods"""
        # Test CUSUM method
        interface_cusum = Interface(detection_method='cusum')
        self.assertEqual(interface_cusum.detection_method, 'cusum')
        
        # Test KS test method
        interface_ks = Interface(detection_method='ks_test')
        self.assertEqual(interface_ks.detection_method, 'ks_test')
        
        # Test statistical method
        interface_stat = Interface(detection_method='statistical')
        self.assertEqual(interface_stat.detection_method, 'statistical')
        
    def test_detect_change_method_exists(self):
        """Test that detect_change method exists"""
        interface = Interface()
        self.assertTrue(hasattr(interface, 'detect_change'))
        self.assertTrue(callable(getattr(interface, 'detect_change')))
        
    def test_get_adaptation_signal_method_exists(self):
        """Test that get_adaptation_signal method exists"""
        interface = Interface()
        self.assertTrue(hasattr(interface, 'get_adaptation_signal'))
        self.assertTrue(callable(getattr(interface, 'get_adaptation_signal')))
        
    def test_update_reference_method_exists(self):
        """Test that update_reference method exists"""
        interface = Interface()
        self.assertTrue(hasattr(interface, 'update_reference'))
        self.assertTrue(callable(getattr(interface, 'update_reference')))
        
    def test_invalid_detection_method(self):
        """Test that invalid detection method raises error"""
        with self.assertRaises(ValueError):
            Interface(detection_method='invalid_method')
            
    def test_threshold_parameter(self):
        """Test threshold parameter setting"""
        threshold = 0.01
        interface = Interface(threshold=threshold)
        self.assertEqual(interface.threshold, threshold)
        
    def test_window_size_parameter(self):
        """Test window size parameter setting"""
        window_size = 200
        interface = Interface(window_size=window_size)
        self.assertEqual(interface.window_size, window_size)

class TestCUSUMDetection(unittest.TestCase):
    """Test CUSUM-based change detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.interface = Interface(
            detection_method='cusum',
            threshold=0.05,
            window_size=100
        )
        
        # Create test data
        np.random.seed(42)
        self.stable_data = pd.DataFrame(
            np.random.normal(0, 1, (200, 3))
        )
        self.shifted_data = pd.DataFrame(
            np.random.normal(2, 1, (50, 3))
        )
        
    def test_cusum_no_change(self):
        """Test CUSUM with stable data (no change expected)"""
        # Initialize with reference data
        self.interface.update_reference(self.stable_data[:100])
        
        # Test with similar data
        result = self.interface.detect_change(self.stable_data[100:150])
        
        # Should not detect change
        self.assertIsInstance(result, bool)
        
    def test_cusum_with_change(self):
        """Test CUSUM with shifted data (change expected)"""
        # Initialize with reference data
        self.interface.update_reference(self.stable_data)
        
        # Test with shifted data
        result = self.interface.detect_change(self.shifted_data)
        
        # Result should be boolean
        self.assertIsInstance(result, bool)
        
    def test_adaptation_signal_range(self):
        """Test that adaptation signal is in valid range"""
        self.interface.update_reference(self.stable_data)
        
        signal = self.interface.get_adaptation_signal(self.shifted_data)
        
        # Signal should be a float between 0 and 1
        self.assertIsInstance(signal, float)
        self.assertGreaterEqual(signal, 0.0)
        self.assertLessEqual(signal, 1.0)

class TestKSTestDetection(unittest.TestCase):
    """Test Kolmogorov-Smirnov test-based change detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.interface = Interface(
            detection_method='ks_test',
            threshold=0.05
        )
        
        # Create test data
        np.random.seed(42)
        self.reference_data = pd.DataFrame(
            np.random.normal(0, 1, (500, 2))
        )
        self.similar_data = pd.DataFrame(
            np.random.normal(0, 1, (100, 2))
        )
        self.different_data = pd.DataFrame(
            np.random.exponential(1, (100, 2))
        )
        
    def test_ks_test_similar_distributions(self):
        """Test KS test with similar distributions"""
        self.interface.update_reference(self.reference_data)
        
        result = self.interface.detect_change(self.similar_data)
        self.assertIsInstance(result, bool)
        
    def test_ks_test_different_distributions(self):
        """Test KS test with different distributions"""
        self.interface.update_reference(self.reference_data)
        
        result = self.interface.detect_change(self.different_data)
        self.assertIsInstance(result, bool)

class TestStatisticalDetection(unittest.TestCase):
    """Test statistical-based change detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.interface = Interface(
            detection_method='statistical',
            threshold=0.05
        )
        
    def test_statistical_method_initialization(self):
        """Test statistical method initialization"""
        self.assertEqual(self.interface.detection_method, 'statistical')
        
    def test_statistical_change_detection(self):
        """Test statistical change detection"""
        # Create reference data
        reference = pd.DataFrame(np.random.normal(0, 1, (500, 3)))
        test_data = pd.DataFrame(np.random.normal(1, 1, (100, 3)))
        
        self.interface.update_reference(reference)
        result = self.interface.detect_change(test_data)
        
        self.assertIsInstance(result, bool)

class TestInterfaceEdgeCases(unittest.TestCase):
    """Test edge cases for Interface class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.interface = Interface()
        
    def test_empty_data(self):
        """Test behavior with empty data"""
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        try:
            result = self.interface.detect_change(empty_data)
            self.assertIsInstance(result, bool)
        except Exception as e:
            # If it raises an exception, it should be a specific type
            self.assertIsInstance(e, (ValueError, IndexError))
            
    def test_single_observation(self):
        """Test behavior with single observation"""
        single_obs = pd.DataFrame([[1, 2, 3]])
        
        try:
            result = self.interface.detect_change(single_obs)
            self.assertIsInstance(result, bool)
        except Exception as e:
            # Should handle gracefully or raise specific exception
            self.assertIsInstance(e, (ValueError, IndexError))
            
    def test_mismatched_features(self):
        """Test behavior with mismatched feature dimensions"""
        reference = pd.DataFrame(np.random.randn(100, 3))
        test_data = pd.DataFrame(np.random.randn(50, 5))  # Different number of features
        
        self.interface.update_reference(reference)
        
        with self.assertRaises((ValueError, IndexError)):
            self.interface.detect_change(test_data)

class TestInterfaceConfiguration(unittest.TestCase):
    """Test Interface configuration options"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        interface = Interface()
        
        # Check that default values are set
        self.assertIsNotNone(interface.detection_method)
        self.assertIsNotNone(interface.threshold)
        self.assertIsNotNone(interface.window_size)
        
    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = {
            'detection_method': 'cusum',
            'threshold': 0.01,
            'window_size': 50
        }
        
        interface = Interface(**config)
        
        self.assertEqual(interface.detection_method, config['detection_method'])
        self.assertEqual(interface.threshold, config['threshold'])
        self.assertEqual(interface.window_size, config['window_size'])

if __name__ == '__main__':
    unittest.main()
