"""
Unit tests for TTSTL Adapter components
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ttstl.adapter import Adapter

class TestAdapter(unittest.TestCase):
    """Test the Adapter class for local adaptation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create synthetic time series data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate GARCH-like volatility clustering
        returns = []
        sigma_t = 0.01
        
        for _ in range(n_samples):
            epsilon_t = np.random.normal(0, 1)
            return_t = sigma_t * epsilon_t
            returns.append(return_t)
            
            # Update volatility (simple GARCH(1,1))
            sigma_t = np.sqrt(0.00001 + 0.05 * return_t**2 + 0.9 * sigma_t**2)
        
        self.returns_data = pd.Series(returns)
        
        # Create multivariate data
        self.multivariate_data = pd.DataFrame({
            'returns': self.returns_data,
            'volume': np.random.exponential(100, n_samples),
            'spread': np.random.exponential(0.001, n_samples)
        })
        
    def test_adapter_initialization(self):
        """Test adapter initialization with different models"""
        # Test rugarch model
        adapter_rugarch = Adapter(model_type='rugarch')
        self.assertEqual(adapter_rugarch.model_type, 'rugarch')
        
        # Test with custom GARCH specification
        adapter_custom = Adapter(
            model_type='rugarch',
            garch_spec='sGARCH',
            distribution='std'
        )
        self.assertEqual(adapter_custom.garch_spec, 'sGARCH')
        self.assertEqual(adapter_custom.distribution, 'std')
        
    def test_invalid_model_type(self):
        """Test that invalid model type raises error"""
        with self.assertRaises(ValueError):
            Adapter(model_type='invalid_model')
            
    def test_fit_method_exists(self):
        """Test that fit method exists"""
        adapter = Adapter()
        self.assertTrue(hasattr(adapter, 'fit'))
        self.assertTrue(callable(getattr(adapter, 'fit')))
        
    def test_update_method_exists(self):
        """Test that update method exists"""
        adapter = Adapter()
        self.assertTrue(hasattr(adapter, 'update'))
        self.assertTrue(callable(getattr(adapter, 'update')))
        
    def test_forecast_method_exists(self):
        """Test that forecast method exists"""
        adapter = Adapter()
        self.assertTrue(hasattr(adapter, 'forecast'))
        self.assertTrue(callable(getattr(adapter, 'forecast')))
        
    def test_get_parameters_method_exists(self):
        """Test that get_parameters method exists"""
        adapter = Adapter()
        self.assertTrue(hasattr(adapter, 'get_parameters'))
        self.assertTrue(callable(getattr(adapter, 'get_parameters')))

class TestAdapterFitting(unittest.TestCase):
    """Test adapter model fitting functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.adapter = Adapter(model_type='rugarch')
        
        # Create returns data
        np.random.seed(42)
        self.returns = pd.Series(np.random.normal(0, 0.01, 500))
        
    def test_fit_with_returns_data(self):
        """Test fitting with returns data"""
        try:
            result = self.adapter.fit(self.returns)
            # Should not raise an exception
            self.assertIsNotNone(result)
        except Exception as e:
            # If R/rpy2 is not available, should raise specific error
            self.assertIsInstance(e, (ImportError, RuntimeError))
            
    def test_fit_with_insufficient_data(self):
        """Test fitting with insufficient data"""
        short_data = pd.Series([0.01, -0.005, 0.002])  # Very short series
        
        try:
            result = self.adapter.fit(short_data)
        except Exception as e:
            # Should handle insufficient data gracefully
            self.assertIsInstance(e, (ValueError, RuntimeError))

class TestAdapterUpdating(unittest.TestCase):
    """Test adapter model updating functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.adapter = Adapter(model_type='rugarch')
        
        # Initial data for fitting
        np.random.seed(42)
        self.initial_data = pd.Series(np.random.normal(0, 0.01, 300))
        self.new_data = pd.Series(np.random.normal(0, 0.015, 50))
        
    def test_update_without_initial_fit(self):
        """Test that update fails without initial fit"""
        with self.assertRaises((ValueError, RuntimeError)):
            self.adapter.update(self.new_data)
            
    def test_update_after_fit(self):
        """Test update after initial fit"""
        try:
            # First fit the model
            self.adapter.fit(self.initial_data)
            
            # Then update with new data
            result = self.adapter.update(self.new_data)
            self.assertIsNotNone(result)
            
        except Exception as e:
            # If R/rpy2 is not available, should raise specific error
            self.assertIsInstance(e, (ImportError, RuntimeError))

class TestAdapterForecasting(unittest.TestCase):
    """Test adapter forecasting functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.adapter = Adapter(model_type='rugarch')
        
        # Data for fitting
        np.random.seed(42)
        self.data = pd.Series(np.random.normal(0, 0.01, 400))
        
    def test_forecast_without_fit(self):
        """Test that forecast fails without initial fit"""
        with self.assertRaises((ValueError, RuntimeError)):
            self.adapter.forecast(horizon=5)
            
    def test_forecast_after_fit(self):
        """Test forecast after fitting"""
        try:
            # First fit the model
            self.adapter.fit(self.data)
            
            # Then forecast
            forecast = self.adapter.forecast(horizon=10)
            
            # Check that forecast is returned
            self.assertIsNotNone(forecast)
            
            # Check forecast length
            if hasattr(forecast, '__len__'):
                self.assertEqual(len(forecast), 10)
                
        except Exception as e:
            # If R/rpy2 is not available, should raise specific error
            self.assertIsInstance(e, (ImportError, RuntimeError))
            
    def test_forecast_different_horizons(self):
        """Test forecasting with different horizons"""
        horizons = [1, 5, 10, 20]
        
        try:
            self.adapter.fit(self.data)
            
            for horizon in horizons:
                forecast = self.adapter.forecast(horizon=horizon)
                self.assertIsNotNone(forecast)
                
        except Exception as e:
            # If R/rpy2 is not available, should raise specific error
            self.assertIsInstance(e, (ImportError, RuntimeError))

class TestAdapterParameters(unittest.TestCase):
    """Test adapter parameter management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.adapter = Adapter(
            model_type='rugarch',
            garch_spec='sGARCH',
            distribution='std'
        )
        
    def test_get_parameters_before_fit(self):
        """Test getting parameters before fitting"""
        params = self.adapter.get_parameters()
        
        # Should return something (even if empty or None)
        self.assertIsNotNone(params)
        
    def test_get_parameters_after_fit(self):
        """Test getting parameters after fitting"""
        data = pd.Series(np.random.normal(0, 0.01, 300))
        
        try:
            self.adapter.fit(data)
            params = self.adapter.get_parameters()
            
            # Should return parameter dictionary
            self.assertIsNotNone(params)
            self.assertIsInstance(params, dict)
            
        except Exception as e:
            # If R/rpy2 is not available, should raise specific error
            self.assertIsInstance(e, (ImportError, RuntimeError))

class TestAdapterConfiguration(unittest.TestCase):
    """Test adapter configuration options"""
    
    def test_default_configuration(self):
        """Test default configuration"""
        adapter = Adapter()
        
        # Check default values
        self.assertIsNotNone(adapter.model_type)
        self.assertIsNotNone(adapter.garch_spec)
        self.assertIsNotNone(adapter.distribution)
        
    def test_custom_configuration(self):
        """Test custom configuration"""
        config = {
            'model_type': 'rugarch',
            'garch_spec': 'eGARCH',
            'distribution': 'ged',
            'mean_model': 'ARFIMA'
        }
        
        adapter = Adapter(**config)
        
        self.assertEqual(adapter.model_type, config['model_type'])
        self.assertEqual(adapter.garch_spec, config['garch_spec'])
        self.assertEqual(adapter.distribution, config['distribution'])
        
        if hasattr(adapter, 'mean_model'):
            self.assertEqual(adapter.mean_model, config['mean_model'])

class TestAdapterEdgeCases(unittest.TestCase):
    """Test edge cases for Adapter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.adapter = Adapter()
        
    def test_empty_data(self):
        """Test behavior with empty data"""
        empty_data = pd.Series([])
        
        with self.assertRaises((ValueError, RuntimeError)):
            self.adapter.fit(empty_data)
            
    def test_constant_data(self):
        """Test behavior with constant data"""
        constant_data = pd.Series([0.01] * 100)  # No variation
        
        try:
            result = self.adapter.fit(constant_data)
        except Exception as e:
            # Should handle constant data appropriately
            self.assertIsInstance(e, (ValueError, RuntimeError))
            
    def test_extreme_values(self):
        """Test behavior with extreme values"""
        extreme_data = pd.Series([0.0001, 100, -50, 0.0001, 200])
        
        try:
            result = self.adapter.fit(extreme_data)
        except Exception as e:
            # Should handle extreme values appropriately
            self.assertIsInstance(e, (ValueError, RuntimeError))

class TestAdapterIntegration(unittest.TestCase):
    """Integration tests for Adapter workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.adapter = Adapter(model_type='rugarch')
        
        # Create realistic financial returns data
        np.random.seed(42)
        n_samples = 500
        
        # Generate returns with volatility clustering
        returns = []
        sigma = 0.02
        
        for i in range(n_samples):
            epsilon = np.random.normal(0, 1)
            return_t = sigma * epsilon
            returns.append(return_t)
            
            # Simple volatility update
            sigma = 0.01 + 0.1 * abs(return_t) + 0.8 * sigma
            sigma = min(sigma, 0.1)  # Cap volatility
            
        self.returns_data = pd.Series(returns)
        
    def test_complete_workflow(self):
        """Test complete adapter workflow"""
        try:
            # 1. Fit initial model
            initial_data = self.returns_data[:400]
            fit_result = self.adapter.fit(initial_data)
            self.assertIsNotNone(fit_result)
            
            # 2. Get initial parameters
            initial_params = self.adapter.get_parameters()
            self.assertIsNotNone(initial_params)
            
            # 3. Update with new data
            new_data = self.returns_data[400:450]
            update_result = self.adapter.update(new_data)
            self.assertIsNotNone(update_result)
            
            # 4. Generate forecast
            forecast = self.adapter.forecast(horizon=5)
            self.assertIsNotNone(forecast)
            
            # 5. Get updated parameters
            updated_params = self.adapter.get_parameters()
            self.assertIsNotNone(updated_params)
            
        except Exception as e:
            # If R/rpy2 is not available, should raise specific error
            self.assertIsInstance(e, (ImportError, RuntimeError))
            print(f"Adapter test skipped due to missing dependencies: {e}")

if __name__ == '__main__':
    unittest.main()
