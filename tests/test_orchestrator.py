"""
Tests for the TTSTL Orchestrator module.

This test suite validates the orchestrator's ability to coordinate
backbone, interface, and adapter modules while managing artifacts
and maintaining reproducibility.
"""

import json
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from ttstl.orchestrator import Orchestrator, OrchestratorConfig


class TestOrchestratorConfig:
    """Test the OrchestratorConfig dataclass."""
    
    def test_orchestrator_config_defaults(self):
        """Test that default values are set correctly."""
        cfg = OrchestratorConfig(
            ticker="MSFT",
            bdelta=100000.0,
            adelta=10000.0,
            sdelta=1000.0,
            seed=42,
            session_start=0.0,
            session_finish=1000000.0
        )
        
        # Check defaults are set in __post_init__
        assert cfg.quantile == 0.005
        assert cfg.parsimonious_mode is False
        assert cfg.batch_size == 32
        assert cfg.epochs == 12
        assert cfg.selection_threshold == 0.15
        assert cfg.tolerance_threshold == 1e-6
        assert cfg.ols_significance_level == 0.05
        assert cfg.validation_proportion == 0.1
        assert cfg.buffer_proportion == 0.05
        
        # Check default grids are set
        assert cfg.alpha_enet_grid == [0.1, 0.5, 0.9]
        assert len(cfg.lambda_enet_grid) == 20  # logspace(-4, 0, 20)
        assert cfg.tau_adalasso_grid == [0.5, 1.0, 1.5, 2.0]
        assert len(cfg.lambda_adalasso_grid) == 20
        
    def test_orchestrator_config_custom_grids(self):
        """Test that custom grids override defaults."""
        custom_alpha = [0.2, 0.6]
        custom_lambda = [0.1, 0.5, 1.0]
        
        cfg = OrchestratorConfig(
            ticker="AAPL",
            bdelta=50000.0,
            adelta=5000.0,
            sdelta=500.0,
            seed=123,
            session_start=100.0,
            session_finish=500000.0,
            alpha_enet_grid=custom_alpha,
            lambda_enet_grid=custom_lambda
        )
        
        assert cfg.alpha_enet_grid == custom_alpha
        assert cfg.lambda_enet_grid == custom_lambda
        # Defaults should still be set for others
        assert cfg.tau_adalasso_grid == [0.5, 1.0, 1.5, 2.0]


class TestOrchestrator:
    """Test the Orchestrator class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return OrchestratorConfig(
            ticker="TEST",
            bdelta=100000.0,
            adelta=10000.0,
            sdelta=1000.0,
            seed=42,
            session_start=0.0,
            session_finish=1000000.0,
            epochs=2,  # Reduced for testing
            run_id="test123"
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample event and physical time data."""
        np.random.seed(42)
        
        # Create event-driven data
        n_events = 1000
        event_data = pd.DataFrame({
            'timestamp': np.sort(np.random.uniform(0, 1000000, n_events)),
            'price': np.random.uniform(100, 110, n_events),
            'volume': np.random.randint(1, 1000, n_events),
            'side': np.random.choice(['bid', 'ask'], n_events)
        })
        
        # Create physical time data
        timestamps = np.arange(1000, 1000000, 1000)  # Every 1000 time units
        n_phys = len(timestamps)
        phys_data = pd.DataFrame({
            'timestamp': timestamps,
            'micro': np.random.uniform(100, 110, n_phys),
            'bid_price': np.random.uniform(99, 109, n_phys),
            'ask_price': np.random.uniform(101, 111, n_phys),
            'spread': np.random.uniform(0.01, 0.1, n_phys),
            'volume': np.random.randint(100, 10000, n_phys)
        })
        
        return event_data, phys_data
    
    def test_orchestrator_initialization(self, sample_config):
        """Test orchestrator initialization creates proper directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            
            # Mock the component initialization to avoid actual model creation
            with patch('ttstl.orchestrator.CNNLSTM') as mock_cnn, \
                 patch('ttstl.orchestrator.Backbone') as mock_backbone, \
                 patch('ttstl.orchestrator.Interface') as mock_interface, \
                 patch('ttstl.orchestrator.Adapter') as mock_adapter:
                
                orchestrator = Orchestrator(runs_root, sample_config)
                
                # Check run directory was created
                assert orchestrator.run_dir.exists()
                assert orchestrator.run_slug.startswith("2025-")
                assert "TEST" in orchestrator.run_slug
                assert "test123" in orchestrator.run_slug
                
                # Check config.yaml was created
                config_file = orchestrator.run_dir / "config.yaml"
                assert config_file.exists()
                
                with open(config_file, 'r') as f:
                    saved_config = yaml.safe_load(f)
                    assert saved_config['ticker'] == 'TEST'
                    assert saved_config['seed'] == 42
                
                # Check manifest.json was created
                manifest_file = orchestrator.run_dir / "manifest.json"
                assert manifest_file.exists()
                
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                    assert manifest['run_slug'] == orchestrator.run_slug
                    assert manifest['ticker'] == 'TEST'
                    assert manifest['seed'] == 42
                    assert 'created_at' in manifest
                    assert 'code_provenance' in manifest
                    assert 'environment' in manifest
                    assert manifest['artifacts'] == []
                
                # Check backbone index is initialized
                assert orchestrator._bb_idx == -1
    
    def test_run_slug_generation(self, sample_config):
        """Test run slug generation format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            
            with patch('ttstl.orchestrator.CNNLSTM'), \
                 patch('ttstl.orchestrator.Backbone'), \
                 patch('ttstl.orchestrator.Interface'), \
                 patch('ttstl.orchestrator.Adapter'):
                
                orchestrator = Orchestrator(runs_root, sample_config)
                
                # Check slug format: [Date]_[TICKER]_bdelta[ΔB]_adelta[ΔA]_sdelta[ΔS]_seed[seed]_id[ID]
                slug = orchestrator.run_slug
                parts = slug.split('_')
                
                assert len(parts) == 7
                assert parts[0].startswith('2025-')  # Date
                assert parts[1] == 'TEST'            # Ticker
                assert parts[2] == 'bdelta100000'    # bdelta
                assert parts[3] == 'adelta10000'     # adelta
                assert parts[4] == 'sdelta1000'      # sdelta
                assert parts[5] == 'seed42'          # seed
                assert parts[6] == 'idtest123'       # id
    
    def test_time_bar_column_addition(self, sample_config, sample_data):
        """Test that time bar columns are added correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            event_df, phys_df = sample_data
            
            with patch('ttstl.orchestrator.CNNLSTM'), \
                 patch('ttstl.orchestrator.Backbone'), \
                 patch('ttstl.orchestrator.Interface'), \
                 patch('ttstl.orchestrator.Adapter'):
                
                orchestrator = Orchestrator(runs_root, sample_config)
                
                # Test time bar addition
                processed_event = orchestrator._add_time_bar_columns(event_df.copy())
                processed_phys = orchestrator._add_time_bar_columns(phys_df.copy())
                
                # Check columns were added
                assert 'backbone_bar' in processed_event.columns
                assert 'adapter_bar' in processed_event.columns
                assert 'backbone_bar' in processed_phys.columns
                assert 'adapter_bar' in processed_phys.columns
                
                # Check values are correct
                expected_bb_bar = (event_df['timestamp'] // sample_config.bdelta).astype(int)
                expected_ad_bar = (event_df['timestamp'] // sample_config.adelta).astype(int)
                
                # Reset index to ensure proper comparison
                expected_bb_bar.name = 'backbone_bar'
                expected_ad_bar.name = 'adapter_bar'
                
                pd.testing.assert_series_equal(processed_event['backbone_bar'], expected_bb_bar)
                pd.testing.assert_series_equal(processed_event['adapter_bar'], expected_ad_bar)
    
    def test_global_update_directory_creation(self, sample_config, sample_data):
        """Test that global_update creates proper directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            event_df, phys_df = sample_data
            
            with patch('ttstl.orchestrator.CNNLSTM'), \
                 patch('ttstl.orchestrator.Backbone'), \
                 patch('ttstl.orchestrator.Interface'), \
                 patch('ttstl.orchestrator.Adapter'):
                
                orchestrator = Orchestrator(runs_root, sample_config)
                
                # Run global update
                results = orchestrator.global_update(event_df, phys_df)
                
                # Check backbone index was incremented
                assert orchestrator._bb_idx == 0
                
                # Check directory structure was created
                bb_dir = orchestrator.run_dir / "bb_000" / "global_update"
                assert bb_dir.exists()
                
                expected_dirs = [
                    "backbone/architecture/checkpoints",
                    "backbone/stationarity",
                    "backbone/correlation", 
                    "interface",
                    "adapter"
                ]
                
                for subdir in expected_dirs:
                    assert (bb_dir / subdir).exists()
                    assert (bb_dir / subdir).is_dir()
                
                # Check results structure
                assert 'backbone_bar' in results
                assert 'backbone_results' in results
                assert 'interface_results' in results
                assert 'adapter_results' in results
                assert 'artifact_paths' in results
                assert results['backbone_bar'] == 0
    
    def test_local_adapt_requires_global_update(self, sample_config, sample_data):
        """Test that local_adapt fails if global_update hasn't been run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            event_df, phys_df = sample_data
            
            with patch('ttstl.orchestrator.CNNLSTM'), \
                 patch('ttstl.orchestrator.Backbone'), \
                 patch('ttstl.orchestrator.Interface'), \
                 patch('ttstl.orchestrator.Adapter'):
                
                orchestrator = Orchestrator(runs_root, sample_config)
                
                # Try to run local_adapt without global_update
                with pytest.raises(RuntimeError, match="Cannot run local_adapt before global_update"):
                    orchestrator.local_adapt(event_df, phys_df)
    
    def test_local_adapt_after_global_update(self, sample_config, sample_data):
        """Test that local_adapt works after global_update."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            event_df, phys_df = sample_data
            
            with patch('ttstl.orchestrator.CNNLSTM'), \
                 patch('ttstl.orchestrator.Backbone'), \
                 patch('ttstl.orchestrator.Interface'), \
                 patch('ttstl.orchestrator.Adapter'):
                
                orchestrator = Orchestrator(runs_root, sample_config)
                
                # Run global update first
                global_results = orchestrator.global_update(event_df, phys_df)
                
                # Now run local adapt
                local_results = orchestrator.local_adapt(event_df, phys_df)
                
                # Check local adapt directory structure
                bb_dir = orchestrator.run_dir / "bb_000" / "local_adapt"
                assert bb_dir.exists()
                
                expected_dirs = [
                    "backbone",
                    "interface", 
                    "adapter"
                ]
                
                for subdir in expected_dirs:
                    assert (bb_dir / subdir).exists()
                
                # Check results structure
                assert 'backbone_bar' in local_results
                assert local_results['backbone_bar'] == 0
    
    def test_manifest_updates(self, sample_config, sample_data):
        """Test that manifest is updated correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            event_df, phys_df = sample_data
            
            with patch('ttstl.orchestrator.CNNLSTM'), \
                 patch('ttstl.orchestrator.Backbone'), \
                 patch('ttstl.orchestrator.Interface'), \
                 patch('ttstl.orchestrator.Adapter'):
                
                orchestrator = Orchestrator(runs_root, sample_config)
                
                # Check initial manifest
                manifest_file = orchestrator.run_dir / "manifest.json"
                with open(manifest_file, 'r') as f:
                    initial_manifest = json.load(f)
                    assert len(initial_manifest['artifacts']) == 0
                
                # Run global update
                orchestrator.global_update(event_df, phys_df)
                
                # Check manifest was updated
                with open(manifest_file, 'r') as f:
                    updated_manifest = json.load(f)
                    assert len(updated_manifest['artifacts']) == 1
                    
                    entry = updated_manifest['artifacts'][0]
                    assert entry['backbone_bar'] == 0
                    assert entry['mode'] == 'global_update'
                    assert 'timestamp' in entry
                    assert 'artifact_paths' in entry
                    assert 'summary' in entry
                
                # Run local adapt
                orchestrator.local_adapt(event_df, phys_df)
                
                # Check manifest was updated again
                with open(manifest_file, 'r') as f:
                    final_manifest = json.load(f)
                    assert len(final_manifest['artifacts']) == 2
                    
                    local_entry = final_manifest['artifacts'][1]
                    assert local_entry['backbone_bar'] == 0
                    assert local_entry['mode'] == 'local_adapt'
    
    def test_multiple_backbone_bars(self, sample_config, sample_data):
        """Test processing multiple backbone bars."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            event_df, phys_df = sample_data
            
            with patch('ttstl.orchestrator.CNNLSTM'), \
                 patch('ttstl.orchestrator.Backbone'), \
                 patch('ttstl.orchestrator.Interface'), \
                 patch('ttstl.orchestrator.Adapter'):
                
                orchestrator = Orchestrator(runs_root, sample_config)
                
                # Run multiple global updates
                results1 = orchestrator.global_update(event_df, phys_df)
                results2 = orchestrator.global_update(event_df, phys_df)
                
                assert results1['backbone_bar'] == 0
                assert results2['backbone_bar'] == 1
                assert orchestrator._bb_idx == 1
                
                # Check directories were created
                assert (orchestrator.run_dir / "bb_000" / "global_update").exists()
                assert (orchestrator.run_dir / "bb_001" / "global_update").exists()
    
    def test_missing_timestamp_column(self, sample_config):
        """Test error handling for missing timestamp column."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runs_root = Path(temp_dir)
            
            # Create data without timestamp column
            bad_df = pd.DataFrame({
                'price': [100, 101, 102],
                'volume': [1000, 1100, 1200]
            })
            
            with patch('ttstl.orchestrator.CNNLSTM'), \
                 patch('ttstl.orchestrator.Backbone'), \
                 patch('ttstl.orchestrator.Interface'), \
                 patch('ttstl.orchestrator.Adapter'):
                
                orchestrator = Orchestrator(runs_root, sample_config)
                
                with pytest.raises(ValueError, match="DataFrame must contain 'timestamp' column"):
                    orchestrator._add_time_bar_columns(bad_df)


if __name__ == "__main__":
    pytest.main([__file__])
