#!/usr/bin/env python3
"""
Demo script for testing the enhanced TTSTL Orchestrator global_update method with real data.
Tests data partitioning into backbone and adapter bars, then runs mock backbone global_update.
"""

import tempfile
import sys
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ttstl.orchestrator import Orchestrator, OrchestratorConfig
from ttstl.utils.data_downloader import import_to_pandas

def setup_logging():
    """Setup logging to see orchestrator operations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_real_data():
    """Load real data for testing (similar to test_adapter.py)."""
    print("📊 Loading real data from GCS...")
    
    # Specify credentials and data
    credentials_path = 'credentials.json'
    bucket_name = 'evolving-microstructural-tk-data'
    
    # Load event-driven and physical time data
    blob_name_edlob = 'data.nosync/Recorded_Synthetic_Data/01102025_long_event_driven_recorded_lob_0.parquet'
    blob_name_ptlob = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_driven_recorded_lob_0.parquet'
    blob_name_ptme = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_market_event_data_0.parquet'

    df_edlob = import_to_pandas(blob_name=blob_name_edlob, bucket_name=bucket_name, credentials_path=credentials_path)
    df_ptlob = import_to_pandas(blob_name=blob_name_ptlob, bucket_name=bucket_name, credentials_path=credentials_path)
    df_ptme = import_to_pandas(blob_name=blob_name_ptme, bucket_name=bucket_name, credentials_path=credentials_path)

    print(f"✓ Loaded event-driven data: {len(df_edlob):,} samples")
    print(f"✓ Loaded physical time LOB data: {len(df_ptlob):,} samples")
    print(f"✓ Loaded physical time market events: {len(df_ptme):,} samples")
    
    return df_edlob, df_ptlob, df_ptme

def process_data_for_orchestrator(df_edlob, df_ptlob, df_ptme, session_start, session_finish, sdelta):
    """Process and filter data according to orchestrator requirements."""
    print("\n🔧 Processing data for orchestrator...")
    
    # Filter data to session time range
    df_edlob = df_edlob[
        (df_edlob['timestamp'] >= session_start) & 
        (df_edlob['timestamp'] < session_finish)
    ].copy()
    
    df_ptlob = df_ptlob[
        (df_ptlob['timestamp'] >= session_start) & 
        (df_ptlob['timestamp'] < session_finish)
    ].copy()
    
    df_ptme = df_ptme[
        (df_ptme['timestamp'] >= session_start) & 
        (df_ptme['timestamp'] < session_finish)
    ].copy()
    
    print(f"✓ Filtered to session range [{session_start}, {session_finish})")
    print(f"  Event-driven data: {len(df_edlob):,} samples")
    print(f"  Physical time data: {len(df_ptlob):,} samples")
    
    # Create sampling time bars and compute OFI
    print(f"📊 Computing OFI using sampling delta: {sdelta}")
    df_edlob['sampling_time_bar'] = (np.round(df_edlob['timestamp'] / sdelta) * sdelta).astype(int)
    
    # Compute OFI for 10 levels
    for k in range(10):
        df_edlob[f'OFI{k}'] = df_edlob[f'BIDof{k}'] - df_edlob[f'ASKof{k}']

    # Aggregate OFI by sampling time bars
    ofi_columns = [f'OFI{k}' for k in range(10)]
    ofi_df = df_edlob.groupby('sampling_time_bar')[ofi_columns].sum().reset_index()
    ofi_df = ofi_df.rename(columns={'sampling_time_bar': 'timestamp'})
    
    # Merge physical time data
    df_pt = df_ptlob.merge(df_ptme, on='timestamp', how='inner')
    df_pt = df_pt.merge(ofi_df, on='timestamp', how='inner')
    
    # Create micro_return target variable
    df_pt['micro_return'] = df_pt['micro'].pct_change() * 1e3
    df_pt.dropna(subset=['micro_return'], inplace=True)
    
    print(f"✓ Created merged dataset with {len(df_pt):,} samples")
    print(f"✓ Target variable 'micro_return' created")
    print(f"  Timestamp range: {df_pt['timestamp'].min():.0f} to {df_pt['timestamp'].max():.0f}")
    
    return df_edlob, df_pt

def demonstrate_partitioning(orchestrator, event_df, phys_df):
    """Demonstrate the data partitioning functionality."""
    print("\n📊 Demonstrating Data Partitioning")
    print("=" * 50)
    
    # Apply time bar columns
    event_df_processed = orchestrator._add_time_bar_columns(event_df.copy())
    phys_df_processed = orchestrator._add_time_bar_columns(phys_df.copy())
    
    # Show partitioning results
    event_backbone_bars = sorted(event_df_processed['backbone_bar'].unique())
    event_adapter_bars = sorted(event_df_processed['adapter_bar'].unique())
    phys_backbone_bars = sorted(phys_df_processed['backbone_bar'].unique())
    phys_adapter_bars = sorted(phys_df_processed['adapter_bar'].unique())
    
    print(f"🔍 Event-driven data partitioning:")
    print(f"  Backbone bars: {len(event_backbone_bars)} bars ({min(event_backbone_bars)} to {max(event_backbone_bars)})")
    print(f"  Adapter bars: {len(event_adapter_bars)} bars ({min(event_adapter_bars)} to {max(event_adapter_bars)})")
    
    print(f"\n🔍 Physical time data partitioning:")
    print(f"  Backbone bars: {len(phys_backbone_bars)} bars ({min(phys_backbone_bars)} to {max(phys_backbone_bars)})")
    print(f"  Adapter bars: {len(phys_adapter_bars)} bars ({min(phys_adapter_bars)} to {max(phys_adapter_bars)})")
    
    # Show samples per backbone bar
    print(f"\n📈 Samples per backbone bar (first 5):")
    for bb in event_backbone_bars[:5]:
        event_count = len(event_df_processed[event_df_processed['backbone_bar'] == bb])
        phys_count = len(phys_df_processed[phys_df_processed['backbone_bar'] == bb])
        print(f"  BB {bb}: {event_count:,} event samples, {phys_count:,} physical samples")
    
    # Show adapter bars within first backbone bar
    first_bb = event_backbone_bars[0]
    first_bb_event = event_df_processed[event_df_processed['backbone_bar'] == first_bb]
    first_bb_phys = phys_df_processed[phys_df_processed['backbone_bar'] == first_bb]
    
    first_bb_adapter_bars_event = sorted(first_bb_event['adapter_bar'].unique())
    first_bb_adapter_bars_phys = sorted(first_bb_phys['adapter_bar'].unique())
    
    print(f"\n📊 Adapter bars within backbone bar {first_bb}:")
    print(f"  Event-driven: {len(first_bb_adapter_bars_event)} adapter bars")
    print(f"  Physical time: {len(first_bb_adapter_bars_phys)} adapter bars")
    
    return event_df_processed, phys_df_processed

def test_global_update(orchestrator, event_df_processed, phys_df_processed):
    """Test the global_update method for ALL backbone bars to verify directory structure."""
    print("\n🚀 Testing Global Update Method - ALL BACKBONE BARS")
    print("=" * 60)
    
    try:
        # Determine how many backbone bars we should process - FAST TESTING: only process first backbone bar
        backbone_bars = sorted(event_df_processed['backbone_bar'].unique())
        total_backbone_bars = len(backbone_bars)
        
        print(f"Available backbone bars: {backbone_bars}")
        print(f"FAST TESTING MODE: Processing only the first backbone bar ({backbone_bars[0]})")
        
        all_results = []
        
        # Process only the first backbone bar for faster testing
        bb = backbone_bars[0]
        print(f"\n{'='*50}")
        print(f"Processing backbone bar {bb} (1/1 - FAST TESTING)")
        print(f"{'='*50}")
        
        # Filter data for this specific backbone bar
        event_bb_data = event_df_processed[event_df_processed['backbone_bar'] == bb].copy()
        phys_bb_data = phys_df_processed[phys_df_processed['backbone_bar'] == bb].copy()
        
        print(f"Backbone bar {bb} data:")
        print(f"  Event samples: {len(event_bb_data):,}")
        print(f"  Physical samples: {len(phys_bb_data):,}")
        
        # Run global update for this backbone bar
        results = orchestrator.global_update(event_bb_data, phys_bb_data)
        all_results.append(results)
        
        print(f"✅ Backbone bar {bb} completed successfully!")
        print(f"📊 Results summary:")
        print(f"  Backbone bar: {results['backbone_bar']}")
        print(f"  Backbone status: {results['backbone_results']['status']}")
        
        if 'event_samples' in results['backbone_results']:
            print(f"  Event samples processed: {results['backbone_results']['event_samples']:,}")
            print(f"  Physical samples processed: {results['backbone_results']['phys_samples']:,}")
        
        if 'backbone_result' in results['backbone_results']:
            backbone_result = results['backbone_results']['backbone_result']
            if isinstance(backbone_result, dict) and 'full_covariate_set' in backbone_result:
                print(f"  Final covariate count: {len(backbone_result['full_covariate_set'])}")
        
        print(f"  Interface status: {results['interface_results']['status']}")
        print(f"  Adapter status: {results['adapter_results']['status']}")
        
        # Show artifacts created for this backbone bar
        artifact_paths = results['artifact_paths']
        print(f"\n📁 Artifacts created for bb_{results['backbone_bar']:03d}:")
        for module, paths in artifact_paths.items():
            print(f"  {module}: {len(paths)} files")
            for path in paths[:2]:  # Show first 2 files per module
                rel_path = str(path).replace(str(orchestrator.run_dir), "")
                print(f"    - {rel_path}")
            if len(paths) > 2:
                print(f"    ... and {len(paths) - 2} more")
        
        print(f"\n{'='*60}")
        print(f"🎉 FAST TESTING COMPLETE - 1 BACKBONE BAR PROCESSED!")
        print(f"{'='*60}")
        
        # Summary statistics for the single backbone bar
        result = all_results[0]
        total_event_samples = result['backbone_results'].get('event_samples', 0) 
        total_phys_samples = result['backbone_results'].get('phys_samples', 0)
        
        print(f"\n📊 TESTING STATISTICS:")
        print(f"  Backbone bars processed: {len(all_results)} (fast testing mode)")
        print(f"  Event samples processed: {total_event_samples:,}")
        print(f"  Physical samples processed: {total_phys_samples:,}")
        
        # Show covariate count if available
        bb_result = result['backbone_results'].get('backbone_result', {})
        if isinstance(bb_result, dict) and 'full_covariate_set' in bb_result:
            covariate_count = len(bb_result['full_covariate_set'])
            print(f"  Final covariate count: {covariate_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Global update failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demonstration function."""
    print("🚀 TTSTL Orchestrator Global Update Demo with Real Data")
    print("=" * 70)
    
    # Setup logging
    setup_logging()
    
    # Load real data
    df_edlob, df_ptlob, df_ptme = load_real_data()
    
    # Configuration parameters - REDUCED FOR FASTER TESTING (10% of data, 1 backbone bar)
    session_start = 0.0
    session_finish = 1000000.0  # Reduced from 10M to 1M (10% of original data)
    bdelta = 1000000.0         # Set to session length to ensure only 1 backbone bar
    adelta = 100000.0          # Adapter bar length (unchanged)
    sdelta = 100.0             # Sampling bar length (unchanged)
    
    print(f"\n⚙️  Configuration Parameters (FAST TESTING MODE - 10% data, 1 backbone bar):")
    print(f"  Session: [{session_start:.0f}, {session_finish:.0f}]")
    print(f"  Backbone delta: {bdelta:.0f}")
    print(f"  Adapter delta: {adelta:.0f}")
    print(f"  Sampling delta: {sdelta:.0f}")
    
    # Process data for the session
    event_df, phys_df = process_data_for_orchestrator(
        df_edlob, df_ptlob, df_ptme, session_start, session_finish, sdelta
    )
    
    # Use the actual runs directory instead of temporary directory
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    
    # Create orchestrator configuration
    cfg = OrchestratorConfig(
        ticker='DEMO_GLOBAL',
        bdelta=bdelta,
        adelta=adelta,
        sdelta=sdelta,
        seed=42,
        session_start=session_start,
        session_finish=session_finish,
        
        # Small number of epochs for testing training history and model saving
        epochs=3,
        batch_size=16,
        validation_proportion_backbone=0.15,
        validation_proportion_interface=0.25,
        
        # Target variable
        target_col_name='micro_return',
        
        run_id='global_demo'
    )
    
    print(f"\n🏗️  Creating orchestrator...")
    orchestrator = Orchestrator(runs_dir, cfg)
    
    print(f"✓ Orchestrator created with {len(orchestrator.stacks)} stacks")
    print(f"  Run directory: {orchestrator.run_dir}")
    
    # Demonstrate partitioning
    event_df_processed, phys_df_processed = demonstrate_partitioning(
        orchestrator, event_df, phys_df
    )
    
    # Test global update
    success = test_global_update(orchestrator, event_df_processed, phys_df_processed)
    
    if success:
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Results saved to: {orchestrator.run_dir}")
        print(f"📁 Absolute path: {orchestrator.run_dir.absolute()}")
        
        # List some key artifacts for verification
        print(f"\n📋 Key artifacts created:")
        config_file = orchestrator.run_dir / "config.yaml"
        manifest_file = orchestrator.run_dir / "manifest.json"
        if config_file.exists():
            print(f"  ✓ Configuration: {config_file}")
        if manifest_file.exists():
            print(f"  ✓ Manifest: {manifest_file}")
        
        # List backbone artifacts
        bb_dir = orchestrator.run_dir / "bb_000" / "global_update"
        if bb_dir.exists():
            print(f"  ✓ Backbone artifacts: {bb_dir}")
            
            # Check correlation artifacts
            correlation_dir = bb_dir / "backbone" / "correlation"
            if correlation_dir.exists():
                heatmaps = list(correlation_dir.glob("*.png"))
                if heatmaps:
                    print(f"    - Correlation heatmaps: {len(heatmaps)} files")
                    for heatmap in heatmaps[:3]:  # Show first 3
                        print(f"      • {heatmap.name}")
            
            # Check training artifacts
            architecture_dir = bb_dir / "backbone" / "architecture"
            if architecture_dir.exists():
                print(f"    - Architecture artifacts:")
                
                # Check for training history
                train_log = architecture_dir / "train_log.parquet"
                if train_log.exists():
                    print(f"      ✓ Training history: {train_log.name}")
                    # Load and show basic info about training history
                    try:
                        import pandas as pd
                        df = pd.read_parquet(train_log)
                        print(f"        - Epochs recorded: {len(df)}")
                        print(f"        - Final train loss: {df['train_loss'].iloc[-1]:.6f}")
                        print(f"        - Final validation loss: {df['validation_loss'].iloc[-1]:.6f}")
                        print(f"        - Best validation loss: {df['validation_loss'].min():.6f}")
                    except Exception as e:
                        print(f"        - Error reading training history: {e}")
                else:
                    print(f"      ❌ Training history not found")
                
                # Check for model weights
                best_model = architecture_dir / "best.pt"
                if best_model.exists():
                    print(f"      ✓ Model weights: {best_model.name}")
                    # Show file size
                    size_mb = best_model.stat().st_size / (1024 * 1024)
                    print(f"        - Size: {size_mb:.2f} MB")
                else:
                    print(f"      ❌ Model weights not found")
                
                # Check for model metadata
                metadata_file = architecture_dir / "model_metadata.json"
                if metadata_file.exists():
                    print(f"      ✓ Model metadata: {metadata_file.name}")
                    # Load and show basic info
                    try:
                        import json
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        print(f"        - Model class: {metadata.get('model_class', 'N/A')}")
                        print(f"        - Device: {metadata.get('device', 'N/A')}")
                        print(f"        - Epochs trained: {metadata.get('epochs_trained', 'N/A')}")
                    except Exception as e:
                        print(f"        - Error reading metadata: {e}")
                else:
                    print(f"      ❌ Model metadata not found")
    else:
        print(f"\n❌ Demo failed!")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
