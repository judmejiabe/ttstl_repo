#!/usr/bin/env python3
"""
Demo script for TTSTL Local Adaptation using the same workflow and data as global_update.

This script demonstrates the local adaptation workflow:
1. Backbone: Frozen (no updates)
2. Interface: Local variable selection using existing hyperparameters
3. Adapter: Parameter updates similar to global_update
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
    """Load real data for testing (same as demo_global_update.py)."""
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

def main():
    print("="*80)
    print("🔄 TTSTL LOCAL ADAPTATION DEMO")
    print("="*80)
    
    # Setup logging
    setup_logging()
    
    # Load real data from GCS
    try:
        df_edlob, df_ptlob, df_ptme = load_real_data()
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        print("Make sure credentials.json is available and network connection is working.")
        return
    
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
    
    # Process data for orchestrator
    try:
        df_edlob, df_pt = process_data_for_orchestrator(
            df_edlob, df_ptlob, df_ptme,
            session_start, session_finish, sdelta
        )
    except Exception as e:
        print(f"❌ Failed to process data: {e}")
        return
    
    print(f"\n📊 Final data shapes:")
    print(f"  Event data: {len(df_edlob):,} samples")
    print(f"  Physical data: {len(df_pt):,} samples")
    
    if len(df_pt) < 100:
        print("❌ Insufficient data for demo")
        return
    
    # Use the actual runs directory instead of temporary directory
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    
    # Create orchestrator configuration - same as global_update demo but for local adaptation
    cfg = OrchestratorConfig(
        ticker='DEMO_LOCAL',
        bdelta=bdelta,
        adelta=adelta,
        sdelta=sdelta,
        seed=42,
        session_start=session_start,
        session_finish=session_finish,
        
        # Zero epochs for mock run (no neural network training)
        epochs=5,
        batch_size=16,
        validation_proportion_backbone=0.15,
        validation_proportion_interface=0.25,
        
        # Target variable
        target_col_name='micro_return',
        
        # Interface settings - reduced grids for faster testing
        alpha_enet_grid=[0.5],         # Single value for fast testing
        lambda_enet_grid=[0.1],        # Single value for fast testing
        tau_adalasso_grid=[0.5],       # Single value for fast testing
        lambda_adalasso_grid=[0.1],    # Single value for fast testing
        
        run_id='local_demo'
    )
    
    print(f"\n🏗️  Creating orchestrator...")
    orchestrator = Orchestrator(runs_dir, cfg)
    
    print(f"✓ Orchestrator created")
    print(f"  Run directory: {orchestrator.run_dir}")
    
    # Apply time bar columns for partitioning
    event_df_processed = orchestrator._add_time_bar_columns(df_edlob.copy())
    phys_df_processed = orchestrator._add_time_bar_columns(df_pt.copy())
    
    # Show partitioning results  
    backbone_bars = sorted(event_df_processed['backbone_bar'].unique())
    print(f"\n📊 Data partitioning:")
    print(f"  Available backbone bars: {backbone_bars}")
    print(f"  FAST TESTING MODE: Processing only the first backbone bar ({backbone_bars[0]})")
    
    # Filter data for the first backbone bar only
    bb = backbone_bars[0]
    event_bb_data = event_df_processed[event_df_processed['backbone_bar'] == bb].copy()
    phys_bb_data = phys_df_processed[phys_df_processed['backbone_bar'] == bb].copy()
    
    print(f"\nBackbone bar {bb} data:")
    print(f"  Event samples: {len(event_bb_data):,}")
    print(f"  Physical samples: {len(phys_bb_data):,}")
    
    try:
        # STEP 1: Run global update first (required for local adaptation)
        print(f"\n" + "="*60)
        print("STEP 1: RUNNING GLOBAL UPDATE (PREREQUISITE)")
        print("="*60)
        print("Local adaptation requires global update to be run first...")
        print("� Neural network training: ENABLED (epochs=5)")
        print("📊 Processing: First backbone bar only (fast testing)")
        
        global_results = orchestrator.global_update(event_bb_data, phys_bb_data)
        
        if "error" in global_results:
            print(f"❌ Global update failed: {global_results['error']}")
            return
        
        print(f"✅ Global update completed successfully!")
        print(f"  Backbone bar: {global_results['backbone_bar']}")
        print(f"  Backbone status: {global_results['backbone_results']['status']}")
        print(f"  Interface status: {global_results['interface_results']['status']}")
        print(f"  Adapter status: {global_results['adapter_results']['status']}")
        
        # Show processing statistics
        if 'event_samples' in global_results['backbone_results']:
            print(f"  Event samples processed: {global_results['backbone_results']['event_samples']:,}")
            print(f"  Physical samples processed: {global_results['backbone_results']['phys_samples']:,}")
        
        # STEP 2: Run local adaptation
        print(f"\n" + "="*60)
        print("STEP 2: RUNNING LOCAL ADAPTATION")
        print("="*60)
        print("🔄 Backbone: FROZEN (no updates)")
        print("🔍 Interface: Local variable selection with existing hyperparameters")
        print("⚙️  Adapter: Parameter updates similar to global_update")
        print("📊 Processing: Same backbone bar data")
        
        local_results = orchestrator.local_adapt(event_bb_data, phys_bb_data)
        
        if "error" in local_results:
            print(f"❌ Local adaptation failed: {local_results['error']}")
            return
        
        print(f"\n✅ Local adaptation completed successfully!")
        print(f"  Backbone bar: {local_results['backbone_bar']}")
        print(f"  Backbone status: {local_results['backbone_results']['status']}")
        print(f"  Interface status: {local_results['interface_results']['status']}")
        print(f"  Adapter status: {local_results['adapter_results']['status']}")
        
        # Display detailed results
        print(f"\n📊 Detailed Results:")
        
        # Backbone results
        backbone_res = local_results['backbone_results']
        print(f"\n🔄 Backbone (Frozen):")
        print(f"  Status: {backbone_res['status']}")
        print(f"  Artifacts: {backbone_res['artifacts_created']}")
        
        # Interface results
        interface_res = local_results['interface_results']
        print(f"\n🔍 Interface (Local Selection):")
        print(f"  Status: {interface_res['status']}")
        print(f"  Adapter bars processed: {interface_res.get('total_adapter_bars', 'N/A')}")
        print(f"  Successful bars: {interface_res.get('successful_bars', 'N/A')}")
        print(f"  Success rate: {interface_res.get('success_rate', 0):.1%}")
        
        # Adapter results  
        adapter_res = local_results['adapter_results']
        print(f"\n⚙️  Adapter (Parameter Updates):")
        print(f"  Status: {adapter_res['status']}")
        print(f"  Adapter bars processed: {adapter_res.get('total_adapter_bars', 'N/A')}")
        print(f"  Successful bars: {adapter_res.get('successful_bars', 'N/A')}")
        print(f"  Success rate: {adapter_res.get('success_rate', 0):.1%}")
        print(f"  Artifacts: {adapter_res['artifacts_created']}")
        
        # Show artifact paths
        if 'artifact_paths' in local_results:
            print(f"\n📁 Artifacts created:")
            for module, paths in local_results['artifact_paths'].items():
                print(f"  {module}: {len(paths)} files")
                for path in paths[:2]:  # Show first 2 files
                    rel_path = str(path).replace(str(orchestrator.run_dir), "")
                    print(f"    - {rel_path}")
                if len(paths) > 2:
                    print(f"    ... and {len(paths) - 2} more")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Results saved to: {orchestrator.run_dir}")
        print(f"📁 Absolute path: {orchestrator.run_dir.absolute()}")
        
        print(f"\n📋 Key artifacts created:")
        config_file = orchestrator.run_dir / "config.yaml"
        manifest_file = orchestrator.run_dir / "manifest.json"
        if config_file.exists():
            print(f"  ✓ Configuration: {config_file}")
        if manifest_file.exists():
            print(f"  ✓ Manifest: {manifest_file}")
        
        # List both global and local artifacts
        bb_dir_global = orchestrator.run_dir / "bb_000" / "global_update"
        bb_dir_local = orchestrator.run_dir / "bb_000" / "local_adapt"
        
        if bb_dir_global.exists():
            print(f"  ✓ Global update artifacts: {bb_dir_global}")
        if bb_dir_local.exists():
            print(f"  ✓ Local adaptation artifacts: {bb_dir_local}")
            
        print(f"\n" + "="*80)
        print("🎉 LOCAL ADAPTATION DEMO COMPLETE!")
        print("="*80)
        
        # Summary statistics
        total_event_samples = global_results['backbone_results'].get('event_samples', 0)
        total_phys_samples = global_results['backbone_results'].get('phys_samples', 0)
        
        print(f"\n📊 TESTING STATISTICS:")
        print(f"  Backbone bars processed: 1 (fast testing mode)")
        print(f"  Event samples processed: {total_event_samples:,}")
        print(f"  Physical samples processed: {total_phys_samples:,}")
        
        # Show covariate count if available
        bb_result = global_results['backbone_results'].get('backbone_result', {})
        if isinstance(bb_result, dict) and 'full_covariate_set' in bb_result:
            covariate_count = len(bb_result['full_covariate_set'])
            print(f"  Final covariate count: {covariate_count}")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        print(f"Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
