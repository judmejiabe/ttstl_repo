#!/usr/bin/env python3
"""
Example demonstrating the new simplified Orchestrator API.

The orchestrator now handles backbone bar iteration automatically,
making the user experience much simpler.
"""

import logging
from pathlib import Path
from ttstl.orchestrator import Orchestrator, OrchestratorConfig
from ttstl.utils.data_downloader import import_to_pandas

def setup_logging():
    """Setup logging to see orchestrator operations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Demonstrate the new simplified API."""
    print("🚀 New Simplified Orchestrator API Demo")
    print("=" * 50)
    
    setup_logging()
    
    # Load some data (using existing data from repository)
    try:
        df_edlob = import_to_pandas(
            blob_name='some_data.parquet',
            credentials_path='credentials.json',
            bucket_name='evolving-microstructural-tk-data'
        )
        df_pt = df_edlob.copy()  # For demo purposes
        print(f"✓ Loaded data: {len(df_edlob):,} samples")
    except Exception as e:
        print(f"❌ Could not load data: {e}")
        print("Using synthetic data for demo...")
        import pandas as pd
        import numpy as np
        
        # Create minimal synthetic data
        n_samples = 10000
        timestamps = np.arange(n_samples) * 100  # 100 time units apart
        df_edlob = pd.DataFrame({
            'timestamp': timestamps,
            'price': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
            'volume': np.random.randint(1, 100, n_samples)
        })
        df_pt = df_edlob.copy()
        print(f"✓ Created synthetic data: {len(df_edlob):,} samples")
    
    # Create configuration
    cfg = OrchestratorConfig(
        ticker='DEMO',
        bdelta=2500.0,           # Backbone bars: 4 bars for 10k samples
        adelta=250.0,            # Adapter bars: 10 per backbone bar  
        sdelta=25.0,             # Sample bars: 10 per adapter bar
        seed=42,
        session_start=0.0,
        session_finish=10000.0,  # Match data range
        
        # Quick training for demo
        epochs=2,
        batch_size=16,
        
        # Simplified interface config
        alpha_enet_grid=[0.5],
        lambda_enet_grid=[0.1],
        tau_adalasso_grid=[1.0],
        lambda_adalasso_grid=[0.1],
        
        run_id='new_api_demo'
    )
    
    print(f"\n🏗️  Creating orchestrator...")
    runs_dir = Path('runs')
    runs_dir.mkdir(exist_ok=True)
    
    orchestrator = Orchestrator(runs_dir, cfg)
    print(f"✓ Orchestrator created")
    print(f"  Run directory: {orchestrator.run_dir}")
    
    # Show the configuration for backbone bars
    event_df_processed = orchestrator._add_time_bar_columns(df_edlob.copy())
    backbone_bars = sorted(event_df_processed['backbone_bar'].unique())
    print(f"\n📊 Data will be processed in {len(backbone_bars)} backbone bars: {backbone_bars}")
    
    try:
        print(f"\n" + "="*60)
        print("🚀 NEW API: AUTOMATIC GLOBAL UPDATE")
        print("="*60)
        print("No need to iterate over backbone bars manually!")
        print("The orchestrator handles everything automatically...")
        
        # OLD WAY (manual iteration):
        # for bb in backbone_bars:
        #     event_bb_data = event_df_processed[event_df_processed['backbone_bar'] == bb]
        #     phys_bb_data = phys_df_processed[phys_df_processed['backbone_bar'] == bb]
        #     results = orchestrator.global_update_single_bar(event_bb_data, phys_bb_data)
        
        # NEW WAY (automatic iteration):
        all_global_results = orchestrator.global_update(df_edlob, df_pt)
        
        print(f"✅ Global update completed for {len(all_global_results)} backbone bars!")
        
        # Show summary of results
        for i, result in enumerate(all_global_results):
            print(f"  Backbone bar {result['backbone_bar']}: {result['backbone_results']['status']}")
        
        print(f"\n" + "="*60)
        print("🔄 NEW API: AUTOMATIC LOCAL ADAPTATION")
        print("="*60)
        print("Again, no manual iteration needed!")
        
        # OLD WAY (manual iteration):
        # for bb in backbone_bars:
        #     event_bb_data = event_df_processed[event_df_processed['backbone_bar'] == bb]
        #     phys_bb_data = phys_df_processed[phys_df_processed['backbone_bar'] == bb]
        #     results = orchestrator.local_adapt_single_bar(event_bb_data, phys_bb_data)
        
        # NEW WAY (automatic iteration):
        all_local_results = orchestrator.local_adapt(df_edlob, df_pt)
        
        print(f"✅ Local adaptation completed for {len(all_local_results)} backbone bars!")
        
        # Show summary of results
        for i, result in enumerate(all_local_results):
            bb_status = result['backbone_results']['status'] if 'backbone_results' in result else 'N/A'
            print(f"  Backbone bar {result['backbone_bar']}: {bb_status}")
        
        print(f"\n🎉 DEMO COMPLETE!")
        print(f"📁 Results saved to: {orchestrator.run_dir}")
        print(f"📊 Processed {len(all_global_results)} backbone bars automatically")
        
        print(f"\n💡 KEY BENEFITS OF NEW API:")
        print(f"  ✅ No manual backbone bar iteration required")
        print(f"  ✅ Automatic data partitioning and filtering") 
        print(f"  ✅ Robust error handling across all bars")
        print(f"  ✅ Comprehensive logging and progress tracking")
        print(f"  ✅ Backward compatibility (single bar methods still available)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
