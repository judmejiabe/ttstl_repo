"""
Integration test demonstrating how to use the TTSTL Orchestrator.

This script shows a typical workflow using the orchestrator to coordinate
backbone training, interface selection, and adapter updates.
"""

import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from ttstl.orchestrator import Orchestrator, OrchestratorConfig


def create_sample_trading_data():
    """Create sample trading data for demonstration."""
    np.random.seed(42)
    
    # Create event-driven data (trades, order book updates, etc.)
    n_events = 5000
    event_data = pd.DataFrame({
        'timestamp': np.sort(np.random.uniform(0, 1000000, n_events)),
        'price': 100 + np.cumsum(np.random.normal(0, 0.01, n_events)),
        'volume': np.random.exponential(1000, n_events),
        'side': np.random.choice(['bid', 'ask'], n_events),
        'order_type': np.random.choice(['market', 'limit', 'cancel'], n_events)
    })
    
    # Create physical time data (regular time intervals with market microstructure features)
    timestamps = np.arange(1000, 1000000, 1000)  # Every 1000 time units
    n_phys = len(timestamps)
    
    # Generate correlated microstructure features
    base_price = 100
    price_changes = np.cumsum(np.random.normal(0, 0.001, n_phys))
    
    phys_data = pd.DataFrame({
        'timestamp': timestamps,
        'micro': base_price + price_changes,
        'bid_price': base_price + price_changes - np.random.uniform(0.005, 0.015, n_phys),
        'ask_price': base_price + price_changes + np.random.uniform(0.005, 0.015, n_phys),
        'spread': np.random.uniform(0.01, 0.05, n_phys),
        'volume': np.random.exponential(5000, n_phys),
        'volatility': np.random.uniform(0.001, 0.01, n_phys),
        'order_imbalance': np.random.normal(0, 0.1, n_phys),
        'trade_intensity': np.random.exponential(10, n_phys),
        'kyle_lambda': np.random.uniform(0.1, 1.0, n_phys)
    })
    
    return event_data, phys_data


def main():
    """Demonstrate orchestrator usage."""
    print("🚀 TTSTL Orchestrator Integration Test")
    print("=" * 50)
    
    # Create configuration
    config = OrchestratorConfig(
        ticker="DEMO",
        bdelta=100000.0,    # Backbone time bar: 100,000 time units
        adelta=10000.0,     # Adapter time bar: 10,000 time units  
        sdelta=1000.0,      # Sampling time bar: 1,000 time units
        seed=42,
        session_start=0.0,
        session_finish=1000000.0,
        epochs=2,           # Reduced for demo
        batch_size=16,      # Reduced for demo
        parsimonious_mode=True,
        run_id="demo001"
    )
    
    print(f"📊 Configuration:")
    print(f"   Ticker: {config.ticker}")
    print(f"   Backbone Δ: {config.bdelta:,.0f}")
    print(f"   Adapter Δ: {config.adelta:,.0f}")
    print(f"   Sampling Δ: {config.sdelta:,.0f}")
    print(f"   Session: {config.session_start:,.0f} → {config.session_finish:,.0f}")
    print()
    
    # Create sample data
    print("📈 Generating sample trading data...")
    event_df, phys_df = create_sample_trading_data()
    print(f"   Event-driven data: {len(event_df):,} records")
    print(f"   Physical time data: {len(phys_df):,} records")
    print(f"   Time range: {event_df['timestamp'].min():.0f} → {event_df['timestamp'].max():.0f}")
    print()
    
    # Create orchestrator
    with tempfile.TemporaryDirectory() as temp_dir:
        runs_root = Path(temp_dir)
        print(f"💾 Creating orchestrator with runs in: {runs_root}")
        
        try:
            orchestrator = Orchestrator(runs_root, config)
            print(f"   ✅ Run directory: {orchestrator.run_slug}")
            print(f"   📁 Full path: {orchestrator.run_dir}")
            print()
            
            # Demonstrate multiple backbone bars
            print("🔄 Running multiple backbone bars...")
            
            # Backbone Bar 0
            print("\n--- Backbone Bar 0 ---")
            print("🏗️  Running global_update...")
            global_results_0 = orchestrator.global_update(event_df, phys_df)
            print(f"   ✅ Global update completed")
            print(f"   📊 Backbone results: {global_results_0['backbone_results']['status']}")
            print(f"   🎯 Interface results: {global_results_0['interface_results']['status']}")
            print(f"   ⚙️  Adapter results: {global_results_0['adapter_results']['status']}")
            
            print("🔍 Running local_adapt...")
            local_results_0 = orchestrator.local_adapt(event_df, phys_df)
            print(f"   ✅ Local adaptation completed")
            print(f"   📈 Artifacts created: {len(local_results_0['artifact_paths'])}")
            
            # Backbone Bar 1
            print("\n--- Backbone Bar 1 ---")
            print("🏗️  Running global_update...")
            global_results_1 = orchestrator.global_update(event_df, phys_df)
            print(f"   ✅ Global update completed")
            
            print("🔍 Running local_adapt...")
            local_results_1 = orchestrator.local_adapt(event_df, phys_df)
            print(f"   ✅ Local adaptation completed")
            
            # Show directory structure
            print(f"\n📁 Directory structure created:")
            for path in sorted(orchestrator.run_dir.rglob("*")):
                if path.is_file():
                    rel_path = path.relative_to(orchestrator.run_dir)
                    print(f"   📄 {rel_path}")
                elif path.is_dir() and path != orchestrator.run_dir:
                    rel_path = path.relative_to(orchestrator.run_dir)
                    print(f"   📁 {rel_path}/")
            
            # Show manifest summary
            import json
            manifest_file = orchestrator.run_dir / "manifest.json"
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            print(f"\n📋 Manifest summary:")
            print(f"   🏷️  Run slug: {manifest['run_slug']}")
            print(f"   ⏰ Created: {manifest['created_at']}")
            print(f"   🔧 Python: {manifest['environment']['python']}")
            print(f"   🧠 PyTorch: {manifest['environment']['torch']}")
            print(f"   📊 Artifacts: {len(manifest['artifacts'])} entries")
            
            for i, artifact in enumerate(manifest['artifacts']):
                print(f"      {i+1}. BB{artifact['backbone_bar']:03d}/{artifact['mode']} "
                      f"({len(artifact['artifact_paths'])} artifact groups)")
            
            print(f"\n✅ Integration test completed successfully!")
            
        except Exception as e:
            print(f"❌ Integration test failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()
