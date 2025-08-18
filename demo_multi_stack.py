#!/usr/bin/env python3
"""
Demonstration of the enhanced TTSTL Orchestrator with multiple backbone-interface-adapter stacks.
"""

import tempfile
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ttstl.orchestrator import Orchestrator, OrchestratorConfig

def demonstrate_stack_creation():
    """Demonstrate the multi-stack functionality with updated configuration."""
    
    print("🚀 TTSTL Orchestrator Multi-Stack Demonstration")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Example configuration with partial last bar and new backbone parameters
        cfg = OrchestratorConfig(
            ticker='DEMO_MULTI',
            bdelta=100000.0,      # 100k units per backbone bar
            adelta=10000.0,       # 10k units per adapter bar  
            sdelta=1000.0,        # 1k units per sample bar
            seed=42,
            session_start=0.0,
            session_finish=350000.0,  # Creates 4 stacks (3 full + 1 partial)
            
            # Updated backbone configuration
            architecture='cnn_lstm',
            parsimonious_mode=True,
            epochs=5,             # Reduced for demo
            batch_size=16,        # Reduced for demo
            validation_proportion_backbone=0.15,
            validation_proportion_interface=0.25,
            buffer_proportion=0.08,
            quantile=0.01,
            target_col_name='micro_return',
            
            # Stationarity test parameters
            # Stationarity test parameters
            adf_rejection_rate_threshold=0.85,
            kpss_verification_threshold=0.15,
            na_verification_threshold=0.08,
            
            # Interface parameters
            selection_threshold=0.12,
            tolerance_threshold=1e-5,
            ols_significance_level=0.01,
            
            # Adapter parameters
            update_method='myopic',
            arma_order=(2, 1),  # Different from default for demo
            garch_order=(1, 2),  # Different from default for demo
            dist='std',  # Different from default for demo
            
            run_id='demo_stacks'
        )
        
        print(f"📊 Session Configuration:")
        print(f"   Ticker: {cfg.ticker}")
        print(f"   Session length: {cfg.session_finish - cfg.session_start:,.0f} units")
        print(f"   Backbone bar length: {cfg.bdelta:,.0f} units")
        print(f"   Architecture: {cfg.architecture}")
        print(f"   Target column: {cfg.target_col_name}")
        print()
        
        print(f"🧠 Backbone Configuration:")
        print(f"   Parsimonious mode: {cfg.parsimonious_mode}")
        print(f"   Epochs: {cfg.epochs}")
        print(f"   Batch size: {cfg.batch_size}")
        print(f"   Validation proportion (backbone): {cfg.validation_proportion_backbone}")
        print(f"   Buffer proportion: {cfg.buffer_proportion}")
        print(f"   Quantile: {cfg.quantile}")
        print()
        
        print(f"🔍 Interface Configuration:")
        print(f"   Validation proportion (interface): {cfg.validation_proportion_interface}")
        print(f"   Selection threshold: {cfg.selection_threshold}")
        print(f"   Tolerance threshold: {cfg.tolerance_threshold}")
        print(f"   OLS significance level: {cfg.ols_significance_level}")
        print()
        
        print(f"📈 Stationarity Test Configuration:")
        print(f"   ADF rejection threshold: {cfg.adf_rejection_rate_threshold}")
        print(f"   KPSS verification threshold: {cfg.kpss_verification_threshold}")
        print(f"   NA verification threshold: {cfg.na_verification_threshold}")
        print()
        
        print(f"⚙️  Adapter Configuration:")
        print(f"   Update method: {cfg.update_method}")
        print(f"   ARMA order: {cfg.arma_order}")
        print(f"   GARCH order: {cfg.garch_order}")
        print(f"   Distribution: {cfg.dist}")
        print(f"   Model: {cfg.model}")
        print(f"   ARCH-M: {cfg.archm}")
        print()
        
        # Create orchestrator
        orchestrator = Orchestrator(tmpdir, cfg)
        
        print(f"🏗️  Stack Creation Results:")
        print(f"   Number of stacks created: {len(orchestrator.stacks)}")
        print(f"   Stack keys: {list(orchestrator.stacks.keys())}")
        
        # Show stack structure
        print(f"\n🔍 Stack Structure:")
        for stack_name, adapter in orchestrator.stacks.items():
            print(f"   {stack_name}:")
            print(f"     └── Adapter: {type(adapter).__name__}")
            print(f"         └── Interface: {type(adapter.interface).__name__}")
            print(f"             └── Backbone: {type(adapter.interface.backbone).__name__}")
            
            # Show backbone configuration
            backbone = adapter.interface.backbone
            print(f"                 ├── Parsimonious mode: {backbone.parsimonious_mode}")
            print(f"                 ├── Delta adapter: {backbone.delta_adapter}")
            print(f"                 └── Delta sample: {backbone.delta_sample}")
        
        # Show backward compatibility
        print(f"\n🔄 Backward Compatibility:")
        print(f"   orchestrator.adapter: {type(orchestrator.adapter).__name__}")
        print(f"   orchestrator.interface: {type(orchestrator.interface).__name__}")
        print(f"   orchestrator.backbone: {type(orchestrator.backbone).__name__}")
        print(f"   Points to first stack: {orchestrator.adapter == orchestrator.stacks['backbone_0']}")
        
        # Show run directory
        print(f"\n📁 Run Directory:")
        print(f"   Location: {orchestrator.run_dir}")
        print(f"   Run slug: {orchestrator.run_slug}")
        
        # Calculate actual stack statistics
        session_length = cfg.session_finish - cfg.session_start
        full_stacks = int(session_length // cfg.bdelta)
        residual = session_length % cfg.bdelta
        residual_percentage = (residual / cfg.bdelta) * 100 if residual > 0 else 0
        
        print(f"\n📈 Session Division Analysis:")
        print(f"   Full backbone bars: {full_stacks}")
        print(f"   Residual length: {residual:,.0f} units ({residual_percentage:.1f}% of bdelta)")
        if residual > 0:
            if residual_percentage < 1:
                print(f"   ⚠️  Small residual dropped (< 1% threshold)")
            else:
                print(f"   ⚠️  Partial last bar included (≥ 1% threshold)")
        else:
            print(f"   ✅ Perfect division (no residual)")
        
        # Test configuration access
        print(f"\n🔧 Configuration Validation:")
        print(f"   Config accessible: {orchestrator.cfg == cfg}")
        print(f"   Backbone config applied: {orchestrator.backbone.parsimonious_mode == cfg.parsimonious_mode}")
        print(f"   Delta values match: {orchestrator.backbone.delta_adapter == cfg.adelta}")
        print(f"   Adapter config applied: {orchestrator.adapter.update_method == cfg.update_method}")
        print(f"   GARCH order match: {orchestrator.adapter.garch_order == cfg.garch_order}")
        print(f"   ARMA order match: {orchestrator.adapter.arma_order == cfg.arma_order}")
    
    print("\n" + "=" * 60)
    print("🎉 Multi-stack demonstration completed!")

if __name__ == "__main__":
    demonstrate_stack_creation()
