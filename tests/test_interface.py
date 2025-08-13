import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from ttstl.backbone import Backbone
from ttstl.interface import Interface
from ttstl.architectures.cnn_lstm import CNNLSTM
from ttstl.utils.data_downloader import import_to_pandas
from ttstl.utils.data_preprocessor import get_dataset_cnn_lstm


def main():
    """Test the global_update function in the Interface module."""
    
    print("Starting Interface global_update test...")
    
    # Initialize the CNN-LSTM model - use GPU if available
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = CNNLSTM().to(device)

    # Specify your credentials
    credentials_path = 'credentials.json'

    # Specify your bucket & blob
    bucket_name = 'evolving-microstructural-tk-data'
    blob_name_train = "data.nosync/Recorded_Synthetic_Data/01102025_short_event_driven_recorded_lob_1.parquet"
    blob_name_validation = "data.nosync/Recorded_Synthetic_Data/01102025_short_event_driven_recorded_lob_2.parquet"

    # Download training data for initial backbone training
    df_train = import_to_pandas(
        blob_name=blob_name_train,
        bucket_name=bucket_name,
        credentials_path=credentials_path)
    
    df_validation = import_to_pandas(
        blob_name=blob_name_validation,
        bucket_name=bucket_name,
        credentials_path=credentials_path)

    # Prepare datasets for backbone training
    dataset_train = get_dataset_cnn_lstm(df_train, delta=100, quantile=None)
    dataset_validation = get_dataset_cnn_lstm(df_validation, delta=100, quantile=None)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=4)
    validation_loader = DataLoader(dataset_validation, batch_size=32, shuffle=False, num_workers=4)

    # Initialize backbone with adapter delta
    backbone = Backbone(model=model, 
                        delta_adapter=100000.0,
                        delta_sample=100.0,
                        parsimonious_mode=False)

    criterion = torch.nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the backbone model briefly
    print("Training backbone model...")
    backbone.batch_gradient_descent(train_loader, 
                   criterion, 
                   optimizer, 
                   validation_loader=validation_loader,
                   epochs=1)  # Minimal training for testing
    
    # Load the larger datasets for interface testing
    blob_name_edlob = 'data.nosync/Recorded_Synthetic_Data/01102025_long_event_driven_recorded_lob_0.parquet'
    blob_name_ptlob = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_driven_recorded_lob_0.parquet'
    blob_name_ptme = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_market_event_data_0.parquet'

    print("Loading larger datasets for interface testing...")
    df_edlob = import_to_pandas(
        blob_name=blob_name_edlob,
        bucket_name=bucket_name,
        credentials_path=credentials_path)

    # Prepare event-driven data (same as in backbone test)
    delta = 100
    df_edlob['sampling_time_bar'] = (np.rint((df_edlob['timestamp']) / delta).astype(int) + 1) * delta

    for k in range(10):
        df_edlob[f'OFI{k}'] = df_edlob[f'BIDof{k}'] - df_edlob[f'ASKof{k}']

    ofi_columns = [f'OFI{k}' for k in range(10)]
    ofi_df = df_edlob.groupby('sampling_time_bar')[ofi_columns].sum().reset_index()
    ofi_df = ofi_df.rename(columns={'sampling_time_bar': 'timestamp'})

    # Load physical time data
    df_ptlob = import_to_pandas(
        blob_name=blob_name_ptlob,
        bucket_name=bucket_name,
        credentials_path=credentials_path)

    df_ptme = import_to_pandas(
        blob_name=blob_name_ptme,
        bucket_name=bucket_name,
        credentials_path=credentials_path)

    # Merge physical time data
    df_pt = df_ptlob.merge(df_ptme, on='timestamp', how='inner')
    df_pt = df_pt.merge(ofi_df, on='timestamp', how='inner')
    
    # REDUCE DATASET SIZE for faster testing
    reduction_factor = 0.1
    print(f"Reducing dataset size to first {int(reduction_factor*100)}% of timestamps for faster testing...")
    
    # Get timestamp range for event-driven data
    edlob_min_ts, edlob_max_ts = df_edlob['timestamp'].min(), df_edlob['timestamp'].max()
    edlob_cutoff = edlob_min_ts + reduction_factor * (edlob_max_ts - edlob_min_ts)
    
    # Get timestamp range for physical time data  
    pt_min_ts, pt_max_ts = df_pt['timestamp'].min(), df_pt['timestamp'].max()
    pt_cutoff = pt_min_ts + reduction_factor * (pt_max_ts - pt_min_ts)

    # Filter to first 10% of data
    df_edlob = df_edlob[df_edlob['timestamp'] <= edlob_cutoff].copy()
    df_pt = df_pt[df_pt['timestamp'] <= pt_cutoff].copy()
    
    print(f"Event-driven data reduced from full dataset to {len(df_edlob):,} samples")
    print(f"Physical time data reduced from full dataset to {len(df_pt):,} samples")

    # Create micro_return column for testing
    df_pt['micro_return'] = df_pt['micro'].pct_change() * 1e3
    df_pt['micro_return'] = df_pt['micro_return'].shift(-1)  # Shift to align with next timestamp
    df_pt.dropna(subset=['micro_return'], inplace=True)  # Remove NaNs from returns

    # Reset index to have timestamp as column (required for interface)
    df_pt_reset = df_pt.reset_index() if 'timestamp' not in df_pt.columns else df_pt.copy()
    
    print(df_pt_reset[['timestamp', 'micro', 'micro_return']].head())
    
    print(f"Event-driven data shape: {df_edlob.shape}")
    print(f"Physical time data shape: {df_pt_reset.shape}")
    
    # Run backbone global_update to prepare neural covariate extraction
    print("Running backbone global_update to prepare for interface testing...")
    try:
        backbone.global_update(
            event_driven_backbone_time_bar_data=df_edlob,
            physical_time_driven_backbone_time_bar_data=df_pt_reset,
            criterion=criterion,
            optimizer=optimizer,
            epochs=0,  # No additional training as requested
            validation_proportion=0.1,
            buffer_proportion=0.05,
            batch_size=64,
            adf_rejection_rate_threshold=0.8,
            kpss_verification_threshold=0.2,
            na_verification_threshold=0.1,
            kpss_drop=True,
            correlation_threshold=0.95
        )
        print("✓ Backbone global_update completed successfully")
    except Exception as e:
        print(f"✗ Backbone global_update failed: {e}")
        return False
    
    # Initialize Interface with complete hyperparameter grids
    print("Initializing Interface module...")
    
    # Set up hyperparameter grids for comprehensive testing
    alpha_enet_grid = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # Smaller grid for faster testing
    lambda_enet_grid = [0.01, 0.1, 1.0, 5.0, 10.0]
    tau_adalasso_grid = [0.25, 0.5, 1.0, 2.0]
    lambda_adalasso_grid = [0.01, 0.1, 1.0, 5.0, 10.0]
    
    interface = Interface(backbone_module=backbone,
                         alpha_enet_grid=alpha_enet_grid,
                         lambda_enet_grid=lambda_enet_grid,
                         tau_adalasso_grid=tau_adalasso_grid,
                         lambda_adalasso_grid=lambda_adalasso_grid)
    
    # Test global_update function
    print("\n" + "="*60)
    print("TESTING INTERFACE global_update FUNCTION")
    print("="*60)
    
    try:
        # Call the complete global update method
        print("Running complete global variable selection...")
        global_update_results = interface.global_update(
            event_driven_backbone_time_bar_data=df_edlob,
            physical_time_driven_backbone_time_bar_data=df_pt_reset,
            quantile=0.005,
            target_col_name='micro_return',
            n_lags=3,  # Use fewer lags for faster testing
            validation_proportion=0.2,
            selection_threshold=0.3,  # Select variables significant in ≥30% of adapter bars
            tolerance_threshold=1e-6,  # Minimum coefficient magnitude in Adaptive Lasso
            ols_significance_level=0.05  # 5% significance level for OLS tests
        )
        
        print(f"✓ global_update completed successfully!")
        
        # Validate results structure
        assert 'selected_variables' in global_update_results, "Missing selected_variables"
        assert 'n_selected' in global_update_results, "Missing n_selected"
        assert 'n_total_variables' in global_update_results, "Missing n_total_variables"
        assert 'hyperparameter_results' in global_update_results, "Missing hyperparameter_results"
        assert 'processing_stats' in global_update_results, "Missing processing_stats"
        assert 'variable_selection_info' in global_update_results, "Missing variable_selection_info"
        
        # Validate selected variables
        selected_vars = global_update_results['selected_variables']
        assert isinstance(selected_vars, list), "selected_variables should be a list"
        assert len(selected_vars) <= global_update_results['n_total_variables'], "Too many variables selected"
        
        # Validate processing stats
        stats = global_update_results['processing_stats']
        assert stats['processed_bars'] > 0, "No adapter bars were processed"
        assert stats['total_adapter_bars'] > 0, "No adapter bars found"
        
        # Print comprehensive results
        print("\n" + "="*50)
        print("GLOBAL UPDATE RESULTS")
        print("="*50)
        
        print(f"\n🎯 VARIABLE SELECTION SUMMARY:")
        print(f"   Selected variables: {global_update_results['n_selected']}")
        print(f"   Total variables: {global_update_results['n_total_variables']}")
        print(f"   Selection rate: {global_update_results['n_selected']/global_update_results['n_total_variables']:.1%}")
        print(f"   Selection threshold: {global_update_results['selection_threshold']:.1%}")
        print(f"   Tolerance threshold: {global_update_results['tolerance_threshold']:.2e}")
        
        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"   Total adapter bars: {stats['total_adapter_bars']}")
        print(f"   Successfully processed: {stats['processed_bars']}")
        print(f"   Failed to process: {stats['failed_bars']}")
        print(f"   Success rate: {stats['processed_bars']/stats['total_adapter_bars']:.1%}")
        print(f"   Min bars for selection: {stats['min_bars_for_selection']}")
        
        print(f"\n⚙️ HYPERPARAMETER SUMMARY:")
        hp = global_update_results['hyperparameter_results']
        enet = hp['elastic_net']
        ada = hp['adaptive_lasso']
        print(f"   Elastic Net: α={enet['best_alpha']:.3f}, λ={enet['best_lambda']:.3f}")
        print(f"   Adaptive Lasso: τ={ada['best_tau']:.3f}, λ={ada['best_lambda']:.3f}")
        
        print(f"\n🔍 SELECTED VARIABLES:")
        for i, var in enumerate(selected_vars[:10], 1):  # Show first 10
            var_info = global_update_results['variable_selection_info'][var]
            print(f"   {i:2d}. {var:<30} ({var_info['count']:2d}/{stats['processed_bars']} bars, {var_info['proportion']:.1%})")
        
        if len(selected_vars) > 10:
            print(f"   ... and {len(selected_vars) - 10} more variables")
        
        # Verify interface attributes were set correctly
        print(f"\n🔧 INTERFACE ATTRIBUTES VERIFICATION:")
        assert hasattr(interface, 'selected_variables'), "Interface missing selected_variables"
        assert interface.selected_variables == selected_vars, "Interface selected_variables not set correctly"
        assert hasattr(interface, 'best_alpha_enet'), "Interface missing best_alpha_enet"
        assert hasattr(interface, 'best_lambda_enet'), "Interface missing best_lambda_enet"
        assert hasattr(interface, 'best_tau_adalasso'), "Interface missing best_tau_adalasso"
        assert hasattr(interface, 'best_lambda_adalasso'), "Interface missing best_lambda_adalasso"
        print(f"   ✓ Selected variables: {len(interface.selected_variables)} variables stored")
        print(f"   ✓ Hyperparameters: all optimal values stored")
        
        # Verify extended covariate list was created
        assert hasattr(interface, 'extended_covariate_list'), "Interface missing extended_covariate_list"
        assert len(interface.extended_covariate_list) > 0, "Extended covariate list is empty"
        print(f"   ✓ Extended covariate list: {len(interface.extended_covariate_list)} features")
        
        # Verify data scaler was created
        assert hasattr(interface, 'data_scaler'), "Interface missing data_scaler"
        assert hasattr(interface, 'scaled_variables'), "Interface missing scaled_variables"
        print(f"   ✓ Data scaler: created for {len(interface.scaled_variables)} variables")
        
        print(f"\n✅ ALL global_update TESTS PASSED!")
        print(f"✅ Complete global variable selection workflow validated successfully!")
        
        # Test local_adaptation method
        print("\n" + "="*60)
        print("TESTING INTERFACE local_adaptation FUNCTION")
        print("="*60)
        
        test_local_adaptation_success = test_local_adaptation(interface, df_pt_reset)
        
        if test_local_adaptation_success:
            print(f"\n✅ ALL TESTS PASSED!")
            print(f"✅ Both global_update and local_adaptation workflows validated successfully!")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"✗ global_update test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_local_adaptation(interface: Interface, df_pt_reset: pd.DataFrame) -> bool:
    """Test the local_adaptation function in the Interface module."""
    
    try:
        print("Starting local_adaptation test...")
        
        # Prepare data for local adaptation test
        print("Preparing data for local adaptation...")
        
        # Get augmented data (we need to prepare it as done in global_update)
        augmented_data = interface._prepare_data(
            event_driven_backbone_time_bar_data=df_pt_reset,  # Use same data for both
            physical_time_driven_backbone_time_bar_data=df_pt_reset,
            quantile=0.005,
            target_col_name='micro_return',
            n_lags=3
        )
        
        # Get data for specific adapter bars
        adapter_bars = augmented_data['adapter_bar'].unique()
        print(f"Available adapter bars: {len(adapter_bars)}")
        
        # Test on multiple adapter bars
        test_adapter_bars = adapter_bars[:3]  # Test first 3 adapter bars
        local_adaptation_results = []
        
        for i, test_adapter_bar in enumerate(test_adapter_bars):
            print(f"\n--- Testing Local Adaptation {i+1}/{len(test_adapter_bars)} ---")
            print(f"Adapter Bar: {test_adapter_bar}")
            
            # Get data for this specific adapter bar
            adapter_bar_data = augmented_data[
                augmented_data['adapter_bar'] == test_adapter_bar
            ].copy()
            
            print(f"Adapter bar data shape: {adapter_bar_data.shape}")
            
            # Skip if insufficient data
            if len(adapter_bar_data) < 20:
                print(f"Skipping adapter bar {test_adapter_bar}: insufficient data ({len(adapter_bar_data)} samples)")
                continue
            
            # Run local adaptation
            try:
                local_result = interface.local_adaptation(
                    adapter_bar_data=adapter_bar_data,
                    target_col_name='micro_return',
                    significance_level=0.05
                )
                
                local_adaptation_results.append({
                    'adapter_bar': test_adapter_bar,
                    'result': local_result,
                    'success': True
                })
                
                print(f"✓ Local adaptation completed for adapter bar {test_adapter_bar}")
                
                # Validate results structure
                assert 'global_variables' in local_result, "Missing global_variables"
                assert 'local_selected_variables' in local_result, "Missing local_selected_variables"
                assert 'newly_selected_variables' in local_result, "Missing newly_selected_variables"
                assert 'final_variables' in local_result, "Missing final_variables"
                
                # Print detailed results for this adapter bar
                print(f"  Global variables: {local_result['n_global']}")
                print(f"  Local adaptive lasso selected: {local_result['n_local']}")
                print(f"  Newly selected: {local_result['n_newly_selected']}")
                print(f"  Final model variables: {local_result['n_final']}")
                
                if local_result['wald_test_result']:
                    wald = local_result['wald_test_result']
                    print(f"  Wald test p-value: {wald['p_value']:.6f}")
                    print(f"  Wald test significant: {wald['is_significant']}")
                else:
                    print(f"  No Wald test performed (no new variables)")
                
            except Exception as e:
                print(f"✗ Local adaptation failed for adapter bar {test_adapter_bar}: {e}")
                local_adaptation_results.append({
                    'adapter_bar': test_adapter_bar,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        # Summarize results
        successful_tests = sum(1 for r in local_adaptation_results if r['success'])
        total_tests = len(local_adaptation_results)
        
        print(f"\n" + "="*50)
        print(f"LOCAL ADAPTATION TEST SUMMARY")
        print(f"="*50)
        print(f"Total adapter bars tested: {total_tests}")
        print(f"Successful tests: {successful_tests}")
        print(f"Failed tests: {total_tests - successful_tests}")
        print(f"Success rate: {successful_tests/total_tests:.1%}" if total_tests > 0 else "No tests performed")
        
        if successful_tests > 0:
            # Analyze patterns across successful tests
            global_var_counts = []
            local_var_counts = []
            new_var_counts = []
            final_var_counts = []
            wald_significant_count = 0
            wald_tests_performed = 0
            
            for result_data in local_adaptation_results:
                if result_data['success']:
                    result = result_data['result']
                    global_var_counts.append(result['n_global'])
                    local_var_counts.append(result['n_local'])
                    new_var_counts.append(result['n_newly_selected'])
                    final_var_counts.append(result['n_final'])
                    
                    if result['wald_test_result']:
                        wald_tests_performed += 1
                        if result['wald_test_result']['is_significant']:
                            wald_significant_count += 1
            
            print(f"\n📊 PATTERNS ACROSS SUCCESSFUL TESTS:")
            print(f"   Global variables (consistent): {global_var_counts[0] if global_var_counts else 0}")
            print(f"   Local adaptive lasso selected: {np.mean(local_var_counts):.1f} ± {np.std(local_var_counts):.1f}")
            print(f"   Newly selected variables: {np.mean(new_var_counts):.1f} ± {np.std(new_var_counts):.1f}")
            print(f"   Final model variables: {np.mean(final_var_counts):.1f} ± {np.std(final_var_counts):.1f}")
            
            if wald_tests_performed > 0:
                print(f"   Wald tests performed: {wald_tests_performed}/{successful_tests}")
                print(f"   Wald tests significant: {wald_significant_count}/{wald_tests_performed} ({wald_significant_count/wald_tests_performed:.1%})")
            else:
                print(f"   No Wald tests performed (no new variables selected)")
        
        # Check if at least some tests were successful
        if successful_tests == 0:
            print("✗ No local adaptation tests were successful")
            return False
        elif successful_tests < total_tests:
            print(f"⚠️ Some tests failed, but {successful_tests} were successful")
        
        print(f"\n✅ LOCAL ADAPTATION TESTS COMPLETED!")
        print(f"✅ Local adaptation workflow validated successfully!")
        return True
        
    except Exception as e:
        print(f"✗ local_adaptation test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()