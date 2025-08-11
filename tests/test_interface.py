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
    """Test the _update_hyperparameters function in the Interface module."""
    
    print("Starting Interface _update_hyperparameters test...")
    
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
                        delta_sample=100.0)

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
            parsimonious_mode=False,
            correlation_threshold=0.95
        )
        print("✓ Backbone global_update completed successfully")
    except Exception as e:
        print(f"✗ Backbone global_update failed: {e}")
        return False
    
    # Initialize Interface with complete hyperparameter grids
    print("Initializing Interface module...")
    
    # Set up hyperparameter grids for comprehensive testing
    alpha_enet_grid = [0.1, 0.5, 0.9]  # Smaller grid for faster testing
    lambda_enet_grid = [0.01, 0.1, 1.0]
    tau_adalasso_grid = [0.5, 1.0, 2.0]
    lambda_adalasso_grid = [0.01, 0.1, 1.0]
    
    interface = Interface(backbone_module=backbone,
                         alpha_enet_grid=alpha_enet_grid,
                         lambda_enet_grid=lambda_enet_grid,
                         tau_adalasso_grid=tau_adalasso_grid,
                         lambda_adalasso_grid=lambda_adalasso_grid)
    
    # Test _update_hyperparameters function
    print("\n" + "="*60)
    print("TESTING INTERFACE _update_hyperparameters FUNCTION")
    print("="*60)
    
    try:
        # Call the complete hyperparameter tuning method
        print("Running complete hyperparameter tuning...")
        hyperparameter_results = interface._update_hyperparameters(
            event_driven_backbone_time_bar_data=df_edlob,
            physical_time_driven_backbone_time_bar_data=df_pt_reset,
            quantile=0.005,
            parsimonious_mode=False,
            target_col_name='micro_return',
            n_lags=3,  # Use fewer lags for faster testing
            validation_proportion=0.2
        )
        
        print(f"✓ _update_hyperparameters completed successfully!")
        
        # Validate results structure
        assert 'elastic_net' in hyperparameter_results, "Missing elastic_net results"
        assert 'adaptive_lasso' in hyperparameter_results, "Missing adaptive_lasso results"
        assert 'data_info' in hyperparameter_results, "Missing data_info results"
        
        # Validate Elastic Net results
        enet_results = hyperparameter_results['elastic_net']
        assert 'best_alpha' in enet_results, "Missing best_alpha in elastic_net results"
        assert 'best_lambda' in enet_results, "Missing best_lambda in elastic_net results"
        assert enet_results['best_alpha'] in alpha_enet_grid, f"Best alpha {enet_results['best_alpha']} not in grid"
        assert enet_results['best_lambda'] in lambda_enet_grid, f"Best lambda {enet_results['best_lambda']} not in grid"
        
        # Validate Adaptive Lasso results
        ada_results = hyperparameter_results['adaptive_lasso']
        assert 'best_tau' in ada_results, "Missing best_tau in adaptive_lasso results"
        assert 'best_lambda' in ada_results, "Missing best_lambda in adaptive_lasso results"
        assert 'best_bic' in ada_results, "Missing best_bic in adaptive_lasso results"
        assert ada_results['best_tau'] in tau_adalasso_grid, f"Best tau {ada_results['best_tau']} not in grid"
        assert ada_results['best_lambda'] in lambda_adalasso_grid, f"Best lambda {ada_results['best_lambda']} not in grid"
        
        # Validate data info
        data_info = hyperparameter_results['data_info']
        assert 'total_samples' in data_info, "Missing total_samples in data_info"
        assert 'training_samples' in data_info, "Missing training_samples in data_info"
        assert 'validation_samples' in data_info, "Missing validation_samples in data_info"
        assert 'n_features' in data_info, "Missing n_features in data_info"
        assert 'n_lags' in data_info, "Missing n_lags in data_info"
        
        # Print comprehensive results
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING RESULTS")
        print("="*50)
        
        print(f"\n🎯 BEST HYPERPARAMETERS:")
        print(f"   Elastic Net:")
        print(f"     Alpha (mixing): {enet_results['best_alpha']:.3f}")
        print(f"     Lambda (regularization): {enet_results['best_lambda']:.3f}")
        print(f"   Adaptive Lasso:")
        print(f"     Tau (weight power): {ada_results['best_tau']:.3f}")
        print(f"     Lambda (regularization): {ada_results['best_lambda']:.3f}")
        print(f"     Best BIC: {ada_results['best_bic']:.6f}")
        
        print(f"\n📊 DATA INFORMATION:")
        print(f"   Total samples: {data_info['total_samples']}")
        print(f"   Training samples: {data_info['training_samples']}")
        print(f"   Validation samples: {data_info['validation_samples']}")
        print(f"   Number of features: {data_info['n_features']}")
        print(f"   Number of lags: {data_info['n_lags']}")
        print(f"   Validation bars: {len(data_info['validation_bars'])}")
        
        print(f"\n⚙️ HYPERPARAMETER GRIDS TESTED:")
        print(f"   Elastic Net: {len(alpha_enet_grid)} × {len(lambda_enet_grid)} = {len(alpha_enet_grid) * len(lambda_enet_grid)} combinations")
        print(f"   Adaptive Lasso: {len(tau_adalasso_grid)} × {len(lambda_adalasso_grid)} = {len(tau_adalasso_grid) * len(lambda_adalasso_grid)} combinations")
        
        # Verify interface attributes were set correctly
        print(f"\n🔧 INTERFACE ATTRIBUTES VERIFICATION:")
        assert interface.best_alpha_enet == enet_results['best_alpha'], "Interface best_alpha_enet not set correctly"
        assert interface.best_lambda_enet == enet_results['best_lambda'], "Interface best_lambda_enet not set correctly"
        assert interface.best_tau_adalasso == ada_results['best_tau'], "Interface best_tau_adalasso not set correctly"
        assert interface.best_lambda_adalasso == ada_results['best_lambda'], "Interface best_lambda_adalasso not set correctly"
        print(f"   ✓ best_alpha_enet: {interface.best_alpha_enet}")
        print(f"   ✓ best_lambda_enet: {interface.best_lambda_enet}")
        print(f"   ✓ best_tau_adalasso: {interface.best_tau_adalasso}")
        print(f"   ✓ best_lambda_adalasso: {interface.best_lambda_adalasso}")
        
        # Verify extended covariate list was created
        assert hasattr(interface, 'extended_covariate_list'), "Interface missing extended_covariate_list"
        assert len(interface.extended_covariate_list) > 0, "Extended covariate list is empty"
        print(f"   ✓ Extended covariate list created with {len(interface.extended_covariate_list)} features")
        
        # Verify data scaler was created
        assert hasattr(interface, 'data_scaler'), "Interface missing data_scaler"
        assert hasattr(interface, 'scaled_variables'), "Interface missing scaled_variables"
        print(f"   ✓ Data scaler created for {len(interface.scaled_variables)} variables")
        
        print(f"\n✅ ALL _update_hyperparameters TESTS PASSED!")
        print(f"✅ Complete hyperparameter tuning workflow validated successfully!")
        return True
        
    except Exception as e:
        print(f"✗ _update_hyperparameters test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()