import numpy as np
import torch

from torch.utils.data import DataLoader

from ttstl.backbone import Backbone
from ttstl.architectures.cnn_lstm import CNNLSTM
from ttstl.utils.data_downloader import import_to_pandas
from ttstl.utils.data_preprocessor import get_dataset_cnn_lstm


def main():

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

    # Download from the bucket using credentials
    df_train = import_to_pandas(
        blob_name=blob_name_train,
        bucket_name=bucket_name,
        credentials_path=credentials_path)
    
    df_validation = import_to_pandas(
        blob_name=blob_name_validation,
        bucket_name=bucket_name,
        credentials_path=credentials_path)

    dataset_train = get_dataset_cnn_lstm(df_train, delta=100, quantile=None)
    dataset_validation = get_dataset_cnn_lstm(df_validation, delta=100, quantile=None)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=4)
    validation_loader = DataLoader(dataset_validation, batch_size=32, shuffle=False, num_workers=4)

    criterion = torch.nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Test both parsimonious modes
    for parsimonious_mode in [False, True]:
        print(f"\n{'='*80}")
        print(f"TESTING PARSIMONIOUS_MODE = {parsimonious_mode}")
        print(f"{'='*80}")
        
        # Create backbone with current parsimonious mode
        backbone = Backbone(model=model, 
                            delta_adapter=100000.0, 
                            delta_sample=100.0,
                            parsimonious_mode=parsimonious_mode)

        print(f"✓ Backbone created with parsimonious_mode = {backbone.parsimonious_mode}")

        # Train the model with small data first
        print("Training model with small dataset...")
        backbone.batch_gradient_descent(train_loader, 
                       criterion, 
                       optimizer, 
                       validation_loader=validation_loader,
                       epochs=2)
        
        # Load the larger datasets for testing global_update
        blob_name_edlob = 'data.nosync/Recorded_Synthetic_Data/01102025_long_event_driven_recorded_lob_0.parquet'
        blob_name_ptlob = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_driven_recorded_lob_0.parquet'
        blob_name_ptme = 'data.nosync/Recorded_Synthetic_Data/01102025_long_physical_time_market_event_data_0.parquet'

        df_edlob = import_to_pandas(
            blob_name=blob_name_edlob,
            bucket_name=bucket_name,
            credentials_path=credentials_path)

        # This is a patch only for testing purposes, the dataset will be passed
        # without the need to do manipulations.
        delta = 100
        df_edlob['sampling_time_bar'] = (np.rint((df_edlob['timestamp']) / delta).astype(int) + 1) * delta

        for k in range(10):
            df_edlob[f'OFI{k}'] = df_edlob[f'BIDof{k}'] - df_edlob[f'ASKof{k}']

        ofi_columns = [f'OFI{k}' for k in range(10)]

        ofi_df = df_edlob.groupby('sampling_time_bar')[ofi_columns].sum().reset_index()

        ofi_df = ofi_df.rename(columns={'sampling_time_bar': 'timestamp'})

        df_ptlob = import_to_pandas(
            blob_name=blob_name_ptlob,
            bucket_name=bucket_name,
            credentials_path=credentials_path)

        df_ptme = import_to_pandas(
            blob_name=blob_name_ptme,
            bucket_name=bucket_name,
            credentials_path=credentials_path)

        
        df_pt = df_ptlob.merge(df_ptme, on = 'timestamp', how = 'inner')
        df_pt = df_pt.merge(ofi_df, on = 'timestamp', how = 'inner')
        
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
        
        df_pt.set_index('timestamp', inplace=True)

        print(f"Testing with {len(df_pt.columns)} variables in physical time data")

        na_counts = df_pt.isnull().sum()
        total_na = na_counts.sum()
        print(f"Total NA values: {total_na}")
        
        # Test the global_update function
        print(f"\nTESTING GLOBAL_UPDATE FUNCTION (parsimonious_mode={parsimonious_mode})")
        print("-" * 60)
    
        # Test the global_update function
        print(f"\nTESTING GLOBAL_UPDATE FUNCTION (parsimonious_mode={parsimonious_mode})")
        print("-" * 60)
        
        # Reset the dataframe index to get timestamp as a column for event-driven data
        df_edlob_for_update = df_edlob.copy()
        df_pt_for_update = df_pt.reset_index()  # Reset index to get timestamp as column
        
        print(f"Event-driven data shape: {df_edlob_for_update.shape}")
        print(f"Physical time-driven data shape: {df_pt_for_update.shape}")
        print(f"Event-driven data time range: {df_edlob_for_update['timestamp'].min():.2f} to {df_edlob_for_update['timestamp'].max():.2f}")
        print(f"Physical time-driven data time range: {df_pt_for_update['timestamp'].min():.2f} to {df_pt_for_update['timestamp'].max():.2f}")
        
        # Use the already trained backbone
        backbone_update = backbone
        
        try:
            # Run global_update with neural covariate extraction and stationarity analysis
            print(f"\nRunning global_update with parsimonious_mode={parsimonious_mode}...")
            
            best_validation_loss = backbone_update.global_update(
                event_driven_backbone_time_bar_data=df_edlob_for_update,
                physical_time_driven_backbone_time_bar_data=df_pt_for_update,
                criterion=criterion,
                optimizer=optimizer,
                epochs=0,  # Set to 0 for testing as requested
                validation_proportion=0.1,
                buffer_proportion=0.05,
                batch_size=64,  # Larger batch size for efficiency
                # Stationarity analysis parameters
                adf_rejection_rate_threshold=0.8,
                kpss_verification_threshold=0.2,
                na_verification_threshold=0.1,
                kpss_drop=True,
                # Neural covariate parameters
                correlation_threshold=0.95
            )
            
            print(f"\n✓ Global update completed successfully!")
            
            # Save correlation heatmaps to disk
            print(f"\n📊 Saving correlation matrix heatmaps to runs/test_runs/")
            backbone_update.save_correlation_heatmaps(save_dir="runs/test_runs")
            
            # Print information about the computed correlation matrices
            print(f"\nCORRELATION MATRIX ANALYSIS RESULTS (parsimonious_mode={parsimonious_mode})")
            print("-" * 60)
            
            if parsimonious_mode:
                # In parsimonious mode, we expect only the neural_covariates_correlation matrix
                if backbone_update.neural_covariates_correlation is not None:
                    print(f"✓ Parsimonious correlation matrix computed: {backbone_update.neural_covariates_correlation.shape}")
                    print(f"  Shape: {backbone_update.neural_covariates_correlation.shape}")
                    print(f"  Range: [{backbone_update.neural_covariates_correlation.min().min():.3f}, {backbone_update.neural_covariates_correlation.max().max():.3f}]")
                    
                    # Print variable order information
                    cols = list(backbone_update.neural_covariates_correlation.columns)
                    extra_vars = [col for col in cols if 'micro_return' in col]
                    neural_vars = [col for col in cols if 'forward_pass' in col or 'hidden_activations' in col]
                    stationary_vars = [col for col in cols if col not in extra_vars and col not in neural_vars]
                    
                    print(f"  Variable structure: {len(extra_vars)} extra + {len(neural_vars)} neural + {len(stationary_vars)} stationary")
                    if extra_vars:
                        print(f"  Extra variables: {extra_vars}")
                    if neural_vars[:3]:  # Show first 3
                        print(f"  Neural variables (first 3): {neural_vars[:3]}")
                    if stationary_vars[:3]:  # Show first 3
                        print(f"  Stationary variables (first 3): {stationary_vars[:3]}")
                        
                else:
                    print("✗ Parsimonious correlation matrix not computed")
                
                # These should be None in parsimonious mode
                if backbone_update.manually_extracted_covariates_correlation is None:
                    print("✓ Manually extracted correlation matrix correctly set to None")
                else:
                    print("✗ Manually extracted correlation matrix should be None in parsimonious mode")
                    
                if backbone_update.cross_correlation_matrix is None:
                    print("✓ Cross-correlation matrix correctly set to None")
                else:
                    print("✗ Cross-correlation matrix should be None in parsimonious mode")
            else:
                # In non-parsimonious mode, we expect all three matrices
                if backbone_update.neural_covariates_correlation is not None:
                    print(f"✓ Neural + extra variables correlation matrix computed: {backbone_update.neural_covariates_correlation.shape}")
                    print(f"  Shape: {backbone_update.neural_covariates_correlation.shape}")
                    print(f"  Range: [{backbone_update.neural_covariates_correlation.min().min():.3f}, {backbone_update.neural_covariates_correlation.max().max():.3f}]")
                else:
                    print("✗ Neural + extra variables correlation matrix not computed")
                    
                if backbone_update.manually_extracted_covariates_correlation is not None:
                    print(f"✓ Stationary + extra variables correlation matrix computed: {backbone_update.manually_extracted_covariates_correlation.shape}")
                    print(f"  Shape: {backbone_update.manually_extracted_covariates_correlation.shape}")
                    print(f"  Range: [{backbone_update.manually_extracted_covariates_correlation.min().min():.3f}, {backbone_update.manually_extracted_covariates_correlation.max().max():.3f}]")
                else:
                    print("✗ Stationary + extra variables correlation matrix not computed")
                    
                if backbone_update.cross_correlation_matrix is not None:
                    print(f"✓ Cross-correlation matrix computed: {backbone_update.cross_correlation_matrix.shape}")
                    print(f"  Shape: {backbone_update.cross_correlation_matrix.shape}")
                    print(f"  Range: [{backbone_update.cross_correlation_matrix.min().min():.3f}, {backbone_update.cross_correlation_matrix.max().max():.3f}]")
                else:
                    print("✗ Cross-correlation matrix not computed")
            
            print("-" * 60)
            

        except Exception as e:
            print(f"\n✗ Global update failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("PARSIMONIOUS MODE CORRELATION MATRIX TESTING COMPLETED!")
    print(f"{'='*80}")
    

if __name__ == "__main__":
    main()
