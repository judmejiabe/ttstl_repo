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

    backbone = Backbone(model=model, 
                        delta_adapter=100000.0, 
                        delta_sample=100.0)

    criterion = torch.nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    for i, var in enumerate(df_pt.columns):
        print(f"Column {i}: {var}")

    na_counts = df_pt.isnull().sum()
    total_na = na_counts.sum()
    print("NA values per column:")
    print(na_counts)
    print(f"Total NA values: {total_na}")
    print(df_pt.info())
    print(df_pt.head())
    
    # Test the global_update function
    print("\n" + "="*50)
    print("TESTING GLOBAL_UPDATE FUNCTION")
    print("="*50)
    
    # Reset the dataframe index to get timestamp as a column for event-driven data
    df_edlob_for_update = df_edlob.copy()
    df_pt_for_update = df_pt.reset_index()  # Reset index to get timestamp as column
    
    print(f"Event-driven data shape: {df_edlob_for_update.shape}")
    print(f"Physical time-driven data shape: {df_pt_for_update.shape}")
    print(f"Event-driven data time range: {df_edlob_for_update['timestamp'].min():.2f} to {df_edlob_for_update['timestamp'].max():.2f}")
    print(f"Physical time-driven data time range: {df_pt_for_update['timestamp'].min():.2f} to {df_pt_for_update['timestamp'].max():.2f}")
    
    # Use the same trained model for global_update test (already trained on small data)
    backbone_update = backbone  # Use the already trained backbone
    
    try:
        # Run global_update with neural covariate extraction and stationarity analysis
        print("\nRunning global_update with neural covariate extraction and stationarity analysis...")
        print("📊 Correlation matrix plots will be saved to runs/test_runs/")
        
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
            parsimonious_mode=False,  # Use hidden activations
            correlation_threshold=0.95
        )
        
        print(f"\n✓ Global update completed successfully!")
        
        # Print information about the computed correlation matrices
        print("\n" + "="*50)
        print("CORRELATION MATRIX ANALYSIS RESULTS")
        print("="*50)
        
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
        
        print("="*50)
        

    except Exception as e:
        print(f"\n✗ Global update failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("="*50)
    

if __name__ == "__main__":
    main()
