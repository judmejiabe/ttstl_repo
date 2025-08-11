from ttstl.utils.data_downloader import import_to_pandas
from ttstl.utils.data_preprocessor import preprocess_mbo_data_cnn_lstm, get_datasets_cnn_lstm

def main():
    # Specify your credentials
    credentials_path = 'credentials.json'

    # Specify your bucket & blob
    bucket_name = 'evolving-microstructural-tk-data'
    blob_name = "data.nosync/Recorded_Synthetic_Data/01102025_short_event_driven_recorded_lob_1.parquet"

    # Download from the bucket using credentials
    df = import_to_pandas(
        blob_name=blob_name,
        bucket_name=bucket_name,
        credentials_path=credentials_path)

    standardized_data, target = preprocess_mbo_data_cnn_lstm(df, delta=100, quantile = None)

    print(f"Original data shape: {df.shape}")
    print(f"Standardized data shape: {standardized_data.shape}")
    print(f"Target shape: {target.shape}")
    
    print("First few rows of standardized data and target:")
    print(standardized_data.head())
    print(target.head())

if __name__ == "__main__":
    main()