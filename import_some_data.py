#!/usr/bin/env python3
import os
from pathlib import Path

# 1. Import the downloader functions
from ttstl.utils.data_downloader import (get_storage_client, 
                                         download_from_bucket, 
                                         import_to_pandas)

def main():
    # 2. (Option A) Rely on your environment variable:
    #    Make sure you’ve already done:
    #      export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
    #
    #    In that case you don’t need to pass credentials_path below:
    credentials_path = 'credentials.json'

    # 3. Specify your bucket & blob
    bucket_name = 'evolving-microstructural-tk-data'
    blob_name = "data.nosync/Recorded_Synthetic_Data/01102025_short_event_driven_recorded_lob_1.parquet"
    destination_folder = Path("some_data")
    destination_folder.mkdir(exist_ok=True)
    local_path = destination_folder / "some_data.parquet"

    # 4. Download from the bucket using credentials
    download_from_bucket(
        blob_name=              blob_name,
        destination_file_name=  str(local_path),
        bucket_name=            bucket_name,
        credentials_path=      credentials_path
    )

    print(f"Finished downloading to {local_path.resolve()} using credentials")

    # 5. Download from the bucket using a client
    client = get_storage_client(credentials_path)
    download_from_bucket(
        blob_name=              blob_name,
        destination_file_name=  str(local_path),
        bucket_name=            bucket_name,
        client=                 client
    )

    print(f"Finished downloading to {local_path.resolve()} using client")

    # 5. Import to pandas DataFrame
    df = import_to_pandas(
        blob_name=              blob_name,
        bucket_name=            bucket_name,
        credentials_path=      credentials_path
    )

    print(df.head())

if __name__ == "__main__":
    main()