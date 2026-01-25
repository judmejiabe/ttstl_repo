__author__ = "Juan Diego Mejia Becerra"
__email__ = "juanmejia.7565@gmail.com"

import os
from typing import Iterator, Optional, Union

import gcsfs
import pandas as pd
import pyarrow.parquet as pq
from google.cloud import storage


def get_storage_client(credentials_path: Optional[str] = None) -> storage.Client:
    """
    Lazily create a GCS client. If credentials_path is provided, set the environment var.
    
    Args:
        credentials_path: Optional path to the Google Cloud credentials JSON file
        
    Returns:
        A Google Cloud Storage client instance
    """
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    return storage.Client()


def download_from_bucket(
    blob_name: str,
    destination_file_name: str,
    bucket_name: str,
    client: Optional[storage.Client] = None,
    credentials_path: Optional[str] = None,
) -> None:
    """
    Download a blob from Google Cloud Storage to a local file.
    
    Args:
        blob_name: Name of the blob in the bucket
        destination_file_name: Local path where the file will be saved
        bucket_name: Name of the Google Cloud Storage bucket
        credentials_path: Optional path to credentials JSON file
        client: Optional pre-configured storage client
    """
    client = client or get_storage_client(credentials_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {blob_name} downloaded to {destination_file_name}")


def import_to_pandas(
    blob_name: str,
    bucket_name: str,
    credentials_path: str,
    columns: Optional[list] = None,
    chunksize: Optional[int] = None,
) -> Union[Iterator[pd.DataFrame], pd.DataFrame]:
    """
    Stream or load data from a blob in Google Cloud Storage directly into pandas DataFrame(s).

    Supports CSV and Parquet files. For large files, supports chunked iteration.
    
    
    Args:
        blob_name: Name of the blob in the bucket
        bucket_name: Name of the Google Cloud Storage bucket
        credentials_path: Optional path to credentials JSON file
        columns: Optional list of columns to load
        chunksize: Optional number of rows (CSV) or batch size (Parquet) for chunked iteration
        
    Returns:
        A pandas DataFrame if chunksize is None, else an iterator of pandas DataFrames.
    """

    fs = gcsfs.GCSFileSystem(token=credentials_path)
    gcs_path = f"{bucket_name}/{blob_name}"

    if blob_name.endswith(".csv"):
        return pd.read_csv(
            f"gs://{gcs_path}",
            storage_options={"token": credentials_path},
            usecols=columns,
            chunksize=chunksize,
        )
    elif blob_name.endswith(".parquet"):
        if chunksize:
            with fs.open(gcs_path, "rb") as f:
                pq_file = pq.ParquetFile(f)
                return (
                    batch.to_pandas()
                    for batch in pq_file.iter_batches(
                        batch_size=chunksize, columns=columns
                    )
                )
        else:
            with fs.open(gcs_path, "rb") as f:
                return pd.read_parquet(f, engine="pyarrow", columns=columns)
    else:
        raise ValueError("Unsupported file type")
