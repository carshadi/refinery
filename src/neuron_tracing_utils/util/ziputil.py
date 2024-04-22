import argparse
import multiprocessing
import os
from zipfile import ZipFile

from google.cloud import storage


def download_blob(bucket_name: str, download_dir: str, blob_name: str) -> str:
    """
    Download a blob from the Google Cloud Storage (GCS) to a local directory.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket.
    download_dir : str
        Local directory to download the file.
    blob_name : str
        Name of the blob to download.

    Returns
    -------
    str
        The path to the downloaded file.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_path = os.path.join(download_dir, os.path.basename(blob_name))

    try:
        blob.download_to_filename(local_path)
        print(f"Downloaded {local_path}")
        return local_path
    except Exception as e:
        print(f"Failed to download {local_path}: {e}")
        raise


def extract_zip(zip_path: str, extract_dir: str):
    """
    Extract a ZIP file to a specified local directory.

    Parameters
    ----------
    zip_path : str
        Path to the ZIP file to extract.
    extract_dir : str
        Directory where the contents of the ZIP will be extracted.

    Returns
    -------
    None
    """
    try:
        with ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(extract_dir)
            print(f"Extracted {zip_path} to {extract_dir}")
    except Exception as e:
        print(f"Failed to extract {zip_path}: {e}")
        raise


def download_extract_blob(
    bucket_name: str,
    download_dir: str,
    extract_dir: str,
    blob_name: str
):
    """
    Handles the downloading and extracting of a blob from Google Cloud Storage.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket.
    download_dir : str
        Local directory to download the file.
    extract_dir : str
        Local directory to extract the contents of the file.
    blob_name : str
        Name of the blob to be processed.

    Returns
    -------
    None
    """
    zip_path = download_blob(bucket_name, download_dir, blob_name)
    extract_zip(zip_path, extract_dir)


def download_extract_gcs_folder(
    bucket_name: str,
    prefix: str,
    download_dir: str,
    extract_dir: str
):
    """
    Main function to download and extract ZIP files from a GCS bucket in
    parallel.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket.
    prefix : str
        Prefix to filter the blobs within the bucket.
    download_dir : str
        Directory to download the ZIP files.
    extract_dir : str
        Directory to extract the contents of the ZIP files.

    Returns
    -------
    None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))
    zip_blobs = [blob.name for blob in blobs if
                 blob.name.lower().endswith('.zip')]
    jobs = [(bucket_name, download_dir, extract_dir, blob_name) for blob_name
            in zip_blobs]

    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(download_extract_blob, jobs)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments using argparse.

    Returns
    -------
    argparse.Namespace
        Namespace containing the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download and extract ZIP files from Google Cloud Storage."
    )
    parser.add_argument(
        "--zip-url",
        type=str,
        required=True,
        help="Full URL to the directory (e.g., 'gs://bucket_name/prefix')."
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        required=True,
        help="Directory to download the ZIP files."
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        required=True,
        help="Directory to extract the contents of the ZIP files."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.zip_url.startswith("gs://"):
        # TODO: support S3
        raise ValueError("Invalid GCS URL. It should start with 'gs://'")

    parts = args.zip_url[5:].split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.extract_dir, exist_ok=True)

    download_extract_gcs_folder(
        bucket_name,
        prefix,
        args.download_dir,
        args.extract_dir
    )


if __name__ == '__main__':
    main()
