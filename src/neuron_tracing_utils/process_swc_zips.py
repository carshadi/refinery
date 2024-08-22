import argparse
import glob
import multiprocessing
import os
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List

import boto3
import numpy as np
import scyjava

from neuron_tracing_utils.resample import resample_tree
from neuron_tracing_utils.util import sntutil, swcutil
from neuron_tracing_utils.util.java import snt


def unzip_file(zip_path: str, extract_path: str = None) -> None:
    """
    Unzips a given zip file.

    Parameters:
        zip_path (str): The path of the zip file to be unzipped.
        extract_path (str, optional): The path where the unzipped files
        should be placed.
            If not provided, the files will be unzipped to the same
            directory as the zip file.

    Returns:
        None
    """
    if not extract_path:
        extract_path = os.path.dirname(zip_path)

    if not os.path.isfile(zip_path):
        print(f"No file found at {zip_path}")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def process_swc_file(
    swc_path: str,
    length_threshold: int = 200,
    voxel_size: tuple = (1, 1, 1),
    node_spacing: int = 20,
    radius: float = 1.0,
    bbox: np.array = None
) -> Any:
    """
    Processes an SWC file, performs resampling and filtering based on cable
    length.

    Parameters:
        swc_path (str): The path of the SWC file to be processed.
        length_threshold (int, optional): The minimum length of cable to
        consider, default is 200.
        voxel_size (tuple, optional): The scaling factor for each dimension,
        default is (1, 1, 1).
        node_spacing (int, optional): The spacing between nodes after
        resampling, default is 20.
        radius (float, optional): The radius to be set for the resampled
        tree, default is 1.0.
        bbox (np.array, optional): The bounding box to filter the SWC file,

    Returns:
        snt.Tree: The processed Tree object if it passes the length
        threshold, else None.
    """
    arr = swcutil.swc_to_ndarray(swc_path)
    arr[:, 1] = 2

    if bbox is not None:
        points = arr[:, 2:5]
        if not np.any(np.all((points >= bbox[0]) & (points <= bbox[1]), axis=1)):
            return None

    g = sntutil.ndarray_to_graph(arr)

    tree = g.getTree()
    tree.scale(*voxel_size)
    if node_spacing > 0:
        resample_tree(tree, node_spacing)
    tree.setRadii(radius)

    analyzer = snt.TreeAnalyzer(tree)
    if tree.getNodes().size() == 0 or analyzer.getCableLength() < length_threshold:
        return None

    return tree


def process_zip_file(
    zip_path: str,
    output_dir: str,
    length_threshold: int,
    voxel_size: tuple,
    node_spacing: int,
    radius: float,
    bbox: np.array = None
) -> int:
    """
    Processes the files in a given zip file and saves the processed files to
    the output directory.

    Parameters:
        zip_path (str): The path of the zip file.
        output_dir (str): The directory where the processed files will be
        saved.
        length_threshold (int): The minimum length of cable to consider.
        voxel_size (tuple): The scaling factor for each dimension.
        node_spacing (int): The spacing between nodes after resampling.
        radius (float): The radius to be set for the resampled tree.
        bbox (np.array, optional): The bounding box to filter the SWC file.

    Returns:
        int: The number of files filtered out.
    """
    scyjava.start_jvm()

    filtered_count = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        unzip_file(zip_path, temp_dir)

        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)

                tree = process_swc_file(
                    file_path,
                    length_threshold,
                    voxel_size,
                    node_spacing,
                    radius,
                    bbox
                )
                if tree is not None:
                    tree.saveAsSWC(os.path.join(output_dir, file))
                else:
                    filtered_count += 1

    return filtered_count


def process_all_zip_files(
    input_dir: str,
    output_dir: str,
    length_threshold: int,
    voxel_size: tuple,
    node_spacing: int,
    radius: float,
    max_workers: int = None,
    bbox: np.array = None
) -> int:
    """
    Processes all zip files in the input directory using multiple workers
    and saves the processed files.

    Parameters:
        input_dir (str): The directory containing the zip files.
        output_dir (str): The directory where the processed files will be
        saved.
        length_threshold (int): The minimum length of cable to consider.
        voxel_size (tuple): The scaling factor for each dimension.
        node_spacing (int): The spacing between nodes after resampling.
        radius (float): The radius to be set for the resampled tree.
        max_workers (int, optional): The maximum number of workers to use,
        default is the number of processors.
        bbox (np.array, optional): The bounding box to filter the SWC file.

    Returns:
        int: The number of files filtered out.
    """
    zip_files = glob.glob(f"{input_dir}/*.zip")
    total_files = len(zip_files)
    print("Total zip files: ", total_files)

    filtered_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for zip_file in zip_files:
            futures.append(
                executor.submit(
                    process_zip_file,
                    zip_file,
                    output_dir,
                    length_threshold,
                    voxel_size,
                    node_spacing,
                    radius,
                    bbox
                )
            )
        for i, fut in enumerate(futures):
            print(f"Processing zip {i + 1}/{total_files}")
            try:
                filtered_count += fut.result()
            except Exception as e:
                print(e)

    return filtered_count


def zip_files(file_paths: List[str], zip_path: str):
    """
    Creates a zip file from the given file paths.

    Parameters:
        file_paths (List[str]): Paths of the files to be zipped.
        zip_path (str): Path for the output zip file.
    """
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in file_paths:
            zipf.write(file, os.path.basename(file))


def upload_file_to_s3(
    file_name: str, bucket: str, folder: str, object_name: str = None
):
    """
    Upload a file to an S3 bucket in a specified folder.

    Parameters:
        file_name (str): File to upload
        bucket (str): Bucket to upload to
        folder (str): Folder within the bucket to upload the file
        object_name (str, optional): S3 object name. If not specified,
        the file_name is used

    Returns:
        bool: True if file was uploaded, else False
    """
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = boto3.client("s3")
    full_object_name = f"{folder.rstrip('/')}/{object_name}"

    try:
        s3_client.upload_file(file_name, bucket, full_object_name)
    except Exception as e:
        print(e)
        return False
    return True


def upload_to_s3(
    zip_paths: List[str], bucket: str, folder: str, n_workers: int = 8
):
    """
    Uploads multiple files to a specific folder in an S3 bucket in parallel.

    Parameters:
        zip_paths (List[str]): Paths of the zip files to upload.
        bucket (str): S3 bucket name.
        folder (str): Folder within the bucket where files will be uploaded.
        n_workers (int, optional): Number of workers to use.
    """
    with multiprocessing.Pool(n_workers) as pool:
        results = [
            pool.apply_async(upload_file_to_s3, (zip_path, bucket, folder))
            for zip_path in zip_paths
        ]
        for r in results:
            r.get()


def create_zips(
    files: list, output_dir: str, n_workers: int = 8
) -> List[str]:
    """
    Creates zip files from the processed SWC files in parallel.

    Parameters:
        files: List[str]: Paths of the processed SWC files.
        input_dir (str): Directory containing processed SWC files.
        output_dir (str): Directory to save the zip files.
        n_workers (int, optional): Number of zip files to create.

    Returns:
        List[str]: Paths of the created zip files.
    """

    files_per_zip = len(files) // n_workers

    zip_paths = [
        os.path.join(output_dir, f"swcs_{i}.zip") for i in range(n_workers)
    ]
    file_chunks = [
        files[i * files_per_zip: (i + 1) * files_per_zip]
        for i in range(n_workers)
    ]

    with multiprocessing.Pool(n_workers) as pool:
        pool.starmap(zip_files, zip(file_chunks, zip_paths))

    return zip_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process SWC files in zip format."
    )
    parser.add_argument(
        "-i",
        help="The input directory containing zip files.",
    )
    parser.add_argument(
        "-o",
        help="The output directory to save processed files.",
    )
    parser.add_argument("--s3-bucket", help="S3 bucket name.", default=None)
    parser.add_argument(
        "--s3-folder",
        help="Folder within the bucket to upload the files.",
        default=None
    )
    parser.add_argument(
        "--length_threshold",
        type=int,
        default=0,
        help="The minimum length of cable to consider.",
    )
    parser.add_argument(
        "--voxel_size",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        help="The scaling factor for each dimension.",
    )
    parser.add_argument(
        "--node_spacing",
        type=int,
        default=0,
        help="The spacing between nodes after resampling.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="The radius to be set for the resampled tree.",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Only keep swcs within the bounding box.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="The number of worker processes to use.",
    )
    args = parser.parse_args()

    os.makedirs(args.o, exist_ok=True)

    swc_dir = os.path.join(args.o, "swcs")
    os.makedirs(swc_dir, exist_ok=True)

    zip_dir = os.path.join(args.o, "zips")
    os.makedirs(zip_dir, exist_ok=True)

    bbox = None
    if args.bbox is not None:
        bbox = [int(x.strip()) for x in args.bbox.split(",")]
        bbox = np.array([bbox[0:3], bbox[3:6]])
        print("Bounding box: ", bbox)

    filtered_count = process_all_zip_files(
        args.i,
        swc_dir,
        args.length_threshold,
        args.voxel_size,
        args.node_spacing,
        args.radius,
        args.workers,
        bbox=bbox
    )

    files = [f for f in glob.glob(f"{swc_dir}/*.swc")]
    total_files = len(files)

    print("Total swcs before filtering: ", total_files + filtered_count)
    print("Total swcs after filtering: ", total_files)
    print("Total swcs filtered: ", filtered_count)

    if args.s3_bucket is not None:
        # Create zip files from processed SWC files
        print("Creating zip files...")
        zip_paths = create_zips(files, zip_dir, args.workers)
        # Upload zip files to S3
        print("Uploading zip files to S3...")
        upload_to_s3(zip_paths, args.s3_bucket, args.s3_folder, args.workers)
