import argparse
import concurrent.futures
import multiprocessing
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import s3fs
import scyjava
import zarr

from neuron_tracing_utils.util.java import snt


def group_swcs(file_names: List[str]) -> Dict[Tuple[str, str, str], List[str]]:
    """
    Group SWC files by neuron ID, sample, and tracer initials.

    Parameters
    ----------
    file_names : List[str]
        List of SWC file names.

    Returns
    -------
    Dict[Tuple[str, str, str], List[str]]
        Dictionary with keys as tuples of (neuron ID, sample, tracer initials)
        and values as lists of file paths that belong to the same group.
    """
    grouped_files = defaultdict(list)
    for file_path in file_names:
        file_name = os.path.basename(file_path)
        parts = file_name.split("-")
        neuron_id, sample, compartment, tracer_initials = (
            parts[0],
            parts[1],
            parts[2],
            parts[3]
        )
        group_key = (neuron_id, sample, compartment, tracer_initials)
        grouped_files[group_key].append(file_path)
    return grouped_files


def read_values(arr: Any, coords_chunk: np.ndarray) -> np.ndarray:
    """
    Read specific values from an array based on coordinates.

    Parameters
    ----------
    arr : Any
        5D array representing the full-resolution Zarr dataset.
    coords_chunk : np.ndarray
        The array of coordinates to read values from.

    Returns
    -------
    np.ndarray
        Array of values at the specified coordinates.
    """
    zs, ys, xs = coords_chunk[:, 0], coords_chunk[:, 1], coords_chunk[:, 2]
    return arr[0, 0, zs, ys, xs]


def swc_vals(
    arr: Any,
    swc: str,
    voxel_size: Tuple[float],
    num_threads: int = 1
) -> List[float]:
    """
    Calculate values from an SWC file using a specified voxel size and array.

    Parameters
    ----------
    arr : Any
        The array to process.
    swc : str
        Path to the SWC file.
    voxel_size : List[float]
        The size of each voxel.
    num_threads : int
        Number of threads to use for concurrent processing.

    Returns
    -------
    List[float]
        List of processed values.
    """
    coords = np.array(
        [
            [
                int(n.z / voxel_size[2]),
                int(n.y / voxel_size[1]),
                int(n.x / voxel_size[0])
            ] for n in snt.Tree(swc).getNodes()
        ]
    )

    all_vals = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads
    ) as executor:
        futures = []
        chunks = np.array_split(coords, num_threads)
        for chunk in chunks:
            futures.append(executor.submit(read_values, arr, chunk))
        for future in concurrent.futures.as_completed(futures):
            all_vals.extend(future.result())

    return all_vals


def process_swc(
    arr: Any,
    swc: str,
    output_dir: str,
    voxel_size: Tuple[float],
    num_threads: int = 1
):
    """
    Process an SWC file by saving calculated values based on the voxel size.

    Parameters
    ----------
    arr : Any
        Array to process.
    swc : str
        SWC file to process.
    output_dir : str
        Output directory to save the result.
    voxel_size : List[float]
        Voxel dimensions used for processing.
    num_threads : int
        Number of threads to use for concurrent processing.

    Returns
    -------
    None
    """
    vals = swc_vals(arr, swc, voxel_size, num_threads)
    np.save(os.path.join(output_dir, Path(swc).stem + ".npy"), vals)


def process_group(
    arr: Any,
    swcs: List[str],
    output_dir: str,
    voxel_size: Tuple[float],
    num_threads: int = 1
):
    """
    Process a group of SWC files.

    Parameters
    ----------
    arr : Any
        Array from which to read data.
    swcs : List[str]
        List of SWC file paths to process.
    output_dir : str
        Directory to save processed files.
    voxel_size : List[float]
        Voxel size for the SWCs.
    num_threads : int
        Number of threads to use for processing.

    Returns
    -------
    None
    """
    known_types = {"correct", "omit", "split"}

    for swc in swcs:
        print("Processing", swc)
        t0 = time.time()
        if any(swc_type in swc for swc_type in known_types):
            process_swc(arr, swc, output_dir, voxel_size, num_threads)
            print("Time taken:", time.time() - t0)
        else:
            raise ValueError(f"Unknown SWC type in file: {swc}")


def open_zarr_with_cache(
    path: str,
    cache_size: int = 1024 ** 3,
    region: str = "us-west-2"
) -> zarr.hierarchy.Group:
    """
    Open a Zarr file using an S3 path with caching.

    Parameters
    ----------
    path : str
        The path to the Zarr file.
    cache_size : int
        Size of the cache in bytes.

    Returns
    -------
    zarr.hierarchy.Group
        The Zarr group opened with the specified cache.
    """
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region))
    store = s3fs.S3Map(root=path, s3=s3, check=False)
    cache = zarr.LRUStoreCache(store, max_size=cache_size)
    z = zarr.group(store=cache, overwrite=False)
    return z


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the script.

    Returns
    -------
    argparse.Namespace
        The parsed arguments with values specified by the user or defaults.
    """
    parser = argparse.ArgumentParser(
        description="Process SWC files and analyze neuron data."
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Directory where the output files will be saved."
    )
    parser.add_argument(
        "--swc-dir",
        type=str,
        help="Directory where SWC files are located."
    )
    parser.add_argument(
        "--voxel-size", type=str,
        help="Voxel sizes for the X, Y, and Z dimensions."
    )
    parser.add_argument(
        "--image-path", type=str,
        help="Path to the Zarr image file."
    )
    parser.add_argument(
        "--aws-region", type=str, default="us-west-2",
        help="AWS region for S3 storage."
    )
    parser.add_argument(
        "--cache-size", type=int, default=1024 ** 3,
        help="Size of the Zarr cache in bytes."
    )
    parser.add_argument(
        "--num-threads", type=int, default=multiprocessing.cpu_count(),
        help="Number of threads to use for processing."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    swc_dir = args.swc_dir
    voxel_size = tuple(map(float, args.voxel_size.split(',')))
    impath = args.image_path
    num_threads = args.num_threads

    os.makedirs(output_dir, exist_ok=True)

    swc_files = [os.path.join(swc_dir, f) for f in sorted(os.listdir(swc_dir))
                 if f.endswith(".swc")]
    groups = group_swcs(swc_files)

    z = open_zarr_with_cache(impath, args.cache_size, args.aws_region)
    arr = z["0"]

    for g in groups.values():
        process_group(arr, g, output_dir, voxel_size, num_threads)


if __name__ == "__main__":
    scyjava.start_jvm()
    main()
