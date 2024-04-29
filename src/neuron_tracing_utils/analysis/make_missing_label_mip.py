import argparse
import os
import tempfile
from pathlib import Path
from typing import Tuple

import nrrd
import numpy as np
import tqdm
import zarr


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace:
        Namespace containing command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process Zarr images and compute MIPs."
    )
    parser.add_argument(
        '--label-dir',
        type=str,
        help='Directory containing label files (.npy).'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Directory to save the output MIP image.'
    )
    parser.add_argument(
        '--image-path',
        type=str,
        help='Path to the Zarr image dataset.'
    )
    return parser.parse_args()


def _update_mip(
    arr: np.ndarray,
    origin: list,
    output_array: np.ndarray
) -> None:
    """Compute the maximum intensity projection and update the output array.

    Parameters
    ----------
    arr (np.ndarray):
        The input array from which to compute the MIP.
    origin (list):
        The starting index where the MIP should be placed.
    output_array (np.ndarray):
        The array where the MIP is updated.
    """
    arr_mip = arr.max(axis=0)

    output_slice = output_array[
                   origin[0]:origin[0] + arr_mip.shape[0],
                   origin[1]:origin[1] + arr_mip.shape[1]
                   ]

    output_array[
    origin[0]:origin[0] + arr_mip.shape[0],
    origin[1]:origin[1] + arr_mip.shape[1]
    ] = np.maximum(output_slice, arr_mip)


def _memmap(
    shape: Tuple[int, int],
    dtype: np.dtype = np.float32,
    mode: str = 'w+'
) -> Tuple[np.memmap, str]:
    """
    Create a memory-mapped array in a temporary file.

    Parameters
    ----------
    shape (Tuple[int, int]):
        The shape of the array to create.
    dtype (np.dtype, optional):
        Data-type, defaults to np.float32.
    mode (str, optional):
        The mode in which the file is opened, defaults
        to 'w+'.

    Returns
    -------
    Tuple[np.memmap, str]:
        Tuple containing the memory-mapped array and
        the path to the temporary file.
    """
    temp_fd, temp_filename = tempfile.mkstemp(suffix='.dat')
    os.close(temp_fd)

    mmap = np.memmap(temp_filename, dtype=dtype, mode=mode, shape=shape)
    mmap[:] = np.zeros(shape, dtype=dtype)
    return mmap, temp_filename


def main() -> None:
    args = parse_args()

    label_dir = Path(args.label_dir)
    output = Path(args.output)
    image_path = args.image_path

    z = zarr.open(image_path, mode='r')['0']
    mip_shape = z.shape[3:]
    # mip, filename = _memmap(mip_shape, dtype=np.float32)
    mip = np.zeros(mip_shape, dtype=np.uint32)

    files = list(label_dir.glob("*.npy"))
    print(f"Processing {len(files)} files...")
    for file in tqdm.tqdm(files):
        try:
            arr = np.load(file)
            if not arr.size:
                continue

            origin = [int(x) for x in file.stem.split('_')[2:]]
            _update_mip(arr, origin, mip)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    try:
        print(f"Saving MIP to {output}")
        nrrd.write(str(output), mip, compression_level=1)
    except Exception as e:
        print(f"Error saving MIP: {e}")
    finally:
        del mip
        # os.unlink(filename)


if __name__ == "__main__":
    main()
