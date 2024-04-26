import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Tuple, List

import numpy as np

from neuron_tracing_utils.util.ioutil import open_ts


def get_component(
    arr: np.ndarray, seed: Tuple[int, int, int]
) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    Get the component coordinates of a label in a 3D segmentation mask,
    by iteratively expanding from a seed point and checking for label matches.

    Parameters
    ----------
    arr : np.array
        Segmentation mask.
    seed : tuple
        Seed point to start search from.

    Returns
    -------
    component : list
        List of coordinates of the component.
    label : int
        Label value of the component.
    """
    component = []
    visited = set()
    stack = [seed]
    label = arr[seed].read().result()
    if label == 0:
        return [], 0
    while stack:
        point = stack.pop()
        if point in visited:
            continue
        visited.add(point)
        val = arr[point].read().result()
        if val == label:
            component.append(point)
            stack.extend(get_neighbors(point, arr.shape))
    return component, label


def get_neighbors(
    point: Tuple[int, int, int], shape: Tuple[int, int, int]
) -> List[Tuple[int, int, int]]:
    """
    Get the 6 neighbors of a point in a 3D array.

    Parameters
    ----------
    point : tuple
        Coordinates of the point (z, y, x).
    shape : tuple
        Shape of the array (depth, height, width).

    Returns
    -------
    neighbors : list
        List of coordinates of the neighbors.
    """
    neighbors = []
    z, y, x = point
    max_z, max_y, max_x = shape

    # Check each axis, and add neighbors if within bounds
    if z > 0:
        neighbors.append((z - 1, y, x))
    if z < max_z - 1:
        neighbors.append((z + 1, y, x))
    if y > 0:
        neighbors.append((z, y - 1, x))
    if y < max_y - 1:
        neighbors.append((z, y + 1, x))
    if x > 0:
        neighbors.append((z, y, x - 1))
    if x < max_x - 1:
        neighbors.append((z, y, x + 1))

    return neighbors


def get_seed_component(
    seed: Tuple[int, int, int], ts: np.ndarray
) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    Get the component and its label for a given seed point in the
    segmentation mask.

    Parameters
    ----------
    seed : Tuple[int, int, int]
        Seed point coordinates (z, y, x).
    ts : np.ndarray
        Segmentation mask array.

    Returns
    -------
    Tuple[List[Tuple[int, int, int]], int]
        Component coordinates and label value.
    """
    c, l = get_component(ts, tuple(seed))
    return c, l


def get_comp_bounding_box(
    comp: List[Tuple[int, int, int]]
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Get the bounding box coordinates of a component.

    Parameters
    ----------
    comp : List[Tuple[int, int, int]]
        List of component coordinates.

    Returns
    -------
    Tuple[Tuple[int, int, int], Tuple[int, int, int]]
        Minimum and maximum coordinates of the bounding box.
    """
    zs, ys, xs = zip(*comp)
    min_z, max_z = min(zs), max(zs)
    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min(xs), max(xs)
    return (min_z, min_y, min_x), (max_z, max_y, max_x)


def save_as_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    filename: str
) -> None:
    """
    Save a mesh as an OBJ file.

    Parameters
    ----------
    vertices : np.ndarray
        Array of vertex coordinates.
    faces : np.ndarray
        Array of face vertex indices.
    normals : np.ndarray
        Array of vertex normals.
    filename : str
        Path to the output OBJ file.
    """
    with open(filename, 'w') as file:
        # Write vertices
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write vertex normals
        for normal in normals:
            file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

        # Write faces with normals
        for face, normal_indices in zip(
                faces,
                faces
        ):  # Assumes normals align with vertices
            file.write(
                f"f {face[0] + 1}//{normal_indices[0] + 1} {face[1] + 1}//"
                f"{normal_indices[1] + 1} {face[2] + 1}//"
                f"{normal_indices[2] + 1}\n"
            )


def process_seed(
    seed: Tuple[int, int, int],
    ts: np.ndarray,
    mesh_dir: str
) -> float:
    """
    Process a seed point in the segmentation mask, compute the component,
    and save the component data.

    Parameters
    ----------
    seed : Tuple[int, int, int]
        Seed point coordinates (z, y, x).
    ts : np.ndarray
        Segmentation mask array.
    mesh_dir : str
        Directory to save the component data.

    Returns
    -------
    float
        Length of the diagonal of the bounding box around the component.
    """
    comp, label = get_seed_component(seed, ts)

    bbmin, bbmax = get_comp_bounding_box(comp)

    # get the length of the diagonal of the bounding box
    diag = np.linalg.norm(np.array(bbmax) - np.array(bbmin))

    origin = np.array(bbmin).astype(int)

    data = ts[bbmin[0]:bbmax[0], bbmin[1]:bbmax[1],
           bbmin[2]:bbmax[2]].read().result()

    data = data.astype(np.uint32)

    np.save(f"{mesh_dir}/{origin[0]}_{origin[1]}_{origin[2]}.npy", data)

    return diag


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Get missing labels")
    parser.add_argument(
        "--label-mask",
        type=str,
        default="gs://allen-nd-goog/from_google/whole_brain/665081"
                "/202402_73227862_855_mean200.0/label_mask",
    )
    parser.add_argument(
        "--seed-list",
        type=str,
        default="/scratch/missing_points.txt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=16 * 1024 ** 3,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    cache_size = args.cache_size

    label_path = args.label_mask
    ts = open_ts(label_path, total_bytes_limit=cache_size)[0]
    print(ts)

    with open(args.seed_list, "r") as f:
        seeds = []
        for line in f:
            seeds.append(tuple(map(int, line.strip().split())))
    print(f"Found {len(seeds)} seeds")

    mesh_dir = os.path.join(args.output, "data")
    os.makedirs(mesh_dir, exist_ok=True)

    print("Getting components...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        diags = np.array(
            list(
                executor.map(
                    partial(process_seed, ts=ts, mesh_dir=mesh_dir),
                    seeds
                )
            )
        )
    print("Done. ", time.time() - t0)

    diags = np.array(diags)
    print(f"Mean: {np.mean(diags)}")
    print(f"Min: {np.min(diags)}")
    print(f"Max: {np.max(diags)}")
    print(f"Std: {np.std(diags)}")

    diag_path = os.path.join(args.output, "diags.npy")
    print("Saving diags to ", diag_path)
    np.save(diag_path, diags)


if __name__ == "__main__":
    main()
