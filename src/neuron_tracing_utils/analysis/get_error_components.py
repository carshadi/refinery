import argparse
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Set, Dict, Tuple, Any

import numpy as np
import scyjava
from tensorstore import TensorStore

from neuron_tracing_utils.resample import resample_tree
from neuron_tracing_utils.util import swcutil, sntutil, graphutil
from neuron_tracing_utils.util.ioutil import open_ts
from neuron_tracing_utils.util.java import snt


def unique_labels(
    graph: Any,
    tensorstore: TensorStore,
) -> Set[int]:
    """
    Calculates unique labels for the vertices of a graph based on their
    spatial location and a tensor store.

    Parameters
    ----------
    graph : snt.DirectedWeightedGraph
        Graph object representing ground truth SWC file.
    tensorstore : TensorStore
        Tensorstore object used to read label data.
    voxel_size : Tuple[float, float, float]
        Tuple containing voxel sizes in z, y, x dimensions respectively.

    Returns
    -------
    Set[int]
        A set of unique labels.
    """
    points = np.array([[v.z, v.y, v.x] for v in graph.vertexSet()]).astype(int)
    zs, ys, xs = points[:, 0], points[:, 1], points[:, 2]
    labels = tensorstore[zs, ys, xs].read().result()
    return set(labels)


def process_swc_file(
    swc_path: str,
    label_mask: TensorStore,
) -> Set[int]:
    """
    Process a SWC file to find unique labels associated with its components.

    Parameters
    ----------
    swc_path : str
        Path to the SWC file.
    label_mask : TensorStore
        Label mask object for label lookup.
    voxel_size : Tuple[float, float, float]
        Voxel size used for normalization.

    Returns
    -------
    Set[int]
        Set of unique labels for the components in the SWC file.
    """
    arr = swcutil.swc_to_ndarray(swc_path)
    graph = sntutil.ndarray_to_graph(arr)
    return unique_labels(graph, label_mask)


def map_filenames_to_paths(folder: str) -> Dict[str, str]:
    """
    Maps filenames without their extensions to their full paths in the 
    specified directory.

    Parameters
    ----------
    folder : str
        Directory path containing files.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping file names (without extension) to full file paths.
    """
    filename_map = {}
    for filename in os.listdir(folder):
        if filename.endswith('.swc'):
            file_key = filename.split('.', 1)[0]
            full_path = os.path.join(folder, filename)
            filename_map[file_key] = full_path
    return filename_map


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Namespace object containing parsed values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--label-mask-path",
        type=str,
        help="Path to the label mask file."
    )
    parser.add_argument(
        "-g",
        "--gt-swc-dir",
        type=str,
        help="Path to the directory containing the SWC files."
    )
    parser.add_argument(
        "-s",
        "--seg-swc-dir",
        type=str,
        help="Path to the directory containing the segmented SWC files."
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Path to the output directory."
    )
    parser.add_argument(
        "-v",
        "--voxel-size",
        type=str,
        help="Voxel size for the X, Y, and Z dimensions."
    )
    parser.add_argument(
        "-n",
        "--node-spacing",
        type=int,
        help="Node spacing for resampling."
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        help="Number of workers to use.",
        default=multiprocessing.cpu_count()
    )
    return parser.parse_args()


def main():
    """
    Main function to run the process.

    Parse arguments, process SWC files, and handle input/output directories.
    """
    args = parse_args()

    label_mask_path = args.label_mask_path
    label_mask = open_ts(label_mask_path)[0]
    print(label_mask.shape)

    voxel_size = tuple(map(float, args.voxel_size.split(',')))

    gt_swc_dir = args.gt_swc_dir

    swc_labels = defaultdict(set)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(
            process_swc_file,
            os.path.join(gt_swc_dir, f),
            label_mask,
        )
            for f in sorted(os.listdir(gt_swc_dir))]
        names = [f.split('_')[0] for f in sorted(os.listdir(gt_swc_dir))]
        for i, future in enumerate(futures):
            try:
                swc_labels[names[i]].update(future.result())
            except Exception as e:
                print(e)

    outdir = args.outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    d = map_filenames_to_paths(folder=args.seg_swc_dir)
    for name in names:
        labels = swc_labels[name]
        neuron_dir = os.path.join(outdir, name)
        os.makedirs(neuron_dir, exist_ok=True)
        for label in labels:
            try:
                p = d[str(label)]
            except KeyError:
                print(f"Could not find {label}.swc")
                continue
            t = snt.Tree(p)
            t.scale(*voxel_size)
            if args.node_spacing > 0:
                resample_tree(t, args.node_spacing)
            t.setRadii(1.0)
            t.saveAsSWC(neuron_dir + f"/{label}.swc")


if __name__ == "__main__":
    scyjava.start_jvm()
    main()
