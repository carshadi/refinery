import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import scyjava
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from neuron_tracing_utils.util import sntutil, swcutil
from neuron_tracing_utils.util.ioutil import open_ts
from neuron_tracing_utils.util.java import snt
from neuron_tracing_utils.util.graphutil import get_components_iterative


def partition_graph(g, label_mask):
    """
    Partition a graph based on a label mask, categorizing edges into omitted,
    split, or correct graphs.

    Parameters:
    g (snt.DirectedWeightedGraph): The graph to partition.
    label_mask (TensorStore): The label mask used for partitioning.

    Returns:
    tuple: A tuple containing sets of components for omitted, split,
    and correct
           graphs and a list of non-duplicate edges.
    """
    omit_graph = snt.DirectedWeightedGraph()
    split_graph = snt.DirectedWeightedGraph()
    correct_graph = snt.DirectedWeightedGraph()
    non_duplicate_edges = []
    z_coords = []
    y_coords = []
    x_coords = []
    for e in g.edgeSet():
        s = e.getSource()
        t = e.getTarget()
        sx, sy, sz = int(s.getX()), int(s.getY()), int(s.getZ())
        tx, ty, tz = int(t.getX()), int(t.getY()), int(t.getZ())
        if (tx, ty, tz) != (sx, sy, sz):
            non_duplicate_edges.append(e)
            z_coords.extend([sz, tz])
            y_coords.extend([sy, ty])
            x_coords.extend([sx, tx])
    values = label_mask[z_coords, y_coords, x_coords].read().result()
    for i, e in enumerate(non_duplicate_edges):
        s = e.getSource()
        t = e.getTarget()
        v1 = values[2 * i]
        v2 = values[2 * i + 1]
        if v1 == 0 or v2 == 0:
            graph = omit_graph
        elif v1 != v2:
            graph = split_graph
        else:
            graph = correct_graph
        graph.addVertex(s)
        graph.addVertex(t)
        graph.addEdge(s, t, e)
    return (get_components_iterative(omit_graph),
            get_components_iterative(split_graph),
            get_components_iterative(correct_graph),
            non_duplicate_edges)


def _save_swc(
    arr, output_path, voxel_size, node_type=2, color=(1, 1, 1), radius=1.0,
):
    """
    Save an SWC array to a file.

    Parameters:
    arr (np.array): The SWC data array.
    output_path (str): Path to save the SWC file.
    voxel_size (tuple): The voxel size in microns.
    node_type (int): The neuron node type.
    color (tuple): The color of the node.
    radius (float): The radius of the node.
    """
    arr[:, 1] = node_type
    arr[:, 2:5] *= np.array(voxel_size)
    arr[:, 5] = radius
    swcutil.ndarray_to_swc(arr, output_path, color)


def merge_components_as_ndarray(components):
    """
    Merge graph components into a single ndarray.

    Parameters:
    components (list): A list of graph components to merge.

    Returns:
    np.array: The merged ndarray.
    """
    arrays = []
    for c in components:
        arrays.append(sntutil.tree_to_ndarray(c.getTree()))

    if not arrays:
        return None

    current_idx = arrays[0].shape[0] + 1
    remaped_arrs = [arrays[0]]
    for arr in arrays[1:]:
        remaped = remap_swc_array_ids(arr, current_idx)
        remaped_arrs.append(remaped)
        current_idx += remaped.shape[0]

    return np.concatenate(remaped_arrs)


def remap_swc_array_ids(swc_array, start_id=1):
    """
    Remap node IDs in an SWC array so that the lowest ID starts at 'start_id'.

    Parameters:
    swc_array (np.array): NumPy array containing SWC data with shape (m, 7).
    start_id (int): The starting ID for the lowest node.

    Returns:
    np.array: The SWC array with remapped IDs.
    """
    # Ensure the array is sorted by the first column (node ID)
    swc_array = swc_array[swc_array[:, 0].argsort()]

    # Initialize the new ID and create a mapping from old IDs to new IDs
    id_mapping = {row[0]: start_id + i for i, row in enumerate(swc_array)}

    # Update the node IDs and the parent IDs
    for row in swc_array:
        row[0] = id_mapping[row[0]]
        if row[6] != -1:  # Update the parent ID if it's not -1 (root node)
            row[6] = id_mapping[row[6]]

    return swc_array


def process_swc_file(swc_path, label_mask, output_base_dir, name, voxel_size):
    """
    Process an SWC file, partitioning its graph based on a label mask.

    Parameters:
    swc_path (str): Path to the SWC file.
    label_mask (array-like): The label mask used for partitioning.
    output_base_dir (str): Base directory for output files.
    name (str): Base name for the output files.
    voxel_size (tuple): The voxel size in microns.

    Returns:
    dict: A dictionary containing statistics about the processed graph.
    """
    g = snt.Tree(swc_path).getGraph()
    try:
        omit, split, correct, non_duplicate_edges = partition_graph(
            g, label_mask
        )
    except Exception as e:
        print(e)
        return

    total_length = len(non_duplicate_edges)

    omit_length = sum(c.edgeSet().size() for c in omit)
    omit_arr = merge_components_as_ndarray(omit)
    if omit_arr is not None:
        _save_swc(
            omit_arr,
            os.path.join(output_base_dir, f"{name}_omit.swc"),
            voxel_size=voxel_size,
            color=(1, 0, 0)
        )

    split_length = sum(c.edgeSet().size() for c in split)
    split_arr = merge_components_as_ndarray(split)
    if split_arr is not None:
        _save_swc(
            split_arr,
            os.path.join(output_base_dir, f"{name}_split.swc"),
            voxel_size=voxel_size,
            color=(0, 0, 1)
        )

    correct_length = sum(c.edgeSet().size() for c in correct)
    correct_arr = merge_components_as_ndarray(correct)
    if correct_arr is not None:
        _save_swc(
            correct_arr,
            os.path.join(output_base_dir, f"{name}_correct.swc"),
            voxel_size=voxel_size,
            color=(0, 1, 0)
        )

    return {
        "omit_length": omit_length,
        "split_length": split_length,
        "correct_length": correct_length,
        "total_length": total_length,
    }


def process_swc_file_wrapper(swc, args, label_mask, voxel_size):
    """
    Wrapper function to process SWC files in parallel.

    Parameters:
    swc (str): The SWC file name.
    args (argparse.Namespace): Parsed command line arguments.
    label_mask (array-like): The label mask.
    voxel_size (tuple): The voxel size in microns.

    Returns:
    dict: A dictionary containing statistics about the processed graph.
    """
    print(f"Processing {swc}")
    name = swc.replace(".swc", "")
    data = process_swc_file(
        os.path.join(args.swc_dir, swc),
        label_mask,
        args.out_dir,
        name,
        voxel_size
    )
    data["name"] = name
    return data


def save_boxplot(data, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    plt.figure(figsize=(13, 10))

    # Customizing the appearance of the mean marker
    meanprops = {
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "10"
    }

    # Plotting horizontal box plots for each proportion with mean markers
    ax1 = plt.subplot(3, 1, 1)  # Adjusting subplot orientation
    sns.boxplot(
        x=data['omit_proportion'],
        showmeans=True,
        meanprops=meanprops,
        ax=ax1
    )
    ax1.set_title('Omitted edges')
    ax1.set_xlabel('')  # Adjusting to remove the x-axis label

    # Save the figure for the first plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'omit_proportion_boxplot.png'))
    plt.close()  # Close the figure to free up memory

    # Repeat the process for the other plots
    plt.figure(figsize=(13, 10))
    ax2 = plt.subplot(1, 1, 1)
    sns.boxplot(
        x=data['split_proportion'],
        showmeans=True,
        meanprops=meanprops,
        ax=ax2
    )
    ax2.set_title('Split edges')
    ax2.set_xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'split_proportion_boxplot.png'))
    plt.close()

    plt.figure(figsize=(13, 10))
    ax3 = plt.subplot(1, 1, 1)
    sns.boxplot(
        x=data['correct_proportion'],
        showmeans=True,
        meanprops=meanprops,
        ax=ax3
    )
    ax3.set_title('Same Label edges')
    ax3.set_xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correct_proportion_boxplot.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process SWC files and partition the graph based on a "
                    "label mask."
    )

    parser.add_argument(
        "--label-mask",
        type=str,
        help="Path to the TensorStore on Google Cloud Storage.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        help="Output directory path.",
    )

    parser.add_argument(
        "--swc-dir",
        type=str,
        help="Directory containing SWC files.",
    )
    parser.add_argument(
        "--voxel-size",
        type=str,
        help="Voxel size in microns.",
        default="1.0,1.0,1.0",
    )

    args = parser.parse_args()

    voxel_size = tuple(map(float, args.voxel_size.split(",")))

    scyjava.start_jvm()

    label_mask = open_ts(args.label_mask)
    label_mask = label_mask[0]

    os.makedirs(args.out_dir, exist_ok=True)

    with ThreadPoolExecutor(
            max_workers=multiprocessing.cpu_count()
    ) as executor:
        swc_files = os.listdir(args.swc_dir)
        data = list(
            executor.map(
                process_swc_file_wrapper,
                swc_files,
                [args] * len(swc_files),
                [label_mask] * len(swc_files),
                [voxel_size] * len(swc_files)
            )
        )

    df = pd.DataFrame(data)
    df = df.set_index("name")

    df['omit_proportion'] = df['omit_length'] / df['total_length']
    df['split_proportion'] = df['split_length'] / df['total_length']
    df['correct_proportion'] = df['correct_length'] / df['total_length']

    df.sort_values(by="omit_proportion", ascending=False, inplace=True)
    print(df.to_string(index=False))

    df.to_csv(os.path.join(args.out_dir, "results.csv"))

    save_boxplot(df, args.out_dir)


if __name__ == "__main__":
    main()
