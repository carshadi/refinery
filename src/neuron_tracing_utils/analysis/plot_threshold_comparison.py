import argparse
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def group_npy_files(
    file_names: List[str],
) -> Dict[Tuple[str, str, str], List[str]]:
    """
    Group neuron files by neuron ID and tracer initials,
    where the group corresponds to different compartment tracings of the
    same cell.
    For example, the files "N001-653158-dendrite-CA.swc" and
    "N001-653158-axon-CA.swc"
    would be grouped together.

    Parameters
    ----------
    file_names : List[str]
        List of absolute file paths.

    Returns
    -------
    Dict[Tuple[str, str, str], List[str]]
        A dictionary where the key is a tuple (neuron ID, sample, tracer
        initials) and the value is a list of file paths.
    """
    grouped_files = defaultdict(list)
    for file_path in sorted(file_names):
        file_path = str(file_path)
        file_name = os.path.basename(file_path)
        parts = file_name.split("-")
        neuron_id, sample, compartment = (
            parts[0],
            parts[1],
            parts[2],
        )
        group_key = (neuron_id, sample, compartment)
        grouped_files[group_key].append(file_path)
    return grouped_files


def plot_group_violin_plots_on_ax(npy_paths1, npy_paths2, ax):
    """
    Plot box plots for the error intensities of a group of neuron files.

    Parameters
    ----------
    npy_paths1 : List[str]
        List of absolute file paths for condition 1.
    npy_paths2 : List[str]
        List of absolute file paths for condition 2.
    ax : matplotlib.axes.Axes
        The axis to plot the box plots on.
    """
    data = []
    for path1, path2 in zip(npy_paths1, npy_paths2):
        name, typ = Path(path1).stem.split("_")
        if typ == "omit":
            data.append(
                pd.DataFrame(
                    {
                        'Intensity': np.load(path1),
                        'Type': typ,
                        'Seg ver': Path(path1).parent.parent.name
                    }
                )
            )
            data.append(
                pd.DataFrame(
                    {
                        'Intensity': np.load(path2),
                        'Type': typ,
                        'Seg ver': Path(path2).parent.parent.name
                    }
                )
            )

    data = pd.concat(data, ignore_index=True)

    sns.violinplot(
        data=data,
        x='Type',
        y='Intensity',
        hue='Seg ver',
        density_norm='count',
        cut=0,
        inner='quartile',
        dodge=True,
        log_scale=True,
        ax=ax
    )

    ax.set_ylabel("Intensity")
    ax.set_xlabel(None)
    ax.set_title(name)
    ax.legend(loc='upper right')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npy-dir1",
        type=str,
        help="Directory where the NPY files are located."
    )
    parser.add_argument(
        "--npy-dir2",
        type=str,
        help="Directory where the NPY files are located."
    )
    args = parser.parse_args()

    # Set the default font sizes
    # plt.rcParams['font.size'] = 12
    # plt.rcParams['axes.titlesize'] = 14
    # plt.rcParams['axes.labelsize'] = 14
    # plt.rcParams['xtick.labelsize'] = 14
    # plt.rcParams['ytick.labelsize'] = 14

    npy_dir1 = args.npy_dir1
    npy_dir2 = args.npy_dir2

    npy_files1 = list(Path(npy_dir1).rglob("*.npy"))
    grouped_files = group_npy_files(npy_files1)

    npy_files2 = list(Path(npy_dir2).rglob("*.npy"))
    grouped_files2 = group_npy_files(npy_files2)

    N = len(grouped_files)
    # Create a rectangular grid best fitting the number of plots
    cols = math.ceil(math.sqrt(N))
    rows = N // cols + (N % cols > 0)
    print(rows, cols)

    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4 * rows),
        sharey=True
    )

    for i, group_key in enumerate(grouped_files):
        group1 = grouped_files[group_key]
        group2 = grouped_files2[group_key]
        plot_group_violin_plots_on_ax(group1, group2, axs[i // cols, i % cols])
        if i > 0:
            axs[i // cols, i % cols].legend().remove()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
