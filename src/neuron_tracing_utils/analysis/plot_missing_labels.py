import argparse
import os
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_hist(
    data: np.ndarray,
    filepath: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plot a histogram of the data and save the plot to a file.

    Parameters
    ----------
    data : np.ndarray
        An array of numerical data points to plot the histogram.
    filepath : str, optional
        The path to save the histogram plot. If None, the plot is not saved.
    **kwargs : dict
        Additional keyword arguments for plot customization (e.g., 'xlabel',
        'ylabel', 'figsize').

    Returns
    -------
    None
    """
    figsize = kwargs.get('figsize', (8, 6))
    plt.figure(figsize=figsize)

    n, bins, patches = plt.hist(
        data,
        bins=100,
        color='gray',
        alpha=0.6,
        edgecolor='gray',
        label='Frequency'
    )

    # Calculate statistics
    mean = data.mean()
    std = data.std()
    max_value = data.max()

    # Highlight the bins within one standard deviation
    for i in range(len(bins) - 1):
        if mean - std <= bins[i] <= mean + std:
            patches[i].set_facecolor('blue')
            patches[i].set_alpha(0.3)

    # Label for mean
    plt.axvline(
        mean,
        color='red',
        linestyle='dashed',
        linewidth=2,
        label='Mean'
    )

    # Create a patch for the legend to represent the shaded area
    std_patch = mpatches.Patch(
        color='blue',
        alpha=0.3,
        label='Mean ± 1 Std. Dev.'
    )

    # Create a patch for the maximum value for the legend
    max_value_patch = mpatches.Patch(
        color='white',
        label=f'Max Value: {max_value:.2f}'
    )

    if "xlabel" in kwargs:
        plt.xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs:
        plt.ylabel(kwargs["ylabel"])

    # Add the custom patches to the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([std_patch, max_value_patch])
    plt.legend(handles=handles)

    # Save the plot
    if filepath:
        plt.savefig(filepath, dpi=300)

    plt.show()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        The namespace containing the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv-path',
        type=str,
        required=True,
        help='Path to the CSV file.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save output plots.'
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to execute the script logic.

    Returns
    -------
    None
    """
    args = parse_args()
    csv_path = args.csv_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(csv_path)
    print(data)

    plot_hist(
        data['diagonal'].values,
        filepath=os.path.join(output_dir, 'diagonal_hist.png'),
        xlabel='Diagonal Length (px)',
        ylabel='Frequency',
    )
    plot_hist(
        data['num_voxels'].values,
        filepath=os.path.join(output_dir, 'num_voxels_hist.png'),
        xlabel='Number of Voxels',
        ylabel='Frequency',
    )

    # Joint plot
    plt.scatter(
        data['diagonal'].values,
        data['num_voxels'].values,
        alpha=0.5,
        s=0.5
    )
    plt.xlabel('Diagonal Length (px)')
    plt.ylabel('Number of Voxels')
    plt.savefig(os.path.join(output_dir, 'scatter.png'), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
