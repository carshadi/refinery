import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Any

import scyjava

from neuron_tracing_utils.util import swcutil, sntutil


def parse_file_name(file_path: Path) -> Tuple[str, str, str, str]:
    """
    Parses the file name to extract  neuron ID, sample,
    compartment, and tracer initials from the stem of a given file path.

    Parameters
    ----------
    file_path : Path
        Path object of the file to be parsed.

    Returns
    -------
    Tuple[str, str, str, str]
        A tuple containing neuron ID, sample, compartment, and tracer initials.

    Raises
    ------
    ValueError
        If the filename does not contain enough parts to extract required
        metadata.
    """
    file_name = file_path.stem.split("_")[0]
    parts = file_name.split("-")
    if len(parts) < 4:
        raise ValueError(
            "Filename does not contain enough parts to extract metadata"
        )
    return parts[0], parts[1], parts[2], parts[3]


def group_swcs(file_paths: List[Path]) -> Dict[
    Tuple[str, str, str, str], List[Path]]:
    """
    Groups SWC files by neuron ID, sample, compartment, and tracer initials
    based on their filenames.

    Parameters
    ----------
    file_paths : List[Path]
        List of file paths to be grouped.

    Returns
    -------
    Dict[Tuple[str, str, str, str], List[Path]]
        A dictionary where the keys are tuples of metadata (neuron ID,
        sample, compartment, tracer initials),
        and the values are lists of file paths that belong to each group.
    """
    grouped_files = defaultdict(list)
    for file_path in file_paths:
        group_key = parse_file_name(file_path)
        grouped_files[group_key].append(file_path)
    return grouped_files


def swc_to_graph(swc: Path) -> Any:
    """
    Converts an SWC file to a graph.

    Parameters
    ----------
    swc : Path
        Path to the SWC file.

    Returns
    -------
    snt.DirectedWeightedGraph
        Graph object derived from the SWC data.
    """
    arr = swcutil.swc_to_ndarray(swc)
    return sntutil.ndarray_to_graph(arr)


def get_total_length(files: List[Path]) -> float:
    """
    Calculate the total length of neurons from a list of SWC
    files.

    Parameters
    ----------
    files : List[Path]
        List of paths to SWC files.

    Returns
    -------
    float
        The total length of all SWCs.
    """
    total_length = 0
    for file_path in files:
        g = swc_to_graph(file_path)
        total_length += g.sumEdgeWeights()
    return total_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seg-components-dir",
        type=str,
        help="Path to the top-level directory containing the grouped "
             "predicted skeletons,"
             "where there is a folder for all predicted SWCs associated with "
             "each ground-truth"
             "neuron."
    )
    parser.add_argument(
        "--error-swc-dir",
        type=str,
        help="Path to the directory containing error-partitioned ground-truth "
             "SWC files."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        help="Path to the CSV file to write results to."
    )
    args = parser.parse_args()

    predicted_components_dir = Path(args.seg_components_dir)
    error_swc_dir = Path(args.error_swc_dir)
    csv_path = Path(args.csv_path)

    scyjava.start_jvm()

    swc_files = list(error_swc_dir.glob("*.swc"))
    groups = group_swcs(swc_files)

    omit_swcs = {}
    for key, group in groups.items():
        for swc in group:
            if "omit" in str(swc.stem):
                omit_swcs[key] = swc

    csv_data = [
        ["Neuron", "GT Length (mm)", "Predicted Components Length (mm)",
         "Merged Length (mm)"]
    ]
    for neuron_dir in predicted_components_dir.iterdir():
        if not neuron_dir.is_dir():
            continue
        neuron_key = parse_file_name(neuron_dir)

        total_length = get_total_length(list(neuron_dir.glob("*.swc")))

        omit_length = 0
        if neuron_key in omit_swcs and omit_swcs[neuron_key]:
            omit_g = swc_to_graph(omit_swcs[neuron_key])
            omit_length = omit_g.sumEdgeWeights()

        gt_length = get_total_length(groups[neuron_key]) / 1000  # um to mm
        predicted_length = (total_length + omit_length) / 1000  # um to mm
        merged_length = predicted_length - gt_length

        csv_data.append(
            [
                neuron_dir.name,
                round(gt_length, 2),
                round(predicted_length, 2),
                round(merged_length, 2)
            ]
        )

    with csv_path.open(mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        for row in csv_data:
            csv_writer.writerow(row)

    print(f"Results have been written to {csv_path}")


if __name__ == "__main__":
    main()
