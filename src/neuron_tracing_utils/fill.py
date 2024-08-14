import argparse
import ast
import json
import logging
import os
import shutil
from enum import Enum

import numpy as np
import scyjava
from ome_zarr.writer import write_multiscales_metadata
from tqdm import tqdm
import zarr
from numcodecs import blosc
from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mode

from neuron_tracing_utils.util.miscutil import range_with_end
from neuron_tracing_utils.transform import WorldToVoxel
from neuron_tracing_utils.util import imgutil
from neuron_tracing_utils.util.ioutil import (
    ImgReaderFactory,
    get_ome_zarr_metadata,
)
from neuron_tracing_utils.util.java import imagej1
from neuron_tracing_utils.util.java import snt
from neuron_tracing_utils.util.swcutil import collect_swcs

DEFAULT_Z_FUDGE = 0.8


class Cost(Enum):
    """Enum for cost functions a user can select"""

    reciprocal = "reciprocal"
    one_minus_erf = "one-minus-erf"


def fill_paths(path_list, img, cost, threshold, calibration):
    reporting_interval = 1000  # ms
    thread = snt.FillerThread(
        img, calibration, threshold, reporting_interval, cost
    )
    thread.setSourcePaths(path_list)
    thread.setStopAtThreshold(True)
    thread.setStoreExtraNodes(False)
    thread.run()
    return thread


def fill_path(path, img, cost, threshold, calibration):
    thread = snt.FillerThread(img, calibration, threshold, -1, cost)
    thread.setSourcePaths([path])
    thread.setStopAtThreshold(True)
    thread.setStoreExtraNodes(False)
    thread.run()
    return thread


def fill_swc_dir_zarr(
    swc_dir,
    im_path,
    out_fill_dir,
    threshold,
    cal,
    key=None,
    cost_min=None,
    cost_max=None,
    voxel_size=(1.0, 1.0, 1.0),
    n_levels=1,
):
    img = imgutil.get_hyperslice(
        ImgReaderFactory.create(im_path).load(im_path, key=key), ndim=3
    )

    swcs = collect_swcs(swc_dir)

    dtype = _get_label_dtype(len(swcs))

    label_zarr, gscore_zarr = _create_zarr_datasets(
        out_fill_dir,
        [1, 1] + list(img.dimensionsAsLongArray()),
        voxel_size=voxel_size,
        label_dtype=dtype
    )
    label_ds = label_zarr["0"]
    gscore_ds = gscore_zarr["0"]

    cost = snt.Reciprocal(cost_min, cost_max)

    label = 1
    for f in swcs:
        tree = snt.Tree(f)
        segments = _chunk_tree(tree)
        for seg in tqdm(segments):
            filler = fill_path(seg, img, cost, threshold, cal)
            _update_fill_stores(
                filler.getFill(), label_ds, gscore_ds, label
            )
        label += 1

    _downscale_labels(label_zarr, n_levels=n_levels, voxel_size=voxel_size)


def _get_label_dtype(num_files):
    if num_files <= 2 ** 8:
        return np.uint8
    elif num_files <= 2 ** 16:
        return np.uint16
    elif num_files <= 2 ** 32:
        return np.uint32
    else:
        return np.uint64


def _downscale_labels(
    label_ds,
    n_levels=1,
    voxel_size=(1.0, 1.0, 1.0),
    compressor=blosc.Blosc(cname="zstd", clevel=1),
):
    label_arr = label_ds["0"][:]
    pyramid = multiscale(
        label_arr,
        windowed_mode,
        (1, 1, 2, 2, 2),
        preserve_dtype=True
    )[:n_levels]
    pyramid = [l.data for l in pyramid]
    for i in range(1, len(pyramid)):
        label_ds.create_dataset(
            str(i),
            data=pyramid[i],
            chunks=label_ds["0"].chunks,
            dtype=label_arr.dtype,
            compressor=compressor,
            write_empty_chunks=False,
            fill_value=0,
        )
    datasets, axes = get_ome_zarr_metadata(voxel_size, n_levels=n_levels)
    write_multiscales_metadata(label_ds, datasets=datasets, axes=axes)


def _create_zarr_datasets(
    out_fill_dir,
    shape,
    chunks=(1, 1, 64, 64, 64),
    compressor=blosc.Blosc(cname="zstd", clevel=1),
    voxel_size=(1.0, 1.0, 1.0),
    label_dtype=np.uint16,
):
    datasets, axes = get_ome_zarr_metadata(voxel_size)

    label_zarr = zarr.open(
        zarr.DirectoryStore(
            os.path.join(out_fill_dir, "Fill_Label_Mask.zarr"),
            "w",
            dimension_separator="/",
        )
    )
    label_zarr.create_dataset(
        "0",
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        compressor=compressor,
        write_empty_chunks=False,
        fill_value=0,
    )

    g_scores_zarr = zarr.open(
        zarr.DirectoryStore(
            os.path.join(out_fill_dir, "G_Scores.zarr"),
            "w",
            dimension_separator="/",
        )
    )
    g_scores_zarr.create_dataset(
        "0",
        shape=shape,
        chunks=chunks,
        dtype=np.float64,
        compressor=compressor,
        write_empty_chunks=False,
        fill_value=np.nan,
    )
    write_multiscales_metadata(g_scores_zarr, datasets=datasets, axes=axes)

    return label_zarr, g_scores_zarr


def _update_fill_stores(fill, label_ds, gscore_ds, label):
    nodelist = fill.getNodeList()
    nodes = []
    scores = []
    for n in nodelist:
        nodes.append([0, 0, n.z, n.y, n.x])
        scores.append(n.distance)
    nodes = np.array(nodes, dtype=int)
    scores = np.array(scores, dtype=float)

    node_idx = tuple(nodes.T)
    old_scores = gscore_ds.vindex[node_idx]

    new_idx = np.nonzero(np.isnan(old_scores))
    if new_idx[0].size > 0:
        new_nodes = tuple(nodes[new_idx].T)
        label_ds.vindex[new_nodes] = label
        gscore_ds.vindex[new_nodes] = scores[new_idx]

    better_idx = np.nonzero(scores < old_scores)
    if better_idx[0].size > 0:
        better_nodes = tuple(nodes[better_idx].T)
        label_ds.vindex[better_nodes] = label
        gscore_ds.vindex[better_nodes] = scores[better_idx]


def _chunk_tree(tree, seg_len=1000):
    paths = []
    for b in snt.TreeAnalyzer(tree).getBranches():
        idx = list(range_with_end(0, b.size() - 1, seg_len))
        for i in range(len(idx) - 1):
            paths.append(b.getSection(idx[i], idx[i + 1]))
    return paths


def _get_cost(im, cost_str, z_fudge=DEFAULT_Z_FUDGE):
    Reciprocal = snt.Reciprocal
    OneMinusErf = snt.OneMinusErf

    params = {}

    if cost_str == Cost.reciprocal.value:
        mean = float(np.mean(im))
        maximum = float(np.max(im))
        cost = Reciprocal(mean, maximum)
        params["fill_cost_function"] = {
            "name": Cost.reciprocal.value,
            "args": {"min": mean, "max": maximum},
        }
    elif cost_str == Cost.one_minus_erf.value:
        mean = np.mean(im)
        maximum = np.max(im)
        std = np.std(im)
        cost = OneMinusErf(maximum, mean, std)
        # reduce z-score by a factor,
        # so we can numerically distinguish more
        # very bright voxels
        cost.setZFudge(z_fudge)
        params["fill_cost_function"] = {
            "name": Cost.one_minus_erf.value,
            "args": {
                "max": maximum,
                "average": mean,
                "standardDeviation": std,
                "zFudge": z_fudge,
            },
        }
    else:
        raise ValueError(f"Invalid cost {cost_str}")

    return cost, params


def _path_values(img, path):
    ProfileProcessor = snt.ProfileProcessor

    shape = ProfileProcessor.Shape.CENTERLINE
    processor = ProfileProcessor(img, path)
    processor.setShape(shape)
    vals = np.array(processor.call(), dtype=float)
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--swcs",
        type=str,
        help="directory of .swc files to fill",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="directory to output mask volumes",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="directory of images associated with the .swc files",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="path to the n5/zarr dataset"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="distance threshold for fill algorithm",
    )
    parser.add_argument(
        "--transform",
        type=str,
        help='path to the "transform.txt" file',
        default=None,
    )
    parser.add_argument(
        "--voxel-size",
        type=str,
        help="voxel size of images",
        default=None,
    )
    parser.add_argument(
        "--cost",
        type=str,
        choices=[cost.value for cost in Cost],
        default=Cost.reciprocal.value,
        help="cost function for the Dijkstra search",
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)
    parser.add_argument(
        "--cost-min",
        type=float,
        default=None,
        help="the value at which the cost function is maximized, "
        "expressed in number of standard deviations from the mean intensity.",
    )
    parser.add_argument(
        "--cost-max",
        type=float,
        default=None,
        help="the value at which the cost function is minimized, expressed in"
        "number of standard deviations from the mean intensity.",
    )
    parser.add_argument(
        "--label-scales",
        type=int,
        default=1,
        help="number of levels in the multiscale zarr pyramid for the label mask",
    )

    args = parser.parse_args()
    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "args.json"), "w") as f:
        args.__dict__["script"] = parser.prog
        json.dump(args.__dict__, f, indent=2)

    logging.basicConfig(format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    os.makedirs(args.output, exist_ok=True)

    calibration = imagej1.Calibration()
    if args.transform is not None:
        voxel_size = WorldToVoxel(args.transform).scale
    elif args.voxel_size is not None:
        voxel_size = ast.literal_eval(args.voxel_size)
    else:
        raise ValueError(
            "Either --transform or --voxel-size must be specified."
        )
    logging.info(f"Using voxel size: {voxel_size}")
    calibration.pixelWidth = voxel_size[0]
    calibration.pixelHeight = voxel_size[1]
    calibration.pixelDepth = voxel_size[2]

    fill_swc_dir_zarr(
        swc_dir=args.swcs,
        im_path=args.image,
        out_fill_dir=args.output,
        threshold=args.threshold,
        cal=calibration,
        key=args.dataset,
        cost_min=args.cost_min,
        cost_max=args.cost_max,
        voxel_size=voxel_size,
        n_levels=args.label_scales,
    )


if __name__ == "__main__":
    main()
