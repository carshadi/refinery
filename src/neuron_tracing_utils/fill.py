import argparse
import ast
import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import numpy as np
import dask
import dask.array as da
from dask_image.ndfilters import median_filter
from distributed import Client, LocalCluster, Lock
import scyjava
from ome_zarr.writer import write_multiscales_metadata
from tqdm import tqdm
import zarr
from numcodecs import blosc
from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mode

from neuron_tracing_utils.util.miscutil import range_with_end
from neuron_tracing_utils.util.ioutil import (
    get_ome_zarr_metadata,
    open_n5_zarr_as_ndarray,
)
from neuron_tracing_utils.util.java import snt
from neuron_tracing_utils.util.java import imagej1
from neuron_tracing_utils.util.swcutil import collect_swcs

from imglyb import to_imglib


class Cost(Enum):
    """Enum for cost functions a user can select"""
    reciprocal = "reciprocal"
    one_minus_erf = "one-minus-erf"
    gaussian_mixture = "gaussian-mixture"


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
        cost_str,
        threshold,
        cal,
        key=None,
        voxel_size=(1.0, 1.0, 1.0),
        n_levels=1,
        profile_mode='mean',
        profile_radius=1,
        profile_shape='sphere',
        mixture_components=2,
        z_score_penalty=1.0,
        threads=1
):
    print("Loading image")
    arr = da.from_zarr(open_n5_zarr_as_ndarray(im_path)[key]).squeeze()

    print("Running median filter")
    arr = median_filter(arr, footprint=np.ones((3, 3, 3)))

    print("Calculating image stats")
    arr, mean, std = dask.compute(arr, arr.mean(), arr.std())
    print(f"Mean: {mean}, Std: {std}")

    swcs = collect_swcs(swc_dir)

    dtype = _get_label_dtype(len(swcs))

    label_zarr, gscore_zarr = _create_zarr_datasets(
        out_fill_dir,
        (1, 1, *arr.shape),
        voxel_size=voxel_size,
        label_dtype=dtype
    )
    label_ds = label_zarr["0"]
    gscore_ds = gscore_zarr["0"]

    # Convert to Java object
    img = to_imglib(arr)

    t0 = time.time()
    print("Starting fill")
    label = 1
    for f in swcs:
        tree = snt.Tree(f)
        segments = _chunk_tree(tree)
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for seg in segments:
                def _worker(seg=seg):
                    vals = _path_values(
                        img,
                        seg,
                        mode=profile_mode,
                        radius=profile_radius,
                        shape=profile_shape
                    )
                    cost, _ = _get_cost(
                        cost_str,
                        vals,
                        mean,
                        std,
                        mixture_components,
                        z_score_penalty
                    )
                    return fill_path(seg, img, cost, threshold, cal)

                futures.append(executor.submit(_worker))

            fillers = [fut.result() for fut in tqdm(futures)]

        for filler in fillers:
            _update_fill_stores(
                filler.getFill(), label_ds, gscore_ds, label
            )

        label += 1

    t1 = time.time()
    print(f"Time to fill: {t1 - t0}")

    print("Downscaling labels")
    t0 = time.time()
    _downscale_labels(label_zarr, n_levels=n_levels, voxel_size=voxel_size)
    t1 = time.time()
    print(f"Time to downscale labels: {t1 - t0}")


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
        label_zarr,
        n_levels=1,
        voxel_size=(1.0, 1.0, 1.0),
        compressor=blosc.Blosc(cname="zstd", clevel=1),
):
    label_arr = da.from_array(label_zarr["0"], chunks=label_zarr["0"].chunks)
    pyramid = multiscale(
        label_arr,
        windowed_mode,
        (1, 1, 2, 2, 2),
        preserve_dtype=True
    )[:n_levels]
    pyramid = [l.data for l in pyramid]
    for i in range(1, len(pyramid)):
        ds = label_zarr.create_dataset(
            str(i),
            shape=pyramid[i].shape,
            chunks=label_arr.chunksize,
            dtype=label_arr.dtype,
            compressor=compressor,
            write_empty_chunks=False,
            fill_value=0,
            overwrite=True
        )
        # TODO: why is lock necessary here?
        da.store(pyramid[i], ds, compute=True, return_stored=False, lock=Lock())

    datasets, axes = get_ome_zarr_metadata(voxel_size, n_levels=n_levels)
    write_multiscales_metadata(label_zarr, datasets=datasets, axes=axes)


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


def _chunk_tree(tree, seg_len=100):
    paths = []
    for b in snt.TreeAnalyzer(tree).getBranches():
        idx = list(range_with_end(0, b.size() - 1, seg_len))
        for i in range(len(idx) - 1):
            paths.append(b.getSection(idx[i], idx[i + 1]))
    return paths


def _get_cost(cost_str, path_values, im_mean, im_std, mixture_components=2, z_score_penalty=1.0):
    params = {}

    if cost_str == Cost.reciprocal.value:
        cost_min = im_mean
        cost_max = np.median(path_values)
        cost = snt.Reciprocal(cost_min, cost_max)
        params["fill_cost_function"] = {
            "name": Cost.reciprocal.value,
            "args": {"min": cost_min, "max": cost_max},
        }
    elif cost_str == Cost.one_minus_erf.value:
        cost_max = np.percentile(path_values, 80)
        cost = snt.OneMinusErf(cost_max, im_mean, im_std)
        params["fill_cost_function"] = {
            "name": Cost.one_minus_erf.value,
            "args": {
                "max": cost_max,
                "average": im_mean,
                "standardDeviation": im_std,
            },
        }
    elif cost_str == Cost.gaussian_mixture.value:
        cost = snt.GaussianMixtureCost(
            path_values,
            mixture_components,
            im_mean,
            im_std,
            z_score_penalty
        )
        params["fill_cost_function"] = {
            "name": Cost.gaussian_mixture.value,
            "args": {
                "values": path_values.tolist(),
                "components": mixture_components,
                "mean": im_mean,
                "standard_deviation": im_std,
                "z_score_penalty": z_score_penalty,
            },
        }
    else:
        raise ValueError(f"Invalid cost {cost_str}")

    return cost, params


def _path_values(img, path, mode="mean", radius=1, shape="sphere"):
    ProfileProcessor = snt.ProfileProcessor
    if shape == "sphere":
        shape = ProfileProcessor.Shape.HYPERSPHERE
    elif shape == "disk":
        shape = ProfileProcessor.Shape.DISK
    elif shape == "none":
        shape = ProfileProcessor.Shape.NONE
    else:
        raise ValueError(f"Invalid shape {shape}")
    processor = ProfileProcessor(img, path)
    processor.setShape(shape)
    processor.setRadius(radius)
    if mode == "mean":
        processor.setMetric(ProfileProcessor.Metric.MEAN)
        return np.array(processor.call())
    elif mode == "max":
        processor.setMetric(ProfileProcessor.Metric.MAX)
        return np.array(processor.call())
    elif mode == "raw":
        raw_values = processor.getRawValues(1)
        a = []
        for vals in dict(raw_values).values():
            a.extend(vals)
        return np.array(a)
    else:
        raise ValueError(f"Invalid mode {mode}")


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
        help="path to the ome-zarr container",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="path to the zarr array in the container"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="distance threshold for fill algorithm",
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
        "--label-scales",
        type=int,
        default=1,
        help="number of levels in the multiscale zarr pyramid for the label mask",
    )
    parser.add_argument(
        "--profile-mode",
        type=str,
        default='raw',
        choices=['mean', 'max', 'raw'],
        help="mode for profile processing",
    )
    parser.add_argument(
        "--mixture-components",
        type=int,
        default=2
    )
    parser.add_argument(
        "--z-score-penalty",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--profile-radius",
        type=int,
        default=1,
        help="radius for profile processing"
    )
    parser.add_argument(
        "--profile-shape",
        type=str,
        default="disk",
        choices=["sphere", "disk", "none"],
        help="shape for profile processing"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="number of threads to use for filling"
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

    voxel_size = ast.literal_eval(args.voxel_size)
    logging.info(f"Using voxel size: {voxel_size}")

    calibration = imagej1.Calibration()
    calibration.pixelWidth = voxel_size[0]
    calibration.pixelHeight = voxel_size[1]
    calibration.pixelDepth = voxel_size[2]

    client = Client(LocalCluster(processes=False))

    fill_swc_dir_zarr(
        swc_dir=args.swcs,
        im_path=args.image,
        out_fill_dir=args.output,
        cost_str=args.cost,
        threshold=args.threshold,
        cal=calibration,
        key=args.dataset,
        voxel_size=voxel_size,
        n_levels=args.label_scales,
        profile_mode=args.profile_mode,
        mixture_components=args.mixture_components,
        z_score_penalty=args.z_score_penalty,
        profile_radius=args.profile_radius,
        profile_shape=args.profile_shape,
        threads=args.threads
    )


if __name__ == "__main__":
    main()
