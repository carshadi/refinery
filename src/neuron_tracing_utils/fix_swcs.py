import json
import os
import logging
import argparse
from enum import Enum
from pathlib import Path

from neuron_tracing_utils.util import ioutil
from neuron_tracing_utils.util.graphutil import get_components_iterative
from neuron_tracing_utils.util.imgutil import get_hyperslice
from neuron_tracing_utils.util.ioutil import ImgReaderFactory
from neuron_tracing_utils.util.java import snt
import zarr
import scyjava
import tifffile
import numpy as np


class OutOfBoundsMode(Enum):
    clip = "clip"
    prune = "prune"


def _get_tiff_shape(tiffpath):
    with tifffile.TiffFile(tiffpath) as tif:
        # Open as zarr-store in case we're dealing with
        # ImageJ hyperstacks.
        # tifffile.TiffFile is unable to parse Z-slices with ImageJ
        # Tiffs larger than 4GB.
        z = zarr.open(tif.aszarr(), "r")
        shape = np.array([s for s in z.shape])
        # rearrange to XYZ
        axes = [2, 1, 0]
        return shape[axes]


def _prune_graph(g, mini, maxi):
    to_remove = []
    for v in g.vertexSet():
        point = np.array([v.getX(), v.getY(), v.getZ()])
        if np.any(point < mini) or np.any(point > maxi):
            to_remove.append(v)
    g.removeAllVertices(to_remove)


def _clip_graph(g, mini, maxi):
    for v in g.vertexSet():
        point = np.array([v.getX(), v.getY(), v.getZ()])
        clipped = np.clip(point, mini, maxi)
        if not np.array_equal(point, clipped):
            print(f"input coords: {point}")
            print(f"adjusted coords: {clipped}")
            v.x = clipped[0]
            v.y = clipped[1]
            v.z = clipped[2]


def _fix_graph(graph, img_shape, mode):
    mini = np.array([0, 0, 0])
    maxi = img_shape - 1
    num_points_before = graph.vertexSet().size()
    if mode == OutOfBoundsMode.prune.value:
        _prune_graph(graph, mini, maxi)
        logging.info(
            f"{num_points_before - graph.vertexSet().size()} points pruned"
        )
    elif mode == OutOfBoundsMode.clip.value:
        _clip_graph(graph, mini, maxi)
    else:
        raise ValueError(f"Invalid mode {mode}")


def fix_swcs(in_swc_dir, out_swc_dir, im_path, mode="clip", key=None):
    img = get_hyperslice(
        ImgReaderFactory().create(im_path).load(im_path, key=key)
    )
    img_shape = np.array(img.dimensionsAsLongArray())
    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [os.path.join(root, f) for f in files if f.endswith(".swc")]
        for swc in swcs:
            print(f"fixing {swc}")
            graph = snt.Tree(swc).getGraph()

            _fix_graph(graph, img_shape, mode)

            if graph.vertexSet().size() <= 1:
                continue

            components = get_components_iterative(graph)
            for i, c in enumerate(components):
                if c.vertexSet().size() <= 1:
                    continue
                suffix = ".swc" if len(components) == 1 else f"-{i:03d}.swc"
                out_swc = os.path.join(
                    out_swc_dir,
                    os.path.relpath(swc, in_swc_dir).replace(".swc", suffix),
                )
                Path(out_swc).parent.mkdir(exist_ok=True, parents=True)
                c.getTree().saveAsSWC(out_swc)


def fix_swcs_batch(in_swc_dir, out_swc_dir, imdir, mode="clip"):
    im_fmt = ioutil.get_file_format(imdir)
    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [os.path.join(root, f) for f in files if f.endswith(".swc")]
        if not swcs:
            continue
        im_path = os.path.join(imdir, os.path.basename(root) + im_fmt)
        img = ImgReaderFactory().create(im_path).load(im_path)
        img_shape = np.array(img.dimensionsAsLongArray())
        for f in swcs:
            swc = os.path.join(root, f)
            logging.info(f"fixing {swc}")
            out_swc = os.path.join(
                out_swc_dir, os.path.relpath(swc, in_swc_dir)
            )
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)

            graph = snt.Tree(swc).getGraph()

            _fix_graph(graph, img_shape, mode)

            if graph.vertexSet().size() <= 1:
                continue

            components = get_components_iterative(graph)
            for i, c in enumerate(components):
                if c.vertexSet().size() <= 1:
                    continue
                c.getTree().saveAsSWC(out_swc.replace(".swc", f"-{i}.swc"))


def main():
    parser = argparse.ArgumentParser(
        description="Prune or clip vertices that lay outside the bounds of the image. "
        "If a vertex of degree > 1 is pruned, this will break connectivity "
        "and result in additional .swc outputs."
    )
    parser.add_argument(
        "--input", type=str, help="directory of .swc files to prune"
    )
    parser.add_argument(
        "--output", type=str, help="directory to output pruned .swc files"
    )
    parser.add_argument(
        "--images",
        type=str,
        help="directory of images associated with the .swc files",
    )
    parser.add_argument(
        "--dataset", type=str, help="key for the N5/Zarr dataset", default="0"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in OutOfBoundsMode],
        default=OutOfBoundsMode.clip.value,
        help="how to handle out-of-bounds points",
    )
    parser.add_argument("--log-level", type=int, default=logging.INFO)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "args.json"), "w") as f:
        args.__dict__["script"] = parser.prog
        json.dump(args.__dict__, f, indent=2)

    logging.basicConfig(format="%(asctime)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    scyjava.start_jvm()

    logging.info("Starting fix...")
    fix_swcs(args.input, args.output, args.images, args.mode, key=args.dataset)
    logging.info("Finished fix.")


if __name__ == "__main__":
    main()
