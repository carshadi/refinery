import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import time
import itertools

from refinery.util import sntutil, ioutil, imgutil
from refinery.util.ioutil import ImgReaderFactory
from refinery.util.java import snt
from refinery.util.java import imglib2, imagej1
from refinery.transform import WorldToVoxel

import scyjava
import numpy as np


def astar_swc(
        in_swc,
        out_swc,
        im_path,
        calibration,
        key=None,
        timeout=60,
):
    # declare Java classes we will use
    Euclidean = snt.Euclidean
    Reciprocal = snt.Reciprocal
    BiSearch = snt.BiSearch
    SNT = snt.SNT
    Tree = snt.Tree
    ImgUtils = snt.ImgUtils
    Views = imglib2.Views
    DoubleType = imglib2.DoubleType
    ComputeMinMax = imglib2.ComputeMinMax

    print(f"processing {in_swc}")

    reader = ImgReaderFactory.create(im_path)
    img = imgutil.get_hyperslice(reader.load(im_path, key=key))

    graph = Tree(in_swc).getGraph()

    # just to avoid a concurrent modification exception
    # when iterating over the edge set
    edges = [e for e in graph.edgeSet()]

    spacing = np.array(
        [
            calibration.pixelWidth,
            calibration.pixelHeight,
            calibration.pixelDepth,
        ]
    )

    # A* mode
    # Use heuristic = Dijkstra() instead to default to Dijkstra's algorithm (i.e., h(n) = 0)
    heuristic = Euclidean(calibration)
    for e in edges:
        source = e.getSource()
        target = e.getTarget()

        # these need to be voxel coordinates
        sx = int(round(source.x))
        sy = int(round(source.y))
        sz = int(round(source.z))
        tx = int(round(target.x))
        ty = int(round(target.y))
        tz = int(round(target.z))

        # compute min-max of the subvolume where the start and goal nodes
        # are origin and corner, respectively, plus padding in each dimension
        pad_pixels = 20
        subvolume = ImgUtils.subVolume(img, sx, sy, sz, tx, ty, tz, pad_pixels)
        iterable = Views.iterable(subvolume)
        minmax = ComputeMinMax(iterable, DoubleType(), DoubleType())
        minmax.process()

        # reciprocal of intensity * distance is our cost for moving to a neighboring node
        cost = Reciprocal(
            minmax.getMin().getRealDouble(), minmax.getMax().getRealDouble()
        )

        search = BiSearch(
            img,
            calibration,
            sx,
            sy,
            sz,
            tx,
            ty,
            tz,
            timeout,  # timeout (s)
            -1,  # debug mode reporting interval (ms)
            SNT.SearchImageType.MAP,
            cost,
            heuristic,
        )

        search.run()

        # note the Path result is in world coordinates
        path = search.getResult()
        if path is None:
            print(
                "Search failed for {}: points {} and {}, aborting".format(
                    in_swc, str(source), str(target)
                )
            )
            return

        path_arr = sntutil.path_to_ndarray(path)
        # convert back to voxel coords
        path_arr /= spacing

        assert len(path_arr) > 1

        graph.removeEdge(source, target)
        tmp = graph.addVertex(path_arr[0][0], path_arr[0][1], path_arr[0][2])
        graph.addEdge(source, tmp)
        prev = tmp
        for i in range(1, len(path_arr)):
            tmp = graph.addVertex(
                path_arr[i][0], path_arr[i][1], path_arr[i][2]
            )
            graph.addEdge(prev, tmp)
            prev = tmp
        graph.addEdge(tmp, target)

    tree = graph.getTree()
    # Set a non-zero radius.
    # Some programs (JWS) fail to import .swc files with radii == 0
    tree.setSWCType("axon")
    tree.setRadii(1.0)
    tree.saveAsSWC(out_swc)


def astar_batch(
    in_swc_dir,
    out_swc_dir,
    im_dir,
    calibration,
    key=None,
    threads=1,
):
    im_fmt = ioutil.get_file_format(im_dir)

    in_swcs = []
    out_swcs = []
    im_paths = []

    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [f for f in files if f.endswith(".swc")]
        if not swcs:
            continue

        im_path = os.path.join(im_dir, os.path.basename(root) + im_fmt)
        if not os.path.isfile(im_path):
            raise FileNotFoundError(f"{im_path} does not exist")

        for f in swcs:
            in_swc = os.path.join(root, f)
            in_swcs.append(in_swc)
            logging.info(f"Running A-star on {in_swc}")

            out_swc = os.path.join(
                out_swc_dir, os.path.relpath(in_swc, in_swc_dir)
            )
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)
            out_swcs.append(out_swc)

            im_paths.append(im_path)

    times = len(in_swcs)

    t0 = time.time()
    with ThreadPoolExecutor(threads) as executor:
        executor.map(
            astar_swc,
            in_swcs,
            out_swcs,
            im_paths,
            itertools.repeat(calibration, times),
            itertools.repeat(key, times)
        )
    t1 = time.time()
    logging.info(f"processed {times} swcs in {t1-t0}s")


def astar_swcs(
    in_swc_dir,
    out_swc_dir,
    im_path,
    calibration,
    key=None,
    threads=1,
):
    reader = ImgReaderFactory.create(im_path)
    view = imgutil.get_hyperslice(reader.load(im_path, key=key))

    in_swcs = []
    out_swcs = []

    for root, dirs, files in os.walk(in_swc_dir):
        swcs = [f for f in files if f.endswith(".swc")]
        if not swcs:
            continue
        for f in swcs:
            in_swc = os.path.join(root, f)
            in_swcs.append(in_swc)
            logging.info(f"Running A-star on {in_swc}")

            out_swc = os.path.join(
                out_swc_dir, os.path.relpath(in_swc, in_swc_dir)
            )
            Path(out_swc).parent.mkdir(exist_ok=True, parents=True)
            out_swcs.append(out_swc)

    times = len(in_swcs)

    t0 = time.time()
    with ThreadPoolExecutor(threads) as executor:
        executor.map(
            astar_swc,
            in_swcs,
            out_swcs,
            itertools.repeat(view, times),
            itertools.repeat(calibration, times),
        )
    t1 = time.time()
    logging.info(f"processed {times} swcs in {t1-t0}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, help="directory of .swc files to refine"
    )
    parser.add_argument(
        "--output", type=str, help="directory to output refined .swc files"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="image or directory of images associated with the .swc files",
    )
    parser.add_argument(
        "--transform", type=str, help='path to the "transform.txt" file'
    )
    parser.add_argument(
        "--voxel-size",
        type=str,
        help="voxel size for images, as a string in XYZ order, e.g., '0.3,0.3,1.0'",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="key for the N5/Zarr dataset"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="number of threads to use for processing",
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

    calibration = imagej1.Calibration()
    if args.transform is not None:
        voxel_size = WorldToVoxel(args.transform).scale
    elif args.voxel_size is not None:
        voxel_size = ast.literal_eval(args.voxel_size)
    else:
        raise ValueError(
            "Either --transform or --voxel-size must be specified."
        )
    logging.info(f"Using voxel size {voxel_size}")
    calibration.pixelWidth = voxel_size[0]
    calibration.pixelHeight = voxel_size[1]
    calibration.pixelDepth = voxel_size[2]

    logging.info("Starting A-star...")
    if os.path.isdir(args.image):
        astar_batch(
            args.input,
            args.output,
            args.image,
            calibration,
            args.dataset,
            args.threads,
        )
    elif os.path.isfile(args.image):
        astar_swcs(
            args.input,
            args.output,
            args.image,
            calibration,
            args.dataset,
            args.threads,
        )
    else:
        raise FileNotFoundError(f"{args.image} was not found")
    logging.info("Finished A-star.")


if __name__ == "__main__":
    main()
