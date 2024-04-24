import json
import logging
import os
import sys
from pathlib import Path

import argschema
import dask.array as da
import numpy as np
import scyjava
import tifffile
import zarr
from distributed import Client

from neuron_tracing_utils.util import chunkutil
from neuron_tracing_utils.util.java import snt
from neuron_tracing_utils.util.graphutil import get_components_iterative

SWC_EXTENSION = ".swc"

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)


class InputParameters(argschema.ArgSchema):
    points_file = argschema.fields.Str(required=True)
    image_path = argschema.fields.Str(required=True)
    dataset = argschema.fields.Str(required=True)
    out_dir = argschema.fields.Str(required=True)
    swc_dir = argschema.fields.Str(required=False)
    voxel_spacing = argschema.fields.List(
        argschema.fields.Float,
        cli_as_single_argument=True,
        required=True
    )
    block_shape = argschema.fields.List(
        argschema.fields.Int,
        cli_as_single_argument=True,
        required=True
    )
    scale_swcs = argschema.fields.Boolean(required=False, default=False)
    start_index = argschema.fields.Int(required=False, default=0)


def crop_swc(swc, min, max):
    """
    Crop the given SWC file between the given min and max coordinates.

    Parameters
    ----------
    swc : str
        Path to the SWC file.
    min : list
        Minimum x, y, z coordinates.
    max : list
        Maximum x, y, z coordinates.

    Returns
    -------
    list
        Cropped trees.
    """
    g = snt.Tree(swc).getGraph()
    # Remove vertices outside the bounding box
    to_remove = [
        v for v in g.vertexSet() if
        not (
                min[0] <= v.z < max[0] and
                min[1] <= v.y < max[1] and
                min[2] <= v.x < max[2]
        )
    ]

    g.removeAllVertices(to_remove)

    if g.vertexSet().isEmpty():
        return None

    # Align tracing to block origin (0,0,0)
    trees = []
    for c in get_components_iterative(g):
        t = c.getTree()
        t.translate(-min[2], -min[1], -min[0])
        trees.append(t)

    return trees


def crop_swc_dir(swc_dir, out_dir, min, max, scale=None):
    """
    Crop all SWC files in a directory.

    Parameters
    ----------
    swc_dir : str
        Directory containing SWC files.
    out_dir : str
        Output directory.
    min : list
        Minimum coordinates.
    max : list
        Maximum coordinates.
    """
    swcs = [os.path.join(swc_dir, f) for f in os.listdir(swc_dir) if f.endswith(SWC_EXTENSION)]

    for swc in swcs:
        _LOGGER.info(f"Cropping {swc}")
        trees = crop_swc(swc, min, max)

        if trees is None:
            _LOGGER.info(f"No trees left after cropping {swc}")
            continue

        for ti, t in enumerate(trees):
            if scale is not None:
                # t.scale(x_scale, y_scale, z_scale)
                t.scale(scale[2], scale[1], scale[0])
            t.saveAsSWC(os.path.join(out_dir, Path(swc).name.replace(SWC_EXTENSION, f"-{ti}-cropped.swc")))


def save_block(block_dir, block_idx, block, image_path, block_origin, block_shape):
    """
    Save block data to a file.

    Parameters
    ----------
    out_dir : str
        Output directory.
    block_idx : int
        Index of the block.
    block : ndarray
        Data to be saved.
    image_path : str
        Path to input image.
    block_origin : list
        Origin of the chunk.
    block_shape : list
        Shape of the chunk.
    """
    tifffile.imwrite(os.path.join(block_dir, f"block_{block_idx:03d}.tiff"), block)
    meta = {
        "chunk_origin": list(reversed(block_origin.tolist())),
        "chunk_shape": list(reversed(block_shape.tolist())),
        "image_url": image_path
    }
    with open(os.path.join(block_dir, "metadata.json"), 'w') as f:
        json.dump(meta, f)


def read_points_from_file(filepath):
    """
    Read points from the given file.

    Parameters
    ----------
    filepath : str
        Path to the file containing points.

    Returns
    -------
    list
        List of points.
    """
    if not os.path.exists(filepath):
        _LOGGER.error(f"Points file does not exist: {filepath}")
        sys.exit(1)

    points = []
    with open(filepath, 'r') as f:
        for line in f:
            point = json.loads(line.strip())
            points.append(point)
    return points


def main():
    parser = argschema.ArgSchemaParser(schema_type=InputParameters)
    args = parser.args
    _LOGGER.info(args)

    scyjava.start_jvm()

    client = Client()

    points = read_points_from_file(args['points_file'])

    im_path = args['image_path']
    z = zarr.open(im_path, "r")
    d = da.from_array(z[args['dataset']])

    out_dir = args['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    voxel_spacing = np.array(list(reversed(args['voxel_spacing'])))  # XYZ -> ZYX
    shape = np.array(list(reversed(args['block_shape'])))  # XYZ -> ZYX

    for i, point in enumerate(points):
        index = i + args['start_index']
        point = np.array(list(reversed(point)))  # XYZ -> ZYX
        point_vx = point / voxel_spacing

        origin, corner = chunkutil.chunk_center(point_vx, shape)

        data = d[
               ...,
               origin[0]: corner[0],
               origin[1]: corner[1],
               origin[2]: corner[2],
               ].compute()

        block_dir = os.path.join(out_dir, f"block_{index:03d}")
        os.makedirs(block_dir, exist_ok=True)
        save_block(block_dir, index, data, im_path, origin, shape)

        if "swc_dir" in args:
            cropped_swc_dir = os.path.join(block_dir, "cropped-swcs")
            os.makedirs(cropped_swc_dir, exist_ok=True)

            scale = None
            if args['scale_swcs']:
                _LOGGER.info("Scaling SWCs")
                scale = voxel_spacing

            crop_swc_dir(args["swc_dir"], cropped_swc_dir, origin, corner, scale)


if __name__ == "__main__":
    main()
