import logging
import os.path
from typing import List, Dict, Tuple
from urllib.parse import urlparse

import numpy as np
import tensorstore as ts
import zarr
from tensorstore import TensorStore
from zarr.errors import PathNotFoundError

from neuron_tracing_utils.util.java import imglib2, n5, aws, omezarr

_LOGGER = logging.getLogger(__name__)


class TiffReader:

    def __init__(self):
        self.loader = imglib2.IJLoader()

    def load(self, filepath, **kwargs):
        if "key" in kwargs:
            _LOGGER.warning("Ignoring keyword argument 'key' for TiffLoader")
        return self.loader.get(filepath)


class N5Reader:

    @staticmethod
    def _get_reader(path):
        path = str(path)

        parsed = urlparse(path)
        bucket = parsed.netloc
        prefix = parsed.path

        if path.startswith("s3://"):
            config = aws.ClientConfiguration().withMaxErrorRetry(10).withMaxConnections(100)
            s3 = aws.AmazonS3ClientBuilder.standard().withClientConfiguration(
                config
            ).build()
            reader = n5.N5AmazonS3Reader(s3, bucket, prefix)
            return reader
        elif path.startswith("gs://"):
            # TODO
            raise NotImplementedError("GCS is not currently supported")
        else:
            reader = n5.N5FSReader(path)
            return reader

    def load(self, path, **kwargs):
        key = kwargs.get("key", "volume")
        reader = self._get_reader(path)
        if "cache" in kwargs:
            cache = kwargs["cache"]
            if isinstance(cache, int):
                dataset = n5.N5Utils.openWithBoundedSoftRefCache(
                    reader, key, cache
                )
            elif isinstance(cache, bool) and cache:
                dataset = n5.N5Utils.openWithDiskCache(reader, key)
            else:
                dataset = n5.N5Utils.open(reader, key)
        else:
            dataset = n5.N5Utils.open(reader, key)
        return dataset


class OmeZarrReader:
    def load(self, path, **kwargs):
        MultiscaleImage = omezarr.MultiscaleImage

        key = kwargs.get("key", "0")

        path = str(path)
        if path.startswith("s3://"):
            parsed = urlparse(path)
            bucket = parsed.netloc
            prefix = parsed.path

            config = aws.ClientConfiguration().withMaxErrorRetry(10).withMaxConnections(100)
            s3 = aws.AmazonS3ClientBuilder.standard().withClientConfiguration(
                config
            ).build()

            reader = MultiscaleImage.ZarrKeyValueReaderBuilder(s3, bucket, prefix.strip('/'))
        else:
            reader = MultiscaleImage.ZarrKeyValueReaderBuilder(path)

        multiscale = MultiscaleImage(reader)
        return multiscale.getImg(int(key))


class ImgReaderFactory:
    LOADERS = {
        ".tif": TiffReader,
        ".tiff": TiffReader,
        ".n5": N5Reader,
        ".zarr": OmeZarrReader
    }

    @staticmethod
    def create(path):
        _, ext = os.path.splitext(path)
        return ImgReaderFactory.LOADERS[ext]()


def get_file_format(imdir):
    f = next(iter(os.listdir(imdir)))
    _, ext = os.path.splitext(f)
    return ext


def open_n5_zarr_as_ndarray(path: str):
    try:
        z = zarr.open(path, "r")
    except PathNotFoundError:
        try:
            z = zarr.open(store=zarr.N5FSStore(path), mode="r")
        except PathNotFoundError:
            return None
    return z


def open_ts(
        path: str,
        dataset: str = None,
        total_bytes_limit: int = 200_000_000
) -> TensorStore:
    # TensorStore opens n5 with axis order X,Y,Z, so get
    # a transposed view to be compatible with util code
    if os.path.exists(path) and not path.startswith("file://"):
        path = "file://" + path
    spec = {
        "driver": _get_driver_string(path),
        "kvstore": path,
        "context": {
            "cache_pool": {
                "total_bytes_limit": total_bytes_limit
            }
        },
        # "open": True,
        "recheck_cached_data": "open"
    }
    if spec['driver'] != 'tiff' and dataset is not None:
        spec['path'] = dataset
    ds = ts.open(spec).result()
    if spec['driver'] in ("n5", "neuroglancer_precomputed"):
        return ds.T
    return ds


def _get_driver_string(image_path: str):
    drivers = {
        ".zarr": "zarr",
        ".n5": "n5",
        ".tiff": "tiff",
        ".tif": "tiff",
        "": "neuroglancer_precomputed"
    }
    _, ext = os.path.splitext(image_path)
    return drivers[ext]


def is_n5_zarr(path):
    ret = open_n5_zarr_as_ndarray(path)
    return ret is not None


def get_ome_zarr_metadata(
        voxel_size: List[float],
        n_levels: int = 1
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate the multiscale metadata for a tensorstore.
    Args:
        voxel_size:  The voxel size in micrometers.
        n_levels:  The number of levels in the multiscale pyramid.

    Returns:
        datasets:  The list of datasets in the multiscale pyramid.
        axes:  The list of axes in the multiscale pyramid.
    """
    voxel_size = np.array([1, 1] + list(reversed(voxel_size)))
    scales = [np.concatenate((voxel_size[:2], voxel_size[2:] * 2 ** i)) for i in range(n_levels)]
    coordinate_transformations = [[{"type": "scale", "scale": scale.tolist()}] for scale in scales]
    datasets = [{"path": str(i), "coordinateTransformations": coordinate_transformations[i]} for i in range(n_levels)]
    axes = [{"name": 't', "type": "time", "unit": "millisecond"},
            {"name": 'c', "type": "channel"},
            {"name": 'z', "type": "space", "unit": "micrometer"},
            {"name": 'y', "type": "space", "unit": "micrometer"},
            {"name": 'x', "type": "space", "unit": "micrometer"}]

    return datasets, axes
