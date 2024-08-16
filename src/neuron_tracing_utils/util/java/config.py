import os

_default_snt_version = "4.2.1"
_snt_version = os.getenv("SNT_VERSION", _default_snt_version)

_default_fiji_version = "2.15.1"
_fiji_version = os.getenv("FIJI_VERSION", _default_fiji_version)

_default_bigdataviewer_omezarr_version = "0.2.4"
_bigdataviewer_omezarr_version = os.getenv("BIGDATVIEWER_OMEZARR_VERSION", _default_bigdataviewer_omezarr_version)

_default_java_opts = None
_java_opts = os.getenv("JAVA_OPTS", _default_java_opts)

_default_local_fiji = None
_local_fiji = os.getenv("FIJI_PATH", _default_local_fiji)


def set_snt_version(version):
    global _snt_version
    _snt_version = version


def get_snt_version():
    global _snt_version
    return _snt_version


def set_fiji_version(version):
    global _fiji_version
    _fiji_version = version


def get_fiji_version():
    global _fiji_version
    return _fiji_version


def set_bigdataviewer_omezarr_version(version):
    global _bigdataviewer_omezarr_version
    _bigdataviewer_omezarr_version = version


def get_bigdataviewer_omezarr_version():
    global _bigdataviewer_omezarr_version
    return _bigdataviewer_omezarr_version


def get_java_opts():
    global _java_opts
    return _java_opts


def get_local_fiji_path():
    global _local_fiji
    return _local_fiji


def set_local_fiji_path(path):
    global _local_fiji
    _local_fiji = path
