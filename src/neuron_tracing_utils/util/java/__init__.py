import logging
from pathlib import Path

from .config import *

import scyjava.config as sjconf


_logger = logging.getLogger(__name__)

sjconf.add_option("-Djava.awt.headless=true")

_java_opts = get_java_opts()
if _java_opts is not None:
    for opt in _java_opts.split(" "):
        _logger.info(f"Adding Java option: {opt}")
        sjconf.add_option(opt)


_fiji_path = get_local_fiji_path()
if _fiji_path is not None:
    jars = []
    # search jars directory
    jars.extend(sjconf.find_jars(_fiji_path + "/jars"))
    # search plugins directory
    jars.extend(sjconf.find_jars(_fiji_path + "/plugins"))
    # add to classpath
    sjconf.add_classpath(os.pathsep.join(jars))

    _logger.info("Added " + str(len(jars) + 1) + " JARs to the Java classpath.")
    plugins_dir = str(Path(_fiji_path) / "plugins")
    sjconf.add_option("-Dplugins.dir=" + plugins_dir)

    # All is well -- now adjust the CWD to the ImageJ2 app directory.
    # See https://github.com/imagej/pyimagej/issues/150.
    os.chdir(_fiji_path)

    if len(sjconf.endpoints) > 0:
        sjconf.endpoints.append("org.scijava:scijava-config:MANAGED")

    # Add additional ImageJ endpoints specific to PyImageJ.
    sjconf.endpoints.append("io.scif:scifio-labeling:0.3.1")

else:
    _fiji_version = get_fiji_version()
    _FIJI_ENDPOINT = f"sc.fiji:fiji:{_fiji_version}"
    sjconf.endpoints.append(_FIJI_ENDPOINT)

    _snt_version = get_snt_version()
    _SNT_ENDPOINT = f"org.morphonets:SNT:{_snt_version}"
    sjconf.endpoints.append(_SNT_ENDPOINT)

    _bigdataviewer_omezarr_version = get_bigdataviewer_omezarr_version()
    _OMEZARR_ENDPOINT = f"org.bigdataviewer:bigdataviewer-omezarr:{_bigdataviewer_omezarr_version}"
    sjconf.endpoints.append(_OMEZARR_ENDPOINT)

    sjconf.endpoints.append("net.imagej:imagej-legacy:MANAGED")
