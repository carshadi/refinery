import scyjava


def _java_setup():
    """
    Lazy initialization function for Java-dependent data structures.
    Do not call this directly; use scyjava.start_jvm() instead.
    """
    global N5FSReader
    N5FSReader = scyjava.jimport("org.janelia.saalfeldlab.n5.N5FSReader")
    global N5FSWriter
    N5FSWriter = scyjava.jimport("org.janelia.saalfeldlab.n5.N5FSWriter")
    global N5Utils
    N5Utils = scyjava.jimport("org.janelia.saalfeldlab.n5.imglib2.N5Utils")
    global N5AmazonS3Reader
    N5AmazonS3Reader = scyjava.jimport("org.janelia.saalfeldlab.n5.s3.N5AmazonS3Reader")
    global GzipCompression
    GzipCompression = scyjava.jimport(
        "org.janelia.saalfeldlab.n5.GzipCompression"
    )


scyjava.when_jvm_starts(_java_setup)
