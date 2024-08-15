import scyjava


def _java_setup():
    """
    Lazy initialization function for Java-dependent data structures.
    Do not call this directly; use scyjava.start_jvm() instead.
    """
    global MultiscaleImage
    MultiscaleImage = scyjava.jimport("bdv.img.omezarr.MultiscaleImage")


scyjava.when_jvm_starts(_java_setup)
