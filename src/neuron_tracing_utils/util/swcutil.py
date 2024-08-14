import os

import numpy as np


def collect_abspaths(base_dir):
    for f in os.listdir(base_dir):
        yield os.path.join(base_dir, f)


def collect_swcs(base_dir):
    swcs = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".swc") and os.path.isfile(os.path.join(root, f)):
                swcs.append(os.path.join(root, f))
    return swcs


def swc_to_ndarray(swc_path, add_offset=True):
    swc_lines = []
    offset = None
    with open(os.path.abspath(swc_path), "r") as f:
        for line in f.readlines():
            split = line.split()
            if line[0] == "#":
                if len(split) == 1:
                    continue
                if add_offset and split[1].strip() == "OFFSET":
                    offset = np.array([float(oi.strip()) for oi in split[2:5]])
                continue
            row = [float(c.strip()) for c in line.split()]
            swc_lines.append(row)
    swc_arr = np.array(swc_lines, dtype=float)
    if add_offset and offset is not None:
        swc_arr[:, 2:5] += offset

    return swc_arr


def ndarray_to_swc(swc_arr, out_path, color=(1.0, 1.0, 1.0)):
    header = f"# COLOR {color[0]},{color[1]},{color[2]}"
    lines = []
    for row in swc_arr:
        columns = [
            int(row[0]),
            int(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            int(row[6]),
        ]
        line = " ".join(str(p) for p in columns)
        lines.append(line)

    with open(os.path.abspath(out_path), "w") as f:
        f.write(header+"\n")
        f.write("\n".join(lines))


def path_to_name(swc_path):
    return os.path.splitext(os.path.basename(swc_path))[0]
