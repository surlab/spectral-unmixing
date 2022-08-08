# +
import os
from skimage import io
from matplotlib import pyplot as plt

from src import config as cfg


def imread(fullpath):
    return io.imread(fullpath)


def readfile(path):
    print(f"Reading file at {path}")
    with open(path, "r") as f:
        lines = f.read()
        print(lines)


def save_PMT_curve(detected_photons, true_photons, i="", j="", fp=""):
    filesuffix = f"_{fp}_{i}{j}"
    filename = f"photon_counts{filesuffix}"
    # Trying to use the IBL standard here, with some additional info to seperate duplicates instead of new directories
    subdir = "results"
    if not (os.path.isdir(subdir)):
        os.makedirs(subdir)
    path = os.path.join(subdir, filename)
    np.save(detected_photons, path)

    filename = f"photon_corrections{filesuffix}"
    path = os.path.join(subdir, filename)
    np.save(true_photons, path)


def savefig(fig, filename):
    subdir = os.path.join(*cfg.figure_path)
    if not (os.path.isdir(subdir)):
        os.makedirs(subdir)
    pathname = os.path.join(subdir, filename)
    fig.savefig(pathname)
