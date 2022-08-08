# +
import os
from skimage import io
from datetime import datetime


from src import config as cfg
import numpy as np
import json

from collections import defaultdict
from tifffile import imwrite


def imread(fullpath):
    if '.tiff' in fullpath:
        print('here')
        return io.imread(fullpath)
    else:
        print('there')
        return get_tiff_stack(fullpath, cfg.num_channels)


def readfile(path):
    print(f"Reading file at {path}")
    with open(path, "r") as f:
        lines = f.read()
        print(lines)


def save_PMT_curve(detected_photons, true_photons, i="", j="", fp=""):
    filesuffix = f"_{fp}_{i}{j}"
    filename = f"photon_counts{filesuffix}.npy"
    # Trying to use the IBL standard here, with some additional info to seperate duplicates instead of new directories
    subdir = os.path.join(*cfg.results_path)
    if not (os.path.isdir(subdir)):
        os.makedirs(subdir)
    path = os.path.join(subdir, filename)
    np.save(path, detected_photons)

    filename = f"photon_corrections{filesuffix}.npy"
    path = os.path.join(subdir, filename)
    np.save(path, true_photons)

def save_mean_PMT_curve(detected_photons, true_photons):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    fp = f"mean_{timestamp}"
    save_PMT_curve(detected_photons, true_photons, i='', j='', fp=fp)

def load_PMT_curves():
    subdir = os.path.join(*cfg.results_path)
    curve_dict = defaultdict(dict)
    for filename in os.listdir(subdir):
        if "corrections" in filename:
            path = os.path.join(subdir, filename)
            idx = filename.find("corrections")
            details = filename[idx + 12 :]
            curve_dict[details]["corrections"] = np.load(path)
            counts_filename = filename.replace("corrections", "counts")
            path = os.path.join(subdir, counts_filename)
            curve_dict[details]["counts"] = np.load(path)
    return curve_dict


def savefig(fig, filename):
    subdir = os.path.join(*cfg.figure_path)
    if not (os.path.isdir(subdir)):
        os.makedirs(subdir)
    pathname = os.path.join(subdir, filename)
    fig.savefig(pathname)

def save_valid_curve_json(current_valid_dict: dict={}, all_curves: dict=None):
    if all_curves:
        for key, value in all_curves.items():
            if key not in current_valid_dict:
                current_valid_dict[key]=False


    filepath = get_valid_curves_filepath()
    with open(filepath, 'w') as f:
        json.dump(current_valid_dict, f, indent=4)

def get_valid_curves_filepath():
    subdir = os.path.join(*cfg.results_path)
    filename = cfg.valid_curve_json_filename
    filepath = os.path.join(subdir, filename)
    return filepath


def load_valid_curve_json():
    filepath = get_valid_curves_filepath()
    with open(filepath, 'r') as f:
        valid_curves: dict = json.load(f)
    return valid_curves



def I16_read(file_path, num_channels, length_x_pixels=None, length_y_pixels=None):

    #This function opens an I16 file and split it to channels.
    #Taken from Nedivi lab unmixing

    bytes_per_pixel = 2

    #Open file
    fullfile_name = file_path
    with open (file_path, 'r') as f:

        # In case you need to remove some bytes from the LabView file,
        # change the n_bytes_2_remove value
        n_bytes_2_remove = 0

        # Find the location of the last element (i.e., size of the file) and reduce it by the number of bytes to be removed
        f.seek(0, 2)          # go to the end of the file
        last_pos = f.tell()    # find the last element position
        size_bytes = last_pos - n_bytes_2_remove

        # If the user didn't define the image size, assume square shape
        if length_x_pixels is None:
            length_x_pixels = np.sqrt(size_bytes / (bytes_per_pixel*num_channels))
        if length_y_pixels is None:
            length_y_pixels = length_x_pixels

        # Return to the file begining
        f.seek(0,0)

        # obtain number of channels
        #if isempty(app.num_channels)
        #    app.num_channels = size (img_channels,3)

        # split into channels
        #img_channels = np.zeros([length_x_pixels, length_y_pixels, num_channels])
        #for channel_ind in np.arange(1,app.num_channels+1).reshape(-1):
        #    img_channels[:,:,channel_ind] =
        im_stack = []
        for i in range(num_channels):
            img = np.fromfile(f, np.int16, count=int(length_x_pixels*length_y_pixels))
            img = np.reshape(img, (int(length_x_pixels),int(length_y_pixels))).T
            im_stack.append(img)

        #img_channels = np.reshape(img, (int(length_x_pixels),int(length_y_pixels), int(num_channels)))
        img_channels = np.array(im_stack).T
    return img_channels


def get_tiff_stack(folder_path, num_channels):
    # Run over all I16 files in the folder
    slice_list = []
    for file_name in os.listdir(folder_path):
        if '.I16' in file_name:
            fullpath = os.path.join(folder_path, file_name)
            # Open image slice and reshape it
            img_channels = I16_read(fullpath, num_channels)
            slice_list.append(img_channels)

        # Stack images ( the external "Multipage TIFF stack"
# function is required for this part)
        #for channel_ind in np.arange(1,app.num_channels+1).reshape(-1):
        #    options.append = True
        #    app.saveastiff(uint16(unmixed_img_channels(:,:,channel_ind)),app.tiff_out,options)

    return np.array(slice_list)

def write_4d_tiff(image_stack: np.array , filepath):
    imwrite(filepath, image_stack, imagej=False)

