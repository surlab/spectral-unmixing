# +
import os
from skimage import io
from datetime import datetime


from src import config as cfg
from src import computation as comp
import numpy as np
import json
import logging

from collections import defaultdict
import tifffile as tf


def imread(fullpath, num_channels=None, verbose=False):
    if num_channels is None:
        num_channels = cfg.num_channels
    if '.tif' in fullpath:
        with tf.TiffFile(fullpath) as tif:
            tif_tags = {}
            for tag in tif.pages[0].tags.values():
                name, value = tag.name, tag.value
                tif_tags[name] = value
            image = tif.pages[0].asarray()
        image, tif_tags = io.imread(fullpath), tif_tags
    else:
        image, tif_tags = get_tiff_stack(fullpath, num_channels)
    if verbose:
        print(f"Image open executed without errors. The image is {image.shape}")
    return image, tif_tags


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
    save_array(path, true_photons)
    #np.save(path, true_photons)

def save_mean_PMT_curve(detected_photons, true_photons):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    fp = f"mean_{timestamp}"
    save_PMT_curve(detected_photons, true_photons, i='', j='', fp=fp)

def load_PMT_curves():
    subdir = os.path.join(*cfg.results_path)
    curve_dict = defaultdict(dict)
    for filename in os.listdir(subdir):
        if "corrections" in filename:
            try:
                path = os.path.join(subdir, filename)
                idx = filename.find("corrections")
                details = filename[idx + 12 :]
                curve_dict[details]["corrections"] = load_array(path)
                counts_filename = filename.replace("corrections", "counts")
                path = os.path.join(subdir, counts_filename)
                curve_dict[details]["counts"] = load_array(path)
            except FileNotFoundError as E:
                logging.warning(f'Corresponding photon counts file not found for {filename}')
    return curve_dict


def load_master_PMT_curve():
    subdir = os.path.join(*cfg.results_path)
    counts_path = os.path.join(subdir, f'photon_counts{cfg.master_PMT_curve_corrections_suffix}')
    corrections_path = os.path.join(subdir, f'photon_corrections{cfg.master_PMT_curve_corrections_suffix}')
    counts = load_array(counts_path)
    corrections = load_array(corrections_path)
    return counts, corrections


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
    for file_name in get_ordered_I16_list(folder_path):
        #if '.I16' in file_name:
        fullpath = os.path.join(folder_path, file_name)
        # Open image slice and reshape it
        img_channels = I16_read(fullpath, num_channels)
        slice_list.append(img_channels)

        # Stack images ( the external "Multipage TIFF stack"
# function is required for this part)
        #for channel_ind in np.arange(1,app.num_channels+1).reshape(-1):
        #    options.append = True
        #    app.saveastiff(uint16(unmixed_img_channels(:,:,channel_ind)),app.tiff_out,options)

    return np.array(slice_list), None


def get_ordered_I16_list(path_):
    ordered_file_list = []

    filetype_str = '.I16'
    start_image = 'Z0.I16'
    for filename in os.listdir(path_):
        if start_image in filename:
            idx = filename.find(filetype_str)
            file_prefix = filename[:idx-1]
            i=0
            while True:
                check_filename = f'{file_prefix}{i}{filetype_str}'
                if os.path.exists(os.path.join(path_, check_filename)):
                    i+=1
                    #print(check_filename in ordered_file_list)
                    if check_filename not in ordered_file_list:
                        ordered_file_list.append(check_filename)
                else:
                    break
    for filename in os.listdir(path_):
        if filetype_str in filename:
            i=0
            idx = filename.find(filetype_str)
            file_prefix = filename[:idx-1]
            while True:
                check_filename = f'{file_prefix}{i}{filetype_str}'
                if os.path.exists(os.path.join(path_, check_filename)):
                    i+=1
                    #print(check_filename in ordered_file_list)
                    if check_filename not in ordered_file_list:
                        ordered_file_list.append(check_filename)
                else:
                    break
    return ordered_file_list

def tiffify_filename(filename):
    if not('.tif' in filename[-5:]):
        return filename+'.tiff'
    return filename

def write_composite_4d_tiff(image_stack: np.array , dirpath, filename, verbose=False, **kwargs, ):
    filename, extension = os.path.splitext(filename)
    #filename = f"{filename}{'_composite'}"
    filename = tiffify_filename(filename)
    fullpath = os.path.join(dirpath, filename)
    reorder_axis = np.moveaxis(image_stack, -1, 1)
    if verbose:
        print(f'Saving image to {fullpath}')
    tf.imwrite(fullpath, reorder_axis.astype('uint16'), kwargs, metadata={'axes': 'ZCYX'}, imagej=True) # imagej=False)compression='LZW')#
    if verbose:
        print('Save complete')

def write_color_seperated_4d_tiff(image_stack: np.array , dirpath, filename, **kwargs):
    filename, extension = os.path.splitext(filename)
    for i in range(image_stack.shape[0]):
        filename_ch = f"{filename}_ch{i}"
        filename_ch = tiffify_filename(filename_ch)
        fullpath = os.path.join(dirpath, filename_ch)
        tf.imwrite(fullpath, image_stack[:,:,:,i], kwargs, metadata={'axes': 'ZYX'}, imagej=True)#, compression='LZW')#, metadata={'axes': 'ZCYX'}) # imagej=False




def get_linear_tiff_path(fp):
    subdir = os.path.join(*cfg.save_data_path)
    filename = f'{fp}_linear.tiff'
    filepath = os.path.join(subdir, filename)
    return filepath

def get_unmix_tiff_path(oldname):
    if '.' in oldname:
        oldname = os.path.splitext(oldname)[0]
    subdir = os.path.join(*cfg.save_data_path)
    filename = f'{oldname}_unmixed.tiff'
    filepath = os.path.join(subdir, filename)
    return filepath

def get_coef_path(fp):
    subdir = os.path.join(*cfg.save_data_path)
    filename = f'channel_unmixingCoefs_{fp}.{cfg.channel_set}.{cfg.save_array_as}'
    filepath = os.path.join(subdir, filename)
    return filepath

def load_coefs(fp):
    filepath = get_coef_path(fp)
    return load_array(filepath)

def save_array(savepath, array_in):
    if cfg.save_array_as == 'csv':
        np.savetxt(savepath, array_in, delimiter=",")
    elif cfg.save_array_as == 'npy':
        np.save(savepath, array_in)

def load_array(filepath):
    try:
        return np.load(filepath)
    except Exception as E:
        return np.genfromtxt(filepath)

def get_unmixing_mat(flourophore_list = None):
    #one option is to load it directly from the config so it can be input manually
    if flourophore_list is None:
        try:
            return cfg.unmixing_mat
        except AttributeError as E:
        #another is to load each unmixing vector for the fp and then concatenate
            flourophore_list = cfg.fps
    unmixing_mat = []
    for fp in cfg.fps:
        unmixing_mat.append(load_coefs(fp))
    return np.array(unmixing_mat)


def umixing_app_save(cfg, image, new_image):
    if cfg.save_original_tiff:
        new_filename = f"{cfg.filename}_original"
        write_composite_4d_tiff(image, cfg.save_path, new_filename, verbose=True, compression=cfg.compression,)

    if cfg.save_processed_tiff:
        new_filename = f"{cfg.filename}{bool(cfg.linearize_PMTs)*'_linearized'}{bool(cfg.unmix)*'_unmixed'}{bool(cfg.smoothing)*'_smoothed'}"
        write_composite_4d_tiff(new_image, cfg.save_path, new_filename, verbose=True, compression=cfg.compression)


