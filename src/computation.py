# +
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

from src import config as cfg


def fp_from_tiffname(filename):
    idx = filename.lower().find("fp")
    return filename[idx - 1 : idx + 2]


def fake_pmt(actual_photons):
    if actual_photons < 15:
        return actual_photons
    if actual_photons < 300:
        return actual_photons / (1 + (actual_photons - 15) ** 2 / (300 - 15) ** 2)
    else:
        return 150


def reverse_fake_pmt(detected_photons):
    if detected_photons < 15:
        return detected_photons
    if detected_photons < 150:
        return detected_photons * (1 + (detected_photons - 15) ** 2 / (150 - 15) ** 2)
    else:
        return 300


def fake_pmt_n(photons_array, round=True):
    y = []
    x = []
    for actual_photons in photons_array:
        if round:
            y.append(np.round(fake_pmt(actual_photons)))
        else:
            y.append(fake_pmt(actual_photons))
        x.append(reverse_fake_pmt(y[-1]))
    return np.array(x), np.array(y)


def get_unmixing_ratio(xs_in, ys_in):
    xs = []
    ys = []
    # loop through until you hit nonlinearity. Compute what linear should be
    # lets choose valid range from config
    for x in range(cfg.min_lin_val, cfg.max_lin_val):
        y = np.mean(ys_in[xs_in == x])
        if (y <= cfg.max_lin_val) and (y >= cfg.min_lin_val):
            # we want to normalize the vector to put them all on the same playing field
            length = np.linalg.norm(np.array([x, y]))
            xs.append(x / length)
            ys.append(y / length)
        xs_per_y = np.mean(xs) / np.mean(ys)
    return xs, ys, xs_per_y


def switch_channels(chanX, chanY, xs_per_y):
    print("switching axis")
    xs_per_y = 1 / xs_per_y
    return chanY, chanX, xs_per_y


def compute_PMT_nonlinearity(chanX, chanY, xs_per_y):

    # now loop through and count the number of photons
    detected_photons = []
    true_photons = []

    max_x = np.max(chanX)
    max_y = np.max(chanY)
    if max_y > max_x:
        chanX, chanY, xs_per_y = switch_channels(chanX, chanY, xs_per_y)
    # print(max_x, max_y)
    for x in np.unique(chanX):
        if x < cfg.max_lin_val:
            detected_photons.append(x)
            true_photons.append(x)
        else:
            y = np.mean(chanY[chanX == x])
            if y > x:
                x, y, xs_per_y = switch_channels(x, y, xs_per_y)
                assert x > y
            if y > cfg.max_lin_val:
                # if the y value is out of the linear range then we need to correct that first
                # find the
                y = correct_PMT_nonlinearity(y, detected_photons, true_photons)

            expected_x = xs_per_y * y
            detected_photons.append(x)
            true_photons.append(expected_x)
    return detected_photons, true_photons


def correct_PMT_nonlinearity(
    photons_measured_to_correct, detected_photons_list, true_photons_list
):

    try:
        assert photons_measured_to_correct <= np.max(
            detected_photons_list
        ) and photons_measured_to_correct >= np.min(detected_photons_list)
    except AssertionError as E:
        raise (ValueError("the measured photons are not in the correctable range"))

    # use searchsorted to find where this measured photons would be placed (its a mean so won't be an int)
    idx = np.searchsorted(detected_photons_list, photons_measured_to_correct)
    # grab the detected photons and true photons on either side
    # use linspace to linearly interpolate both with the same number of points

    detected_dense = np.linspace(
        detected_photons_list[idx - 1], detected_photons_list[idx], 10**cfg.resolution
    )
    true_dense = np.linspace(
        true_photons_list[idx - 1], true_photons_list[idx], 10**cfg.resolution
    )

    # Find the index of the closest match between the number of detected photons we want to correct and the list of detected photons for which we have true photon values
    num_elements = len(detected_dense)
    points = np.tile(photons_measured_to_correct, (num_elements, 1))
    diff = detected_dense - points  #
    closest_match_idx = np.argmin(np.abs(diff))
    # print('##########')
    # print(detected_photons_list)
    # print(true_photons_list)
    # print(photons_measured_to_correct)
    # print(detected_dense)
    # print(true_dense)
    # print(points)
    # print(diff)
    # print(closest_match_idx)
    # print(true_dense[closest_match_idx])
    # take that index from the interpolated true photons
    return true_dense[closest_match_idx]
