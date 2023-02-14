from src import config as cfg
from src import data_io as io
from src import plotting as plotting
from src import computation as comp
import numpy as np

from matplotlib import pyplot as plt

import os

def main_scatter_only(fp, i_number, j_number, channel_i, channel_j, alpha=0.01):
    # plot a scatter plot of the existing pixels
    fig, ax, title = plotting.plot_channels(
        channel_i, channel_j, i_number, j_number, alpha=alpha, label=fp, plot=False
    )

    # compute the means, smooth them and work with that from here on out for efficiency
    ordered_is, mean_js = comp.reduce_to_means(channel_i, channel_j)
    mean_js = comp.smooth(mean_js, 5)
    _, ax, _ = plotting.plot_channels(
        ordered_is, mean_js, i_number, j_number, alpha=1, label=fp, ax=ax, color="r"
    )
    io.savefig(fig, title)


def main(fp, i_number, j_number, channel_i, channel_j, alpha=0.01):
    main_scatter_only(fp, i_number, j_number, channel_i, channel_j, alpha)


    # plot and save the unmixing ratio that will be used for this image and pair of channels
    xs, ys, xs_per_y = comp.get_unmixing_ratio(ordered_is, mean_js)
    fig2, ax, title = plotting.plot_unmixing_vectors(
        xs, ys, i_number, j_number, label=fp, plot=True
    )
    io.savefig(fig2, title)

    # compute the PMT correction curve
    detected_photons, true_photons = comp.compute_PMT_nonlinearity(
        ordered_is, mean_js, xs_per_y
    )

    # save the PMT curves
    io.save_PMT_curve(detected_photons, true_photons, i=i_number, j=j_number, fp=fp)

    # Generate a smoothed curve since some of them are not...hoping they average out well and we don't have to rely on this
    f = comp.polyfit(true_photons, detected_photons)
    x = true_photons
    data_interp = np.piecewise(
        x,
        [x < cfg.max_lin_val, x >= cfg.max_lin_val],
        [
            lambda x: comp.correct_PMT_nonlinearity(x, detected_photons, true_photons),
            lambda x: np.polyval(f, x),
        ],
    )
    # mostly doing this to see if theres a speed improvement when we need to apply it to full images.
    # make sure to test both on timeit, because I don't particularly like this. Depends on how much fster the polynomial is

    # Plot the inferred nonlinearity, save the figure
    fig, ax = plotting.plot_pmt_nonlinearity(true_photons, detected_photons, plot=False)
    ax.plot(true_photons, data_interp, color="r")
    io.savefig(fig, f"PMT curve from {fp} on {i_number}{j_number}")

    # use the curve to correct the florescent values. This should come out linear .Save the plot
    max_val = np.maximum(ordered_is, mean_js)
    valid_is = ordered_is[max_val < np.max(detected_photons)]
    valid_js = mean_js[max_val < np.max(detected_photons)]

    corrected_i = comp.correct_PMT_nonlinearity(
        valid_is, detected_photons, true_photons
    )
    corrected_j = comp.correct_PMT_nonlinearity(
        valid_js, detected_photons, true_photons
    )

    fig, ax, title = plotting.plot_channels(
        corrected_i, corrected_j, i_number, j_number, alpha=1, label=f"{fp}_corrected"
    )
    io.savefig(fig, title)

    # not strictly necessary, but nice to be able to see how much noise there will be in your unmixing
    # save histograms of the variance around the mean value for a fixed x value. If you always got a certain number of pixels in y for a single pixel in X theres would be VERY tight.
    fig, ax, title = plotting.plot_spread(channel_i, channel_j, i_number, j_number)
    io.savefig(fig, f"spread_{j_number}_for_{fp}")

    # raise()

class UnmixingSession:
    def __init__(self):
        pass

    def my_init(self, verbose=False):
        supdir, filename = os.path.split(self.open_path)
        if self.save_path is None:
            self.save_path = self.open_path if os.path.isdir(self.open_path) else supdir
        self.filename = os.path.splitext(filename)[0]

        unmixing_mat = []
        for fp, coefs in self.unmixing_coefficient_dict.items():
            unmixing_mat.append(coefs)
        self.unmixing_mat= np.array(unmixing_mat).T
        self.num_channels = self.unmixing_mat.shape[0]

        if verbose:
            print('Using the following values:')
            print(f'Save path = {self.save_path}')
            print(f'Filename = {self.filename}')
            print(f'Number of channels = {self.num_channels}')
            print(f'Unmixing matrix = \n{self.unmixing_mat}')





def process_image(cfg, image):
    if cfg.linearize_PMTs:
        #new_image = main.linearize_PMTs(image)
        #Need to implement this ^^^
        new_image = image
    else:
        new_image = image

    if cfg.unmix:
        nonnegative= 'non_negative_least_squares' in cfg.handle_negatives.lower()
        new_image, residuals = comp.unmix(cfg.unmixing_mat, new_image, nonnegative = nonnegative, verbose=True)
        if 'set_to_zero' in cfg.handle_negatives.lower():
            new_image[new_image<0]=0
    else:
        residuals = None

    if cfg.smoothing:
        if 'original_spline_smoothing' in cfg.smoothing.lower():
            new_image = comp.original_spline_smoothing(new_image)

    return new_image, residuals

