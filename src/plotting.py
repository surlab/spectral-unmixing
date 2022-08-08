# +
from matplotlib import pyplot as plt
import numpy as np
from src import config as cfg


def new_ax(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    return fig, ax


def plot_pmt_nonlinearity(true_photons, detected_photons, plot=True, ax=None):
    # plot measured photons vs actual photons
    fig, ax = new_ax(ax)
    ax.plot(true_photons, detected_photons)
    ax.plot(true_photons, true_photons)
    ax.set_xlabel("true_photons")
    ax.set_ylabel("detected_photons")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("PMT nonlinearity")
    ax.set(xlim=(0, 300), ylim=(0, 300))
    if plot:
        plt.show()
    return fig, ax


def plot_channels_im(im, channel_i, channel_j, ax=None):
    plot_channels(
        im[:, :, :, channel_i], im[:, :, :, channel_j], channel_i, channel_j, ax=ax
    )


def plot_channels(x, y, i, j, plot=True, alpha=0.01, label="", ax=None, color="b"):
    fig, ax = new_ax(ax)
    ax.scatter(x, y, alpha=alpha, color=color)
    ax.plot((0, np.max(x)), (0, np.max(x)))
    ax.set_xlabel("Channel " + str(i))
    ax.set_ylabel("Channel " + str(j))
    title = f"Ch{i} vs Ch{j} for {label}"
    ax.set_title(str(title))
    ax.set_aspect("equal", adjustable="box")
    if plot:
        plt.show()
    return fig, ax, title


def plot_unmixing_vectors(xs, ys, channel_i, channel_j, label="", plot=True, ax=None):
    fig, ax = new_ax(ax)
    ax.scatter(xs, ys, color="b")
    ax.scatter(np.mean(xs), np.mean(ys), marker="+", color="r", s=50)
    ax.plot((0, 1), (0, 1))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel(f"Channel {channel_i}")
    ax.set_ylabel(f"Channel {channel_j}")
    ax.set_aspect("equal", adjustable="box")
    title = "Valid unmixing ratio for " + str(label)
    ax.set_title(title)
    if plot:
        plt.show()
    return fig, ax, title


def plot_spread(xs, ys, channel_i, channel_j, plot=True, ax=None):
    # want it to be narrow
    fig, ax = new_ax(ax)
    bins = np.arange(-cfg.spread_limit, cfg.spread_limit, cfg.spread_bin_size) - 0.5
    #
    for x in np.arange(cfg.min_lin_val, np.max(xs), cfg.spread_interval):
        values = ys[
            np.logical_and(xs >= x - cfg.combine_counts, xs <= x + cfg.combine_counts)
        ]
        # print(values[:100])
        # raise()
        if len(values) > 20:
            values = values - np.mean(values)
            y, these_bins = np.histogram(values, bins=bins, density=True)
            these_bins = bins[1:]
            ax.plot(these_bins, y, label=f"Channel {channel_i}={x}")
            # print(these_bins)
            # print(y)
            # print("#########")
    ax.set_xlabel("Channel" + str(channel_j) + " pixels from mean")
    ax.set_ylabel("Percentage of pixels")
    # ax.set_yscale('log')
    title = (
        "Spread around mean in Channel"
        + str(channel_j)
        + " for fixed Channel"
        + str(channel_i)
    )
    ax.set_title(title)
    ax.legend()
    if plot:
        plt.show()
    return fig, ax, title


def plot_PMT_curves(pmt_curves, plot=True, ax=None):
    fig, ax = new_ax(ax)
    photon_max = 0
    for chan_FP, curve_dict in pmt_curves.items():
        if not ("fake" in chan_FP.lower()):
            ax.plot(curve_dict["corrections"], curve_dict["counts"], label=chan_FP)
            photon_max = max(photon_max, np.max(curve_dict["corrections"]))
    ax.plot((0, photon_max), (0, photon_max))
    ax.set_xlabel("true_photons")
    ax.set_ylabel("detected_photons")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("PMT nonlinearities")
    ax.set(xlim=(0, 300), ylim=(0, 300))
    plt.legend()
    return fig, ax
