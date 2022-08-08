import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# import os
import numpy as np

from src import demo_computation as comp


def plot_spectrum(ax, wavelength, values, label="", color=None, hatch="", alpha=0.2):
    if color is None:
        ax.plot(wavelength, values, label=label)
        # Fill under the curve
        ax.fill_between(x=wavelength, y1=values, alpha=alpha, hatch=hatch)
    else:
        ax.plot(wavelength, values, label=label, color=color)
        # Fill under the curve
        ax.fill_between(x=wavelength, y1=values, color=color, alpha=alpha, hatch=hatch)
    return ax


def plot_unmixing_ratios_2vec(A, ax, label_list=None):
    # 2 indicates that this is only intended to work when it is passed 2 channels as in the first 2-3 examples.
    N, C = A.shape
    numbered_channels = np.arange(0, C, 1)
    colors = get_FP_colors([1, 2])
    if label_list is None:
        label_list = ["FP1", "FP2"]
    ax.quiver(
        [0],
        [0],
        A[0, 0],
        A[1, 0],
        color=colors[0],
        alpha=0.5,
        angles="xy",
        scale_units="xy",
        scale=1,
        label=label_list[0],
    )
    ax.quiver(
        [0],
        [0],
        A[0, 1],
        A[1, 1],
        color=colors[1],
        alpha=0.5,
        angles="xy",
        scale_units="xy",
        scale=1,
        label=label_list[1],
    )

    ax.set_title("Unmixing ratios")
    ax.set_xlabel("Channel 1")
    ax.set_ylabel("channel 2")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    return ax


def plot_unmixing_ratios_bar(A, ax, label_list=None):
    C, N = A.shape
    colors = get_FP_colors(np.zeros(N))
    numbered_channels = np.arange(0, C, 1)
    # colors = plt.cm.rainbow(np.linspace(0, 1, N))
    if label_list is None:
        label_list = [f"fp{n}" for n in range(N)]
    for n in range(N):
        ax.plot(
            numbered_channels, A[:, n], label=label_list[n], alpha=1, color=colors[n]
        )
    ax.set_title("Unmixing ratios")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Percent Flouresence")
    ax.legend()
    ax.set_xticks(numbered_channels)
    return ax


def plot_flourescence_vals(b, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    C = b.shape[0]
    numbered_channels = np.arange(0, C, 1)
    ax.bar(numbered_channels, b, color="0")
    ax.set_title("Detected flourescnece values")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Photon count")
    ax.set_xticks(numbered_channels)
    return ax


def plot_flour_proportion(x, A, ax, label_list=None):
    C, N = A.shape
    colors = get_FP_colors(np.zeros(N))
    numbered_channels = np.arange(0, C, 1)
    # colors = plt.cm.rainbow(np.linspace(0, 1, N))
    bottom = np.zeros(C)
    if label_list is None:
        label_list = [f"fp{n}" for n in range(N)]
    for n in range(N):
        proportions = x[n] * A[:, n]
        ax.bar(
            numbered_channels,
            proportions,
            label=label_list[n],
            alpha=0.5,
            bottom=bottom,
            color=colors[n],
        )
        bottom = bottom + proportions
    ax.set_xlabel("Channel")
    ax.set_xticks(numbered_channels)
    ax.set_title("Proportion of flourescence for FP")
    ax.set_ylabel("Photon count")
    ax.legend()
    return ax


# plot a inferred flourophore amounts
def plot_flourophore_vals(x_known, x_inferred, ax, label_list=None):
    N = x_known.shape[0]
    x = np.arange(0, N, 1)
    colors = get_FP_colors(np.zeros(N))
    width = 0.35
    if label_list is None:
        label_list = [f"fp{n}" for n in range(N)]
    for flour in range(N):
        flour_b = np.zeros(N)
        flour_b[flour] = x_inferred[flour]
        # ax.bar(label_list, flour_b, alpha=.5, label=label_list[flour])
        rects1 = ax.bar(x - width / 2, flour_b, width, alpha=0.5, color=colors[flour])
        flour_b[flour] = x_known[flour]
        # color = rects1.patches[0].get_facecolor()
        # color_list.append(color)
        rects2 = ax.bar(
            x + width / 2, flour_b, width, hatch="///", color=colors[flour], alpha=0.5
        )
    legend_elements = [
        Patch(hatch="///", edgecolor="0", label="True Amount", facecolor="1"),
        Patch(edgecolor="0", label="Inferred Amount", facecolor="1"),
    ]

    ax.legend(handles=legend_elements)
    ax.set_xticks(x)
    ax.set_xticklabels(label_list)
    ax.set_title("Inferred flourophore amounts")
    ax.set_xlabel("Flourophore")
    ax.set_ylabel("Amount")
    # ax.set_xticks(label_list)
    return ax


def unmixing_plots(A, b, x_known, x_inferred, label_list=None, two_channels=False):
    # twoD is a bool. If there are more than two channels then we need to avoid the first plot
    fig, axs = plt.subplots(1, 4 + int(two_channels), figsize=(15, 3))
    plt.tight_layout()
    # plot the unmixing ratios
    if two_channels:
        ax = axs[0]
        ax = plot_unmixing_ratios_2vec(A, ax, label_list)

    ax = axs[0 + int(two_channels)]
    ax = plot_unmixing_ratios_bar(A, ax, label_list)

    # plot the computed floresencs values
    ax = axs[1 + int(two_channels)]
    plot_flourescence_vals(b, ax)

    # plot a stacked bar of the inferred flourophore amounts accounting for the floresence values
    ax = axs[2 + int(two_channels)]
    plot_flour_proportion(x_inferred, A, ax, label_list)

    ax = axs[3 + int(two_channels)]
    plot_flourophore_vals(x_known, x_inferred, ax, label_list)


def get_FP_colors(FP_list):
    cmap = plt.cm.get_cmap("cool")
    colors = []
    for i in np.arange(0, 1, 1 / len(FP_list)):
        colors.append(cmap(i))
    return colors


def get_filter_colors(filter_list):
    cmap = plt.cm.get_cmap("autumn")
    colors = []
    for i in np.arange(0, 1, 1 / len(filter_list))[::-1]:
        colors.append(cmap(i))
    return colors


def ex_em_spectra(fp_dict, filter_set_list=None, excitation_list=None):
    if not (type(filter_set_list) == list):
        filter_set_list = [filter_set_list]
    if not (type(excitation_list) == list):
        excitation_list = [excitation_list]
        # print(Filter_list_of_lists)
    num_sessions = max((len(filter_set_list), len(excitation_list)))
    plot_rows = 1
    if not (excitation_list[0] is None):
        plot_rows = 2
    print(num_sessions)
    fig, axs = plt.subplots(
        plot_rows,
        num_sessions,
        figsize=(num_sessions * 6, 4 * plot_rows),
        squeeze=False,
    )

    colors = get_FP_colors(fp_dict.keys())

    for i in range(num_sessions):
        if not (excitation_list[0] is None):
            try:
                Excitation = excitation_list[i]
            except IndexError as E:
                Excitation = excitation_list[0]

        try:
            filter_set = filter_set_list[i]
        except IndexError as E:
            filter_set = filter_set_list[0]
        filter_colors = get_filter_colors(filter_set)
        row_num = 0
        if not (excitation_list[0] is None):
            row_num = 1
        for k, (filter_name, df) in enumerate(filter_set.items()):
            # df = filter_dict[filter_name]
            plot_spectrum(
                axs[row_num, i],
                df.loc[:, "Wavelength"],
                df.loc[:, "Emission"] * 100,
                # label=f'{filter_name} Emisison',
                color=filter_colors[k],
                alpha=0.1,
            )

        # max_emission=0
        max_emission = 100 * 100
        for j, (fp, df) in enumerate(fp_dict.items()):
            # df = spectra_dict[FP]
            scale_by = 100
            row_num = 0
            if not (excitation_list[0] is None):
                plot_spectrum(
                    axs[0, i],
                    df.loc[:, "Wavelength"],
                    df.loc[:, "Excitation"],
                    label=f"{fp} Excitation",
                    color=colors[j],
                )
                scale_by = df.set_index("Wavelength").loc[Excitation, "Excitation"]
                row_num = 1
            # max_emission = max(max_emission, scale_by)
            plot_spectrum(
                axs[row_num, i],
                df.loc[:, "Wavelength"],
                df.loc[:, "Emission"] * scale_by,
                label=f"{fp} Emisison",
                color=colors[j],
            )

        if not (excitation_list[0] is None):
            df = comp.create_filter(Excitation, 3, max_em=100, steepness=10)
            plot_spectrum(
                axs[0, i],
                df.loc[:, "Wavelength"],
                df.loc[:, "Emission"],
                label=f"Excitation",
                color="0",
            )
            axs[0, i].legend()
            axs[0, i].set_title(f"Imaging Session {i}")
            axs[0, i].set_xlabel("Wavelength")
            axs[0, i].set_ylabel("Relative Excitation Percentage")

        # adjust the y axis to account for the scaling
        axs[row_num, i].set_title(f"Imaging session {i+1}")
        axs[row_num, i].set_yticks([0, max_emission / 2, max_emission])
        axs[row_num, i].set_yticklabels([0, 50, 100])
        axs[row_num, i].legend()
        axs[row_num, i].set_xlabel("Wavelength")
        axs[row_num, i].set_ylabel("Relative Emission Percentage")


def plot_photons(realistic_photons, perfect_photons=None, title="Excitation photons"):
    fig, ax = plt.subplots()
    wave_min = int(min(realistic_photons)) - 1
    wave_max = int(max(realistic_photons)) + 1
    bins = np.arange(wave_min, wave_max, 1) + 0.5

    width = 0.35

    values, edges = np.histogram(realistic_photons, bins=bins, density=True)
    rects1 = ax.bar(edges[:-1] - width / 2 + 0.5, values, width, color="0", alpha=0.5)

    if not (type(perfect_photons) is None):
        if type(perfect_photons) == pd.DataFrame:
            plot_spectrum(
                ax,
                perfect_photons.loc[:, "Wavelength"],
                perfect_photons.loc[:, "Emission"] / 100,
                color="0",
                hatch="///",
            )
            legend_elements = [
                Patch(
                    hatch="///",
                    edgecolor="0",
                    label="Perfect photons",
                    facecolor="0",
                    alpha=0.5,
                ),
                Line2D([0], [0], color="0", lw=2, label="Realistic photons"),
            ]
        else:
            values, edges = np.histogram(perfect_photons, bins=bins, density=True)
            rects2 = ax.bar(
                edges[:-1] + width / 2 + 0.5,
                values,
                width,
                hatch="///",
                color="0",
                alpha=0.5,
            )
            legend_elements = [
                Patch(
                    hatch="///",
                    edgecolor="0",
                    label="Perfect phtons",
                    facecolor="0",
                    alpha=0.5,
                ),
                Patch(
                    edgecolor="0", label="Realistic phtons", facecolor="0", alpha=0.5
                ),
            ]
    ax.legend(handles=legend_elements)
    ax.set_xlim((min(realistic_photons), max(realistic_photons)))
    ax.set_xlabel("Wavelength")
    if len(bins) < 11:
        ax.set_xticks(bins - 0.5)
    else:
        interval = (wave_max - wave_min) / 10
        if interval < 20:
            interval = 10
        if interval > 20:
            interval = 50
        xticks = np.arange(round(wave_min, -1), round(wave_max, -1), interval)
    ax.set_title(title)
    ax.set_ylabel("Fraction of photons")


def plot_emission_count(excited_FP_dict):
    perfect = []
    realistic = []
    for fp, count_dict in excited_FP_dict.items():
        perfect.append(count_dict["perfect"])
        realistic.append(count_dict["realistic"])
    fig, ax = plt.subplots()

    edges = np.arange(0, len(realistic), 1)
    width = 0.35
    rects1 = ax.bar(edges - width / 2, realistic, width=width, color="0", alpha=0.5)

    if not (perfect is None):
        rects2 = ax.bar(
            edges + width / 2, perfect, width=width, hatch="///", color="0", alpha=0.5
        )
        legend_elements = [
            Patch(
                hatch="///",
                edgecolor="0",
                label="Perfect_phtons",
                facecolor="0",
                alpha=0.5,
            ),
            Patch(edgecolor="0", label="Realistic_phtons", facecolor="0", alpha=0.5),
        ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel("Flourophore")
    ax.set_xticks(edges)
    print(list(excited_FP_dict.keys()))
    ax.set_xticklabels(list(excited_FP_dict.keys()))
    ax.set_title("Number of Photons Emitted From each Flourophore")
    ax.set_ylabel("Photons Emitted")
