"""
Figure 1 configuration dictionaries.

This file is intentionally Figure-1-specific so configurations don't become unwieldy
as more figures are added.
"""

import copy


# ============================================================================
# Figure 1: Pipeline graphic
# ============================================================================

# Shared parameters for Figure 1 (common across multiple subpanels)
figure_1_params_shared = {
    "Fluorophores": ["TdTomato", "mCherry", "mNeptune"],
}

# Placeholder subpanel (new Panel A)
figure_1_params_placeholder_1a_presentation = {
    "name": "placeholder_panel_a",
    "placeholder_text": "TODO: add placeholder description for Panel A",
    # Keep presentation sizing consistent with other slide panels.
    "figsize": (8, 5),
}

# Presentation-sized defaults (good for slides)
figure_1_params_1a_presentation = {
    "name": "excitation_spectra",
    "Fluorophores": figure_1_params_shared["Fluorophores"],
    "Excitation wavelengths": [1040, 1080, 1180, 1240],
    "wavelength_range": (950, 1250),
    "smoothing_std": 5,
    "channel_labels": None,
    "figsize": (8, 5),
    "max_xticks": 4,
    "max_yticks": 3,
    "x_tick_step": 50,
    "y_tick_step": 0.5,
    "show_excitation_legend": False,
}

figure_1_params_1b_presentation = {
    "name": "emission_spectra",
    "Fluorophores": figure_1_params_shared["Fluorophores"],
    "emission_filters": ["Orange", "Red", "FarRed"],
    "wavelength_range": (530, 700),
    "use_display_ranges": True,
    "figsize": (8, 5),
    "max_xticks": 4,
    "max_yticks": 3,
    "x_tick_step": 50,
    "y_tick_step": 0.5,
}

figure_1_params_1c_presentation = {
    "name": "unmixing_vectors_bar_chart",
    "Fluorophores": figure_1_params_shared["Fluorophores"],
    "Channel 1": {
        "Excitation wavelength": 1040,
        "emission filter": "Orange",
        "power_mw": 40,
    },
    "Channel 2": {
        "Excitation wavelength": 1180,
        "emission filter": "Red",
        "power_mw": 30,
    },
    "Channel 3": {
        "Excitation wavelength": 1240,
        "emission filter": "FarRed",
        "power_mw": 40,
    },
    # Start new subpanels in manuscript sizing/style from the beginning.
    "figsize": (3.5, 2.4),
    "style": "manuscript",
    "font_scale": 1.25,
    "legend_fontsize": 8,
    "show_legend": False,
}

figure_1_params_1d_presentation = {
    "name": "predicted_angle",
    "Fluorophores": figure_1_params_shared["Fluorophores"],
    "Channel 1": {"Excitation wavelength": 1040, "emission filter": "Orange", "power_mw": 40},
    "Channel 2": {"Excitation wavelength": 1180, "emission filter": "Red", "power_mw": 30},
    "Channel 3": {"Excitation wavelength": 1240, "emission filter": "FarRed", "power_mw": 40},
    "figsize": (3.5, 2.4),
    "style": "manuscript",
    "font_scale": 1.25,
    "legend_fontsize": 8,
    "show_legend": False,
}

# 2-channel 2D scatterplot (legacy figure1.subpanel_5) using 1180 Red and 1240 FarRed
figure_1_params_1g_presentation = {
    "name": "two_channel_scatterplot",
    "Fluorophores": figure_1_params_shared["Fluorophores"],
    # Use the same 1080 Red / 1080 FarRed channels for G,
    # while reusing the balanced subselection + `pixel_labels` from shared H/I.
    "Channel 1": {"Excitation wavelength": 1080, "emission filter": "Red"},
    "Channel 2": {"Excitation wavelength": 1080, "emission filter": "FarRed"},
    "data_dir": "data/fig1_3color_inh_spatial_control_2p3_10072025",
    "figsize": (6, 6),
    "point_color_mode": "by_fp",
}

# 3-channel 3D scatterplot with classification cones (legacy figure2.subpanel_5)
figure_1_params_1h_presentation = {
    "name": "three_channel_3d_scatterplot",
    "data_dir": "data/fig1_3color_inh_spatial_control_2p3_10072025",
    "single_fp_data_dir": "data/fig1_fig2_1color_3mice_singleplane_june20250619",
    "figsize": (10, 8),
    # Keep shared_data source at the canonical 3-channel set:
    # 1040 Orange, 1180 Red, 1240 FarRed.
    "row_dict": {
        "Fluorophores": figure_1_params_shared["Fluorophores"],
        "Channel 1": {"Excitation wavelength": 1040, "emission filter": [550, 580]},  # Orange
        "Channel 2": {"Excitation wavelength": 1180, "emission filter": [590, 620]},  # Red
        "Channel 3": {"Excitation wavelength": 1240, "emission filter": [645, 695]},  # FarRed
    },
    # Balanced pixel selection before plotting (to avoid uneven apparent densities).
    "apply_balanced_pixel_selection": True,
    "balanced_bin_width": 100,
    "balanced_samples_per_bin": 300,
}

# 3-channel triangle projection (legacy figure2.subpanel_6)
figure_1_params_1i_presentation = {
    "name": "three_channel_triangle_projection",
    "data_dir": "data/fig1_3color_inh_spatial_control_2p3_10072025",
    "single_fp_data_dir": "data/fig1_fig2_1color_3mice_singleplane_june20250619",
    "figsize": (8, 8),
    # Same canonical 3-channel source as in panel 1h.
    "row_dict": {
        "Fluorophores": figure_1_params_shared["Fluorophores"],
        "Channel 1": {"Excitation wavelength": 1040, "emission filter": [550, 580]},  # Orange
        "Channel 2": {"Excitation wavelength": 1180, "emission filter": [590, 620]},  # Red
        "Channel 3": {"Excitation wavelength": 1240, "emission filter": [645, 695]},  # FarRed
    },
    # Alternate method: generic basis projection (N vectors -> N-gon)
    "projection_method": "nonlinear_star",
    "star_start_angle_deg": 90.0,
    # Allow signed weights for visualization (no clipping).
    "enforce_nonnegative_weights": False,
    # Balanced pixel selection before plotting (quadrants are based on projected vector directions).
    "apply_balanced_pixel_selection": True,
    "balanced_bin_width": 100,
    "balanced_samples_per_bin": 300,
}

#
# Panel 1j: angle histogram (balanced pixel subset)
#
figure_1_params_1j_presentation = {
    "name": "angle_histogram_balanced",
    "Fluorophores": figure_1_params_shared["Fluorophores"],
    # J uses two plotted axes, but both are pulled from the canonical
    # 3-channel shared_data source (1040/1180/1240).
    "Channel 1": {"Excitation wavelength": 1180, "emission filter": "Red"},
    "Channel 2": {"Excitation wavelength": 1240, "emission filter": "FarRed"},
    "figsize": (6, 2.6 * len(figure_1_params_shared["Fluorophores"])),
    "show_legend": True,
    "font_scale": 1.0,
}

figure_1_params_presentation = {
    "shared_params": figure_1_params_shared,
    "1a": figure_1_params_placeholder_1a_presentation,
    # Shift existing subpanels up by one letter.
    "1b": figure_1_params_1a_presentation,
    "1c": figure_1_params_1b_presentation,
    "1d": figure_1_params_1c_presentation,
    "1e": figure_1_params_1d_presentation,
    "1g": figure_1_params_1g_presentation,
    "1h": figure_1_params_1h_presentation,
    "1i": figure_1_params_1i_presentation,
    "1j": figure_1_params_1j_presentation,
}

# Manuscript-sized defaults (target ~single-column panel)
# Only override what must change; everything else inherits from presentation.
figure_1_params_manuscript = copy.deepcopy(figure_1_params_presentation)
for _k in ["1a", "1b", "1c", "1d", "1e"]:
    figure_1_params_manuscript[_k]["figsize"] = (3.5, 2.4)

# Manuscript styling tweaks: larger axis/title text, compact wrapped legends, limited ticks
for _k in ["1a", "1b", "1c", "1d", "1e"]:
    figure_1_params_manuscript[_k]["style"] = "manuscript"
    figure_1_params_manuscript[_k]["font_scale"] = 1.25
    figure_1_params_manuscript[_k]["legend_fontsize"] = 8
    figure_1_params_manuscript[_k]["max_xticks"] = 4
    figure_1_params_manuscript[_k]["max_yticks"] = 3

# Manuscript sizing tweaks for scatter/triangle panels
figure_1_params_manuscript["1g"]["figsize"] = (3.5, 3.5)
figure_1_params_manuscript["1h"]["figsize"] = (3.5, 3.5)
figure_1_params_manuscript["1i"]["figsize"] = (3.5, 3.5)
figure_1_params_manuscript["1h"]["style"] = "manuscript"
figure_1_params_manuscript["1i"]["style"] = "manuscript"
figure_1_params_manuscript["1j"]["figsize"] = (3.5, 3.2 * len(figure_1_params_shared["Fluorophores"]))
figure_1_params_manuscript["1j"]["style"] = "manuscript"
figure_1_params_manuscript["1j"]["show_legend"] = False

# Manuscript labels and legend visibility tweaks (Figure 1B/1C)
figure_1_params_manuscript["1b"]["ylabel"] = "2P Excitation"
figure_1_params_manuscript["1b"]["fp_legend_above"] = True
figure_1_params_manuscript["1b"]["show_excitation_legend"] = False
figure_1_params_manuscript["1b"]["fp_legend_fontscale"] = 1.3 * 1.2
figure_1_params_manuscript["1b"]["fp_legend_outline_scale"] = 2.0
figure_1_params_manuscript["1c"]["ylabel"] = "Emission"
figure_1_params_manuscript["1c"]["show_fp_legend"] = False


# Data/output directories
figure_1_data_dir = "data/fig1_fig2_1color_3mice_singleplane_june20250619"
figure_1_output_dir = "results/NewFigure1"

