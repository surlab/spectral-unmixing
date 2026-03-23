"""
Generic plotting functions for figure generation.

This module contains reusable plotting functions labeled A through Z.
These functions are designed to be config-driven and reusable across different figures.

Function naming convention: {Letter}_{descriptive_name}_plot()
For example: A_excitation_spectra_plot()
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src import config as cfg
from src.figure1 import (
    load_2p_spectra,
    apply_smoothing_to_spectrum,
    load_filter_transmission,
    get_power_from_pockels,
    compute_angle_to_vector,
)
from src.figure_scatterplot_helpers import (
    load_2p_spectra_flexible,
    compute_data_vector,
    compute_classification_zone,
    vector_angle,
)


def _compute_predicted_signals_per_fp(
    fluorophore_names,
    channel_configs,
    load_spectra_func=None,
    smoothing_std=None,
):
    """
    Shared predictor: compute predicted channel signals per fluorophore.

    This is the single "predictor" used by multiple figure panels:
    - Figure 1 subpanel 1d (C_unmixing_vectors_bar_chart_plot)
    - Figure 1 subpanel 1e (D_predicted_angle_with_nearest_linear_combo_plot)
    - Figure 1 subpanel 1g (2-channel scatterplot wrapper via external predicted_signals)

    Parameters
    ----------
    fluorophore_names : list[str]
    channel_configs : list[dict]
        Each config must include:
          - "Excitation wavelength"
          - "emission filter"
        Optionally:
          - "power_mw" or "power" or "pockels"
    load_spectra_func : callable, optional
        Loader with signature: load_spectra_func(fp_name) -> DataFrame with Wavelength/Excitation/Emission.
    smoothing_std : float | None
        If provided, apply Gaussian smoothing to excitation spectra before integration.

    Returns
    -------
    dict[str, list[float]]
        predicted[fp_name][channel_index] = predicted signal.
    """
    if load_spectra_func is None:
        load_spectra_func = load_2p_spectra_flexible

    # Load spectra for each FP once
    spectra_dict = {}
    for fp_name in fluorophore_names:
        df = load_spectra_func(fp_name)
        if smoothing_std is not None:
            df = apply_smoothing_to_spectrum(df, smoothing_std=float(smoothing_std))
        spectra_dict[fp_name] = df

    predicted = {fp: [] for fp in fluorophore_names}

    for ch in channel_configs:
        exc_wl = ch["Excitation wavelength"]
        filter_name = ch["emission filter"]

        # Resolve power for 2P scaling: (power/20)^2
        power_mw = None
        if "power_mw" in ch:
            try:
                power_mw = float(ch["power_mw"])
            except Exception:
                power_mw = None
        if power_mw is None and "power" in ch:
            try:
                power_mw = float(ch["power"])
            except Exception:
                power_mw = None
        if power_mw is None and "pockels" in ch and ch["pockels"] is not None:
            try:
                power_mw = float(
                    get_power_from_pockels(exc_wl, int(ch["pockels"]), power_mapping=None)
                )
            except Exception:
                power_mw = None
        if power_mw is None:
            power_mw = 20.0  # legacy default when no power/pockels provided
        power_factor = (power_mw / 20.0) ** 2

        # Determine emission integration window
        filter_min, filter_max = None, None
        if hasattr(cfg, "emission_filter_sets") and filter_name in cfg.emission_filter_sets:
            spec = cfg.emission_filter_sets[filter_name]
            if isinstance(spec, (list, tuple)) and len(spec) == 2:
                filter_min, filter_max = float(spec[0]), float(spec[1])
            else:
                # CSV-based filter definition: infer from transmission table
                t_df = load_filter_transmission(filter_name)
                if t_df is not None and len(t_df) > 0:
                    filter_min, filter_max = float(t_df["Wavelength"].min()), float(t_df["Wavelength"].max())

        if filter_min is None or filter_max is None:
            raise ValueError(f"Could not resolve emission wavelength range for filter '{filter_name}'")

        # Load filter transmission for weighted integration (if available)
        filter_transmission_df = load_filter_transmission(filter_name)

        for fp_name in fluorophore_names:
            df = spectra_dict[fp_name]

            # Excitation value at nearest excitation wavelength
            exc_idx = np.abs(df["Wavelength"].values - exc_wl).argmin()
            exc_value = float(df["Excitation"].values[exc_idx])

            # Emission integration window
            em_mask = (df["Wavelength"] >= filter_min) & (df["Wavelength"] <= filter_max)
            emission_in_range = df.loc[em_mask].copy()
            wavelengths = emission_in_range["Wavelength"].values
            emission_vals = emission_in_range["Emission"].values

            if len(wavelengths) == 0:
                predicted[fp_name].append(0.0)
                continue

            # Wavelength spacing for trapezoidal-like integration
            if len(wavelengths) > 1:
                diffs = np.diff(wavelengths)
                spacings = np.concatenate([[diffs[0]], (diffs[:-1] + diffs[1:]) / 2.0, [diffs[-1]]])
            else:
                spacings = np.array([1.0])

            # Apply filter transmission if we have it; otherwise assume 95%
            if filter_transmission_df is not None and len(filter_transmission_df) > 0:
                transmission_interp = np.interp(
                    wavelengths,
                    filter_transmission_df["Wavelength"].values,
                    filter_transmission_df["Transmission"].values,
                    left=0.0,
                    right=0.0,
                ) / 100.0
                filtered_emission = (emission_vals * transmission_interp * spacings).sum()
            else:
                filtered_emission = (emission_vals * spacings).sum() * 0.95

            signal = exc_value * filtered_emission * power_factor
            predicted[fp_name].append(float(signal))

    return predicted


def _set_x_ticks_nice(ax, wavelength_range, n_ticks=4, step=50):
    """
    Set a fixed number of x-axis ticks with "nice" values.

    Ticks are generated evenly across the range then snapped to the nearest `step`.
    It's OK if the outer ticks do not land exactly on the axis limits.
    """
    try:
        n_ticks = int(n_ticks)
    except Exception:
        n_ticks = 4
    n_ticks = max(2, n_ticks)
    try:
        step = float(step)
    except Exception:
        step = 50.0
    step = max(1.0, step)

    x0, x1 = float(wavelength_range[0]), float(wavelength_range[1])
    raw = np.linspace(x0, x1, n_ticks)
    snapped = np.round(raw / step) * step
    snapped = np.unique(snapped.astype(int))
    snapped = snapped[(snapped >= x0) & (snapped <= x1)]

    # Fallbacks if snapping collapsed ticks too much
    if len(snapped) < 2:
        snapped = np.array([int(np.round(x0 / step) * step), int(np.round(x1 / step) * step)])
        snapped = np.unique(snapped)
    ax.set_xticks(snapped.tolist())


def _set_y_ticks_nice(ax, n_ticks=3, step=0.5):
    """
    Set a fixed number of y-axis ticks with "nice" values.

    For our normalized plots this defaults to multiples of 0.5.
    If the axis extends above 1.0 (e.g., 1.15), we still keep ticks at 0, 0.5, 1.0.
    """
    try:
        n_ticks = int(n_ticks)
    except Exception:
        n_ticks = 3
    n_ticks = max(2, n_ticks)
    try:
        step = float(step)
    except Exception:
        step = 0.5
    step = max(0.1, step)

    y0, y1 = ax.get_ylim()
    y0 = float(y0)
    y1 = float(y1)

    # Most of our panels want [0, 0.5, 1.0]
    if n_ticks == 3 and abs(step - 0.5) < 1e-9 and y0 <= 0.0 and y1 >= 1.0:
        ax.set_yticks([0.0, 0.5, 1.0])
        return

    raw = np.linspace(y0, y1, n_ticks)
    snapped = np.round(raw / step) * step
    snapped = np.unique(snapped)
    ax.set_yticks(snapped.tolist())


def _center_legend_text(legend):
    """Center legend text within the legend box."""
    if legend is None:
        return
    try:
        legend._legend_box.align = "center"
    except Exception:
        pass
    for t in legend.get_texts():
        t.set_ha("center")


def _wrap_legend_label(label, width_chars=22):
    """Wrap legend labels to multiple lines for compact legends."""
    if label is None:
        return label
    return textwrap.fill(str(label), width=width_chars, break_long_words=False)


def A_excitation_spectra_plot(params_dict, ax=None, load_spectra_func=None):
    """
    Generic function A: Plot 2P excitation spectra for fluorophores with vertical laser lines.
    
    This is a generic, config-driven version of the excitation spectra plotting function.
    It can be reused across different figures by passing different parameter dictionaries.
    
    Parameters
    ----------
    params_dict : dict
        Configuration dictionary containing:
            - "Fluorophores": list of fluorophore names (e.g., ['mCherry', 'mNeptune'])
            - "Excitation wavelengths": list of excitation wavelengths in nm (optional)
            - "wavelength_range": tuple of (min, max) wavelength in nm (default: (950, 1250))
            - "smoothing_std": float, standard deviation for Gaussian smoothing in nm (default: 5)
            - "channel_labels": list of strings for excitation wavelength labels (optional)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    load_spectra_func : callable, optional
        Function to load spectra. If None, uses default load_2p_spectra.
        Should have signature: load_spectra_func(fluorophore_name, spectra_dir=None) -> DataFrame
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    figsize = params_dict.get("figsize", (8, 5))
    style = params_dict.get("style", "presentation")
    font_scale = float(params_dict.get("font_scale", 1.0))
    legend_fontsize = float(params_dict.get("legend_fontsize", 10.0))
    fp_legend_fontscale = float(params_dict.get("fp_legend_fontscale", 1.0))
    fp_legend_outline_scale = float(params_dict.get("fp_legend_outline_scale", 1.0))
    max_xticks = int(params_dict.get("max_xticks", 4))
    max_yticks = int(params_dict.get("max_yticks", 3))
    x_tick_step = float(params_dict.get("x_tick_step", 50))
    y_tick_step = float(params_dict.get("y_tick_step", 0.5))
    ylabel = params_dict.get("ylabel", "Normalized 2P Excitation")
    fp_legend_above = bool(params_dict.get("fp_legend_above", False))
    show_excitation_legend = bool(params_dict.get("show_excitation_legend", True))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract parameters with defaults
    fluorophore_names = params_dict.get("Fluorophores", [])
    if not fluorophore_names:
        raise ValueError("params_dict must contain 'Fluorophores' list")
    
    excitation_wavelengths = params_dict.get("Excitation wavelengths", None)
    wavelength_range = params_dict.get("wavelength_range", (950, 1250))
    smoothing_std = params_dict.get("smoothing_std", 5)
    channel_labels = params_dict.get("channel_labels", None)
    
    # Use provided loader or default
    # Prefer the flexible loader so Figure 1 configs can include TdTomato (and other FPs)
    if load_spectra_func is None:
        load_spectra_func = load_2p_spectra_flexible
    
    # Load spectra for each fluorophore
    spectra_dict = {}
    for fp_name in fluorophore_names:
        try:
            spectra_dict[fp_name] = load_spectra_func(fp_name)
        except Exception as e:
            print(f"Warning: Could not load spectra for {fp_name}: {e}")
            continue
    
    if not spectra_dict:
        raise ValueError("No spectra could be loaded for any fluorophore")
    
    # Filter to wavelength range and plot
    legend_patches = []
    for fp_name, df in spectra_dict.items():
        color = cfg.fluorophore_colors.get(fp_name, "#808080")
        # Filter to wavelength range
        mask = (df["Wavelength"] >= wavelength_range[0]) & (df["Wavelength"] <= wavelength_range[1])
        df_filtered = df[mask].copy()
        
        if len(df_filtered) == 0:
            print(f"Warning: No data in wavelength range {wavelength_range} for {fp_name}")
            continue
        
        # Apply Gaussian smoothing
        df_filtered = apply_smoothing_to_spectrum(df_filtered, smoothing_std=smoothing_std)
        smoothed_excitation = df_filtered["Excitation"].values
        
        # Normalize excitation to max = 1 for visibility
        max_excitation = smoothed_excitation.max()
        if max_excitation > 0:
            normalized_excitation = smoothed_excitation / max_excitation
        else:
            normalized_excitation = smoothed_excitation
        
        # Plot excitation spectrum
        ax.plot(
            df_filtered["Wavelength"],
            normalized_excitation,
            color=color,
            linewidth=2
        )
        ax.fill_between(
            df_filtered["Wavelength"],
            normalized_excitation,
            alpha=0.3,
            color=color
        )
        
        # Create patch for legend matching the shaded fill (alpha=0.3)
        legend_patches.append(
            Patch(facecolor=color, label=fp_name, alpha=0.3)
        )
    
    # Plot vertical lines for excitation wavelengths and collect a single legend handle
    line_handles = []
    line_handle_for_legend = None
    if excitation_wavelengths is not None:
        for idx, exc_wl in enumerate(excitation_wavelengths):
            if wavelength_range[0] <= exc_wl <= wavelength_range[1]:
                # Get color and style from config (cycle if more than 2 channels)
                line_color = cfg.excitation_line_colors[idx % len(cfg.excitation_line_colors)]
                line_style = cfg.excitation_line_styles[idx % len(cfg.excitation_line_styles)]
                
                line = ax.axvline(
                    exc_wl,
                    color=line_color,
                    linestyle=line_style,
                    linewidth=3,
                    alpha=0.7,
                    label=None  # Legend label handled separately
                )
                line_handles.append(line)
                
                # Create a single dummy handle for the legend to represent all excitation lines
                if line_handle_for_legend is None:
                    from matplotlib.lines import Line2D
                    line_handle_for_legend = Line2D(
                        [0], [0],
                        color="#000000",
                        linestyle="--",
                        linewidth=3,
                        alpha=0.7,
                        label="Selected excitation wavelengths",
                    )
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title("2P Excitation Spectra" if style != "manuscript" else "")
    ax.set_xlim(wavelength_range)
    _set_x_ticks_nice(ax, wavelength_range, n_ticks=max_xticks, step=x_tick_step)
    # Keep x-axis exactly at 0 (avoid matplotlib auto-padding below zero)
    ax.set_ylim(bottom=0)
    ax.margins(y=0)

    # Manuscript styling: slightly larger axis/title text, but do not scale legend text
    if style != "manuscript":
        ax.title.set_fontsize(14 * font_scale)
    ax.xaxis.label.set_fontsize(12 * font_scale)
    ax.yaxis.label.set_fontsize(12 * font_scale)
    ax.tick_params(axis="both", labelsize=10 * font_scale)
    _set_y_ticks_nice(ax, n_ticks=max_yticks, step=y_tick_step)
    
    # Create separate legends:
    #   1) Fluorophore patches
    #   2) Selected excitation wavelengths (line), in its own box
    if legend_patches:
        num_fps = len(legend_patches)

        if style == "manuscript" and fp_legend_above:
            from matplotlib.lines import Line2D

            labels = [h.get_label() for h in legend_patches]

            # Single legend row above plot. Instead of separate patches, highlight each label
            # by giving the text a semi-transparent colored background.
            invisible = Line2D([0], [0], color="none", linestyle="", marker=None)
            leg_labels = ax.legend(
                handles=[invisible] * num_fps,
                labels=labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=num_fps,
                frameon=True,
                fontsize=legend_fontsize * fp_legend_fontscale,
                handlelength=0.0,
                handletextpad=0.0,
                columnspacing=1.2,
                borderaxespad=0.0,
            )
            # White background, no outline
            leg_labels.get_frame().set_facecolor("white")
            leg_labels.get_frame().set_edgecolor("none")
            leg_labels.get_frame().set_linewidth(0.0)
            leg_labels.get_frame().set_alpha(1.0)

            # Apply per-label highlight color
            for t, h in zip(leg_labels.get_texts(), legend_patches):
                rgb = h.get_facecolor()
                # Ensure we have an (r,g,b,a) tuple and then control face/edge alpha separately.
                rgba_edge = mcolors.to_rgba(rgb, alpha=1.0)
                rgba_face = mcolors.to_rgba(rgb, alpha=0.35)
                t.set_bbox(
                    dict(
                        facecolor=rgba_face,
                        edgecolor=rgba_edge,
                        linewidth=1.0 * fp_legend_outline_scale,
                        # Slightly larger padding so thick outline doesn't look inset.
                        boxstyle="round,pad=0.30",
                    )
                )

            _center_legend_text(leg_labels)
            ax.add_artist(leg_labels)
        else:
            # Default (presentation) legend in-box near bottom
            leg_fps = ax.legend(
                handles=legend_patches,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.08),
                ncol=num_fps,
                frameon=True,
                fontsize=legend_fontsize,
            )
            leg_fps.get_frame().set_facecolor("white")
            leg_fps.get_frame().set_edgecolor("0.8")
            leg_fps.get_frame().set_alpha(1.0)
            _center_legend_text(leg_fps)
            ax.add_artist(leg_fps)

    if line_handle_for_legend is not None:
        from matplotlib.lines import Line2D

        # Single-entry legend for excitation wavelengths
        # Wrap label for manuscript so it stays within axis width.
        if style == "manuscript":
            line_handle_for_legend.set_label(_wrap_legend_label(line_handle_for_legend.get_label(), width_chars=20))
        if show_excitation_legend:
            leg_exc = ax.legend(
                handles=[line_handle_for_legend],
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=1,
                frameon=True,
                fontsize=legend_fontsize,
            )
            leg_exc.get_frame().set_facecolor("white")
            leg_exc.get_frame().set_edgecolor("0.8")
            leg_exc.get_frame().set_alpha(1.0)  # Fully opaque
            _center_legend_text(leg_exc)
    
    # Clean panel style
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return fig, ax


def B_emission_spectra_plot(params_dict, ax=None, load_spectra_func=None):
    """
    Generic function B: Plot 1P emission spectra for fluorophores with emission filters overlaid.
    
    Configuration dictionary fields:
        - "Fluorophores": list of fluorophore names (e.g., ['mCherry', 'mNeptune'])
        - "emission_filters": list of filter keys from cfg.emission_filter_sets / cfg.emission_filter_display_ranges
        - "wavelength_range": tuple of (min, max) wavelength in nm (default: (500, 700))
        - "use_display_ranges": bool, if True use cfg.emission_filter_display_ranges for filter shading
    """
    figsize = params_dict.get("figsize", (8, 5))
    style = params_dict.get("style", "presentation")
    font_scale = float(params_dict.get("font_scale", 1.0))
    legend_fontsize = float(params_dict.get("legend_fontsize", 10.0))
    max_xticks = int(params_dict.get("max_xticks", 4))
    max_yticks = int(params_dict.get("max_yticks", 3))
    x_tick_step = float(params_dict.get("x_tick_step", 50))
    y_tick_step = float(params_dict.get("y_tick_step", 0.5))
    ylabel = params_dict.get("ylabel", "Normalized 1P Emission")
    show_fp_legend = bool(params_dict.get("show_fp_legend", True))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    fluorophore_names = params_dict.get("Fluorophores", [])
    if not fluorophore_names:
        raise ValueError("params_dict must contain 'Fluorophores' list")
    
    filter_keys = params_dict.get("emission_filters", [])
    wavelength_range = params_dict.get("wavelength_range", (500, 700))
    use_display_ranges = params_dict.get("use_display_ranges", True)
    filter_box_top = float(params_dict.get("filter_box_top", 1.05))
    filter_box_alpha = float(params_dict.get("filter_box_alpha", 0.22))
    
    if not filter_keys:
        raise ValueError("params_dict for B_emission_spectra_plot must contain 'emission_filters' list")
    
    # Use the flexible loader so we consistently get Wavelength / Emission columns
    if load_spectra_func is None:
        load_spectra_func = load_2p_spectra_flexible
    
    # Load spectra and plot emission curves
    fp_patches = []
    for fp_name in fluorophore_names:
        try:
            df = load_spectra_func(fp_name)
        except Exception as e:
            print(f"Warning: Could not load spectra for {fp_name}: {e}")
            continue
        
        color = cfg.fluorophore_colors.get(fp_name, "#808080")
        mask = (df["Wavelength"] >= wavelength_range[0]) & (df["Wavelength"] <= wavelength_range[1])
        df_filtered = df[mask].copy()
        if len(df_filtered) == 0:
            print(f"Warning: No emission data in wavelength range {wavelength_range} for {fp_name}")
            continue
        
        emission = df_filtered["Emission"].values
        max_em = emission.max()
        if max_em > 0:
            emission_norm = emission / max_em
        else:
            emission_norm = emission
        
        ax.plot(
            df_filtered["Wavelength"],
            emission_norm,
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            df_filtered["Wavelength"],
            emission_norm,
            alpha=0.3,
            color=color,
        )
        
        fp_patches.append(Patch(facecolor=color, label=fp_name, alpha=0.3))
    
    # Plot emission filters as shaded vertical bands
    filter_patches = []
    for filt_key in filter_keys:
        # Determine display range
        if use_display_ranges and filt_key in cfg.emission_filter_display_ranges:
            f_min, f_max = cfg.emission_filter_display_ranges[filt_key]
        else:
            filt_def = cfg.emission_filter_sets.get(filt_key, None)
            if isinstance(filt_def, (list, tuple)) and len(filt_def) == 2:
                f_min, f_max = filt_def
            else:
                # If we only have a CSV, fall back to entire wavelength_range
                f_min, f_max = wavelength_range
        
        # Clip to overall wavelength range for display
        f_min_clipped = max(f_min, wavelength_range[0])
        f_max_clipped = min(f_max, wavelength_range[1])
        if f_min_clipped >= f_max_clipped:
            continue
        
        filt_color = cfg.emission_filter_colors.get(filt_key, "#B0B0B0")
        rect = Rectangle(
            (f_min_clipped, 0),
            f_max_clipped - f_min_clipped,
            filter_box_top,
            facecolor=filt_color,
            alpha=filter_box_alpha,
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(rect)
        
        display_name = cfg.filter_display_names.get(filt_key, filt_key)
        filter_patches.append(Patch(facecolor=filt_color, label=display_name, alpha=0.4))
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title("1P Emission Spectra with Emission Filters" if style != "manuscript" else "")
    ax.set_xlim(wavelength_range)
    _set_x_ticks_nice(ax, wavelength_range, n_ticks=max_xticks, step=x_tick_step)
    ax.set_ylim(0, 1.15)
    ax.margins(y=0)

    # Manuscript styling: slightly larger axis/title text, but do not scale legend text
    if style != "manuscript":
        ax.title.set_fontsize(14 * font_scale)
    ax.xaxis.label.set_fontsize(12 * font_scale)
    ax.yaxis.label.set_fontsize(12 * font_scale)
    ax.tick_params(axis="both", labelsize=10 * font_scale)
    _set_y_ticks_nice(ax, n_ticks=max_yticks, step=y_tick_step)
    
    # Legends: fluorophores at bottom (like subpanel 1a); filters are labeled directly over bands
    if fp_patches and show_fp_legend and style != "manuscript":
        n_fp = len(fp_patches)
        leg_fp = ax.legend(
            handles=fp_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),  # Move closer to x-axis since no second legend box
            ncol=n_fp,
            frameon=True,
            fontsize=legend_fontsize,
        )
        leg_fp.get_frame().set_facecolor("white")
        leg_fp.get_frame().set_edgecolor("0.8")
        leg_fp.get_frame().set_alpha(1.0)
        _center_legend_text(leg_fp)
        ax.add_artist(leg_fp)
    
    # Direct labels for filters: text with horizontal bar centered over each band
    for filt_key in filter_keys:
        if use_display_ranges and filt_key in cfg.emission_filter_display_ranges:
            f_min, f_max = cfg.emission_filter_display_ranges[filt_key]
        else:
            filt_def = cfg.emission_filter_sets.get(filt_key, None)
            if isinstance(filt_def, (list, tuple)) and len(filt_def) == 2:
                f_min, f_max = filt_def
            else:
                f_min, f_max = wavelength_range
        f_min_clipped = max(f_min, wavelength_range[0])
        f_max_clipped = min(f_max, wavelength_range[1])
        if f_min_clipped >= f_max_clipped:
            continue
        x_center = 0.5 * (f_min_clipped + f_max_clipped)
        # Place bar and label so they sit just on top of the shaded filter boxes
        # (boxes extend to filter_box_top in y); bar at filter_box_top, label just above.
        y_bar = filter_box_top
        y_text = filter_box_top + 0.03
        display_name = cfg.filter_display_names.get(filt_key, filt_key)
        # Draw horizontal bar
        ax.hlines(
            y=y_bar,
            xmin=f_min_clipped,
            xmax=f_max_clipped,
            color=cfg.emission_filter_colors.get(filt_key, "#000000"),
            linewidth=2,
        )
        # Draw text label centered above bar
        ax.text(
            x_center,
            y_text,
            display_name,
            ha="center",
            va="bottom",
            fontsize=10 * font_scale,
            color="black",
        )
    
    # Clean panel style
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return fig, ax


def C_unmixing_vectors_bar_chart_plot(params_dict, ax=None, load_spectra_func=None):
    """
    Generic function C: Predicted unmixing vectors as stacked bar charts.

    Expected params_dict fields:
      - "Fluorophores": list of fluorophore names
      - Channel configs under keys like "Channel 1", "Channel 2", ... each with:
          - "Excitation wavelength"
          - "emission filter"
      - Optional:
          - "figsize": figure size tuple
          - "style": "presentation" or "manuscript"
          - "font_scale", "legend_fontsize"
          - "show_legend": default True
    """
    base_figsize = params_dict.get("figsize", (3.5, 2.4))
    figsize = base_figsize
    style = params_dict.get("style", "presentation")
    font_scale = float(params_dict.get("font_scale", 1.0))
    legend_fontsize = float(params_dict.get("legend_fontsize", 10.0))
    show_legend = bool(params_dict.get("show_legend", True))

    fluorophore_names = params_dict.get("Fluorophores", [])
    if not fluorophore_names:
        raise ValueError("params_dict must contain 'Fluorophores' list")

    # Parse channel configs in order
    channel_items = []
    for k, v in params_dict.items():
        if not k.lower().startswith("channel "):
            continue
        try:
            idx = int(k.split()[1])
        except Exception:
            continue
        channel_items.append((idx, v))
    channel_items.sort(key=lambda x: x[0])

    if not channel_items:
        raise ValueError("params_dict must define channel configs under keys like 'Channel 1'")

    channel_configs = [v for _, v in channel_items]

    if load_spectra_func is None:
        load_spectra_func = load_2p_spectra_flexible

    n_channels = len(channel_configs)
    n_subplots = len(fluorophore_names) + 1  # overlay + each FP

    # Plot width uses `figsize` directly (no additional scaling).

    # Scale plot width with the number of channels:
    # - 3 channels: 0.75x
    # - 6+ channels: 1.0x
    # - 4-5 channels: linear interpolation
    if n_channels <= 3:
        width_scale = 0.75
    elif n_channels >= 6:
        width_scale = 1.0
    else:
        width_scale = 0.75 + (n_channels - 3) * ((1.0 - 0.75) / 3.0)

    figsize = (figsize[0] * width_scale, figsize[1])

    # Create figure and stacked axes
    fig, axes = plt.subplots(
        n_subplots,
        1,
        figsize=figsize,
        sharex=True,
        # Increased vertical space to prevent tick-label overlap
        gridspec_kw={"hspace": 0.22},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    x_pos = np.arange(n_channels, dtype=float)
    # Slightly narrower than full bin width (1.0) so bars don't look "overweight",
    # while keeping adjacent bars visually tight.
    bar_width = float(params_dict.get("bar_width", 0.98 if n_channels <= 5 else 0.9))
    axes_xlim = (-0.5, n_channels - 0.5)

    # Shared predictor (unified across 1d/1e and optional 1g)
    predicted = _compute_predicted_signals_per_fp(
        fluorophore_names=fluorophore_names,
        channel_configs=channel_configs,
        load_spectra_func=load_spectra_func,
        smoothing_std=5,
    )

    # Normalize each FP's signals so its brightest channel is 1
    normalized = {}
    for fp_name in fluorophore_names:
        arr = np.array(predicted[fp_name], dtype=float)
        maxv = float(arr.max()) if len(arr) else 0.0
        if maxv > 0:
            normalized[fp_name] = arr / maxv
        else:
            normalized[fp_name] = arr

    # Channel x-axis labels
    channel_labels = []
    for ch in channel_configs:
        exc_wl = ch["Excitation wavelength"]
        filter_name = ch["emission filter"]
        display_filter = cfg.filter_display_names.get(filter_name, filter_name)
        channel_labels.append(f"{exc_wl}\n{display_filter}")

    # Individual FP subplots (top->bottom)
    for si, fp_name in enumerate(fluorophore_names):
        ax_sub = axes[si]
        fp_color = cfg.fluorophore_colors.get(fp_name, "#808080")
        rgba_edge = mcolors.to_rgba(fp_color, alpha=1.0)
        face_alpha = 0.35
        face_rgba = mcolors.to_rgba(fp_color, alpha=face_alpha)
        edge_rgba = mcolors.to_rgba(fp_color, alpha=1.0)

        ax_sub.bar(
            x_pos,
            normalized[fp_name],
            bar_width,
            color=face_rgba,
            edgecolor=edge_rgba,
            linewidth=2,
        )
        ax_sub.set_ylim(0, 1.05)
        ax_sub.set_yticks([0.0, 1.0])
        ax_sub.set_yticklabels(["0", "1"])
        ax_sub.tick_params(axis="y", labelleft=True, length=2, labelsize=5 * font_scale)
        ax_sub.spines["top"].set_visible(False)
        ax_sub.spines["right"].set_visible(False)

    # Overlay of all FPs on the bottom subplot
    ax_overlay = axes[-1]
    for fp_name in fluorophore_names:
        fp_color = cfg.fluorophore_colors.get(fp_name, "#808080")

        face_alpha = 0.25
        face_rgba = mcolors.to_rgba(fp_color, alpha=face_alpha)
        edge_rgba = mcolors.to_rgba(fp_color, alpha=1.0)

        ax_overlay.bar(
            x_pos,
            normalized[fp_name],
            bar_width,
            color=face_rgba,
            edgecolor=edge_rgba,
            linewidth=2,
        )
    ax_overlay.set_ylim(0, 1.05)
    ax_overlay.set_yticks([0.0, 1.0])
    ax_overlay.set_yticklabels(["0", "1"])
    ax_overlay.tick_params(axis="y", labelleft=True, length=2, labelsize=5 * font_scale)
    ax_overlay.spines["top"].set_visible(False)
    ax_overlay.spines["right"].set_visible(False)

    for ax_sub in axes:
        ax_sub.set_xlim(*axes_xlim)
        ax_sub.grid(False)

    # X axis labels only on bottom subplot (overlay)
    ax_bot = axes[-1]
    ax_bot.set_xticks(x_pos)
    ax_bot.set_xticklabels(channel_labels, fontsize=9 * font_scale)
    ax_bot.set_xlabel("Channel", fontsize=11 * font_scale)

    # Right-side subplot labels with highlight + colored outlines (like 1a)
    label_texts = fluorophore_names + ["Overlay"]
    label_colors = [cfg.fluorophore_colors.get(fp, "#808080") for fp in fluorophore_names] + ["#808080"]

    highlight_alpha = 0.35
    outline_lw = 2.0
    # Match the highlight box sizing approach used in panel 1a
    label_pad = 0.30
    label_x = 1.05
    label_y = 0.5

    for ax_sub, txt, c in zip(axes, label_texts, label_colors):
        rgb_face = mcolors.to_rgba(c, alpha=highlight_alpha)
        rgb_edge = mcolors.to_rgba(c, alpha=1.0)
        ax_sub.text(
            label_x,
            label_y,
            txt,
            transform=ax_sub.transAxes,
            ha="left",
            va="center",
            fontsize=legend_fontsize,
            color="black",
            bbox=dict(
                facecolor=rgb_face,
                edgecolor=rgb_edge,
                linewidth=outline_lw,
                boxstyle=f"round,pad={label_pad}",
            ),
            clip_on=False,
        )

    # Master y-axis label (centered vertically)
    fig.text(
        0.03,
        0.5,
        "Relative Signal",
        rotation="vertical",
        va="center",
        ha="center",
        fontsize=12 * font_scale,
    )

    # Hide legend for this subpanel (labels replace legend for this panel)
    # (If show_legend is enabled anyway, we still keep it hidden.)
    plt.tight_layout(rect=(0.08, 0.0, 0.84, 1))
    return fig, axes


def D_predicted_angle_with_nearest_linear_combo_plot(params_dict, ax=None, load_spectra_func=None):
    """
    Generic function D: Predicted angle between target fluorophore vector and:
      1) each individual other fluorophore vector
      2) the orthogonal projection of the target vector onto the span of the others

    This mirrors the "Figure 5 subpanel 5" style: for each target FP, draw horizontal
    lines at angles (0..90 degrees). Lines are colored by the corresponding FP.
    An additional dashed line shows the "nearest linear combination" angle, computed
    via orthogonal projection of the target vector onto the span of the others.

    Required params_dict fields:
      - "Fluorophores": list of fluorophore names
      - channel configs in keys: "Channel 1", "Channel 2", ... each with:
          - "Excitation wavelength"
          - "emission filter"
          - optional "power_mw" (preferred) or "power" (fallback)
    Optional params_dict:
      - style: "presentation" or "manuscript"
      - font_scale, legend_fontsize
      - show_legend: bool
    """
    base_figsize = params_dict.get("figsize", (3.5, 2.4))
    style = params_dict.get("style", "presentation")
    font_scale = float(params_dict.get("font_scale", 1.0))
    label_fontsize = float(params_dict.get("legend_fontsize", 10.0))
    # Keep this panel legend-free; labels replace legend.

    fluorophore_names = params_dict.get("Fluorophores", [])
    if not fluorophore_names:
        raise ValueError("params_dict must contain 'Fluorophores' list")

    # Parse channel configs in order
    channel_items = []
    for k, v in params_dict.items():
        if not k.lower().startswith("channel "):
            continue
        try:
            idx = int(k.split()[1])
        except Exception:
            continue
        channel_items.append((idx, v))
    channel_items.sort(key=lambda x: x[0])
    if not channel_items:
        raise ValueError("params_dict must define channels under keys like 'Channel 1'")
    channel_configs = [v for _, v in channel_items]

    if load_spectra_func is None:
        load_spectra_func = load_2p_spectra_flexible

    # Shared predictor (unified across 1d/1e and optional 1g)
    predicted = _compute_predicted_signals_per_fp(
        fluorophore_names=fluorophore_names,
        channel_configs=channel_configs,
        load_spectra_func=load_spectra_func,
        smoothing_std=None,  # preserve existing Figure 1e/D behavior
    )

    # Convert predicted signals to unit vectors (direction only matters for angles)
    unit_vectors = {}
    for fp_name in fluorophore_names:
        v = np.array(predicted[fp_name], dtype=float)
        norm = float(np.linalg.norm(v))
        if norm > 0:
            unit_vectors[fp_name] = v / norm
        else:
            unit_vectors[fp_name] = v

    def angle_deg_between(u, w):
        dot = float(np.clip(np.dot(u, w), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(dot)))
        return ang if ang <= 90 else 180 - ang

    # Figure 5-like layout: subplots in columns (1 row)
    n_subplots = len(fluorophore_names)
    # For 3 FPs specifically this should not be overly wide.
    # Heuristic: keep <=3 very compact, then scale with n/3.
    if n_subplots <= 3:
        figsize = (base_figsize[0] * 0.62, base_figsize[1])
    else:
        figsize = (base_figsize[0] * 0.62 * (n_subplots / 3.0), base_figsize[1])

    fig, axes = plt.subplots(
        1,
        n_subplots,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"wspace": 0.25},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Line range (Figure 5 uses extended horizontal line span)
    x_min, x_max = -3.0, 5.0
    line_positions = np.array([x_min, x_max])

    highlight_alpha = 0.35
    outline_lw = 2.0
    label_pad = 0.25

    def add_highlight_text_data(ax_sub, x, y, txt, color, rotation=0.0, ha="center", va="center"):
        """Highlighted text in data coordinates."""
        rgba_face = mcolors.to_rgba(color, alpha=highlight_alpha)
        rgba_edge = mcolors.to_rgba(color, alpha=1.0)
        ax_sub.text(
            x,
            y,
            txt,
            transform=ax_sub.transData,
            ha=ha,
            va=va,
            rotation=rotation,
            fontsize=label_fontsize,
            color="black",
            bbox=dict(
                facecolor=rgba_face,
                edgecolor=rgba_edge,
                linewidth=outline_lw,
                boxstyle=f"round,pad={label_pad}",
            ),
            clip_on=False,
        )

    for ax_i, target_fp in enumerate(fluorophore_names):
        ax_sub = axes[ax_i]

        # Compute angles to all other FPs
        other_angles = []
        for other_fp in fluorophore_names:
            if other_fp == target_fp:
                continue
            ang = angle_deg_between(unit_vectors[target_fp], unit_vectors[other_fp])
            other_angles.append((ang, other_fp))

        other_angles.sort(key=lambda x: x[0])

        # Labeling policy to reduce clutter:
        # - 2-3 FPs: only label the target FP
        # - 4-5 FPs: label target + closest 1
        # - 6+ FPs: label target + closest 2
        if n_subplots <= 3:
            num_extra_labels = 0
        elif n_subplots <= 5:
            num_extra_labels = 1
        else:
            num_extra_labels = 2
        nearest_others = other_angles[:num_extra_labels]

        target_color = cfg.fluorophore_colors.get(target_fp, "#808080")

        # Target at 0 degrees
        ax_sub.plot(line_positions, [0, 0], color=target_color, linewidth=6, alpha=0.7)

        # Individual FP angles (colored)
        for ang, other_fp in other_angles:
            other_color = cfg.fluorophore_colors.get(other_fp, "#808080")
            ax_sub.plot(line_positions, [ang, ang], color=other_color, linewidth=3, alpha=0.7)

        # Best linear combo angle via orthogonal projection
        others = [fp for fp in fluorophore_names if fp != target_fp]
        if len(others) == 0:
            ang_combo = 90.0
        else:
            V = np.stack([unit_vectors[fp] for fp in others], axis=1)
            v_t = unit_vectors[target_fp]
            proj = V @ (np.linalg.pinv(V) @ v_t)
            proj_norm = float(np.linalg.norm(proj))
            if proj_norm > 0:
                ang_combo = angle_deg_between(v_t, proj / proj_norm)
            else:
                ang_combo = 90.0

        ax_sub.plot(line_positions, [ang_combo, ang_combo], color="black", linewidth=2, linestyle="--", alpha=0.9)

        # Axis formatting
        ax_sub.set_ylim(0, 90)
        ax_sub.set_yticks([0, 45, 90])
        ax_sub.set_yticklabels([0, 45, 90])
        ax_sub.set_xticks([])
        ax_sub.grid(False)
        ax_sub.spines["top"].set_visible(False)
        ax_sub.spines["right"].set_visible(False)
        ax_sub.spines["bottom"].set_visible(False)

        # Target label
        # For <6 FPs: horizontal and just below the 0-deg line (x-axis-label style).
        # For >=6 FPs: angled to fit in narrower columns.
        x_center = 0.5 * (x_min + x_max)
        if n_subplots < 6:
            ax_sub.text(
                x_center,
                -4.0,
                target_fp,
                transform=ax_sub.transData,
                ha="center",
                va="top",
                rotation=0.0,
                fontsize=label_fontsize,
                color="black",
                bbox=dict(
                    facecolor=mcolors.to_rgba(target_color, alpha=highlight_alpha),
                    edgecolor=mcolors.to_rgba(target_color, alpha=1.0),
                    linewidth=outline_lw,
                    boxstyle=f"round,pad={label_pad}",
                ),
                clip_on=False,
            )
        else:
            ax_sub.text(
                0.5,
                0.0,
                target_fp,
                transform=ax_sub.transAxes,
                ha="center",
                va="bottom",
                rotation=-15,
                fontsize=label_fontsize,
                color="black",
                bbox=dict(
                    facecolor=mcolors.to_rgba(target_color, alpha=highlight_alpha),
                    edgecolor=mcolors.to_rgba(target_color, alpha=1.0),
                    linewidth=outline_lw,
                    boxstyle=f"round,pad={label_pad}",
                ),
                clip_on=False,
            )

        # Highlight nearest neighbor label(s), if any
        # Always center horizontally; if two labels are close, draw the closest last so it sits on top.
        if len(nearest_others) > 0:
            x_center = 0.5 * (x_min + x_max)
            # nearest_others is sorted by angle ascending; reverse draw order => closest label is on top.
            for ang, fp_name in reversed(nearest_others):
                c = cfg.fluorophore_colors.get(fp_name, "#808080")
                add_highlight_text_data(
                    ax_sub=ax_sub,
                    x=x_center,
                    y=min(ang + 2.0, 88.0),
                    txt=fp_name,
                    color=c,
                    rotation=0.0,
                    ha="center",
                    va="bottom",
                )

        if ax_i == 0:
            if style == "manuscript":
                ax_sub.set_ylabel("Angle (deg)", fontsize=11 * font_scale)
            else:
                ax_sub.set_ylabel("Angle (degrees)", fontsize=12 * font_scale)
        else:
            ax_sub.set_ylabel("")

    plt.tight_layout()
    return fig, axes


def F_two_channel_scatterplot_with_vectors_and_cones_plot(params_dict, ax=None):
    """
    Generic function F: 2-channel 2D scatterplot with vectors + classification wedges.

    This is a thin config-driven wrapper around the legacy implementation:
      `src.figure1.subpanel_5`

    Parameters
    ----------
    params_dict : dict
        Expected keys (passed through to the legacy `subpanel_5` as `row_dict`):
          - "name" (optional)
          - "Fluorophores"
          - "Channel 1": {"Excitation wavelength": int, "emission filter": str}
          - "Channel 2": {"Excitation wavelength": int, "emission filter": str}
        Optional wrapper keys:
          - "data_dir": directory containing the aligned image structure
          - "figsize": (w, h) for when ax is not provided
    ax : matplotlib.axes.Axes, optional
        If provided, the plot will be drawn into this axes.
    """
    from src.figure1 import subpanel_5 as legacy_subpanel_5

    figsize = params_dict.get("figsize", (6, 6))
    data_dir = params_dict.get(
        "data_dir",
        "data/fig1_fig2_1color_3mice_singleplane_june20250619",
    )
    point_color_mode = params_dict.get("point_color_mode", "by_fp")
    shared_data = params_dict.get("shared_data", None)
    use_shared_data_for_points = bool(params_dict.get("use_shared_data_for_points", True))

    # Legacy function expects `row_dict`; it ignores unknown keys safely,
    # but we avoid passing wrapper-only keys if present.
    row_dict = dict(params_dict)
    row_dict.pop("figsize", None)
    row_dict.pop("data_dir", None)
    row_dict.pop("shared_data", None)
    row_dict.pop("use_shared_data_for_points", None)

    if ax is None:
        fig, ax_local = plt.subplots(figsize=figsize)
    else:
        ax_local = ax
        fig = ax_local.figure

    # Compute predicted vectors using the unified predictor helper so we don't
    # rely on legacy `compute_predicted_channel_signals()` in `figure1.subpanel_5`.
    fluorophores = row_dict.get("Fluorophores", [])
    channel_keys = []
    channel_configs = []
    for ck in ["Channel 1", "Channel 2"]:
        if ck in row_dict:
            channel_keys.append(ck)
            channel_configs.append(row_dict[ck])

    predicted_raw = _compute_predicted_signals_per_fp(
        fluorophore_names=fluorophores,
        channel_configs=channel_configs,
        load_spectra_func=None,
        smoothing_std=5,
    )
    predicted_signals = {
        fp: {channel_keys[i]: predicted_raw[fp][i] for i in range(len(channel_keys))}
        for fp in fluorophores
    }

    if shared_data is not None and use_shared_data_for_points:
        # Build a preselected point payload for legacy subpanel_5.
        # We map wrapper Channel configs -> the corresponding shared_data channels
        # (ch1_valid/ch2_valid/ch3_valid) by matching excitation wavelength + emission filter.
        def _pick_shared_channel_values(target_channel_cfg):
            excitation_wl = target_channel_cfg.get("Excitation wavelength")
            emission_filter = target_channel_cfg.get("emission filter")
            if not excitation_wl:
                raise ValueError("Channel config missing 'Excitation wavelength'")

            # `shared_data` emission filters are usually numeric ranges
            # like [590, 620], while wrapper configs use names like "Red".
            # Convert wrapper emission-filter names -> numeric ranges using fig2's mapping.
            try:
                from src.figure2 import FILTER_RANGE_TO_NAME as _F2_FILTER_RANGE_TO_NAME

                name_to_range = {
                    name: list(filter_range) for filter_range, name in _F2_FILTER_RANGE_TO_NAME.items()
                }
            except Exception:
                name_to_range = {}

            def _norm_filter_value(val):
                # Normalize shared_data emission filter values for comparison.
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    return [float(val[0]), float(val[1])]
                return val

            desired_filter_norm = emission_filter
            if isinstance(emission_filter, str) and emission_filter in name_to_range:
                desired_filter_norm = name_to_range[emission_filter]

            for short in ["ch1", "ch2", "ch3"]:
                sc = shared_data.get(f"{short}_config", {})
                if not isinstance(sc, dict):
                    continue
                sc_exc = sc.get("Excitation wavelength")
                sc_filter = _norm_filter_value(sc.get("emission filter"))
                if sc_exc == excitation_wl and sc_filter == _norm_filter_value(desired_filter_norm):
                    return np.asarray(shared_data.get(f"{short}_valid", []), dtype=float)

            # Fallback (should rarely be needed): match only by excitation wavelength.
            for short in ["ch1", "ch2", "ch3"]:
                sc = shared_data.get(f"{short}_config", {})
                if not isinstance(sc, dict):
                    continue
                if sc.get("Excitation wavelength") == excitation_wl:
                    return np.asarray(shared_data.get(f"{short}_valid", []), dtype=float)

            raise ValueError(
                "Could not map wrapper channel to shared_data channels. "
                f"Channel cfg={target_channel_cfg} (shared channel cfg keys were: ch1/ch2/ch3 configs)"
            )

        ch1_cfg = row_dict.get("Channel 1", {})
        ch2_cfg = row_dict.get("Channel 2", {})
        ch1_vals = _pick_shared_channel_values(ch1_cfg)
        ch2_vals = _pick_shared_channel_values(ch2_cfg)

        fp_labels_plot = np.asarray(shared_data.get("pixel_labels", []), dtype=object)
        if ch1_vals.shape != ch2_vals.shape or ch1_vals.shape[0] != fp_labels_plot.shape[0]:
            raise ValueError(
                "shared_data channel arrays do not align. "
                f"ch1_vals={ch1_vals.shape}, ch2_vals={ch2_vals.shape}, fp_labels_plot={fp_labels_plot.shape}"
            )

        preselected_points = {
            "ch1_plot": ch1_vals,
            "ch2_plot": ch2_vals,
            "fp_labels_plot": fp_labels_plot,
            "max_value": float(shared_data.get("max_value", 3000.0)),
        }

        fig_out, ax_out = legacy_subpanel_5(
            row_dict,
            ax=ax_local,
            data_dir=data_dir,
            predicted_signals=predicted_signals,
            preselected_points=preselected_points,
        )
    else:
        fig_out, ax_out = legacy_subpanel_5(
            row_dict,
            ax=ax_local,
            data_dir=data_dir,
            predicted_signals=predicted_signals,
        )

    # Standardized axis labels from channel configs (both presentation/manuscript).
    ch1_cfg = row_dict.get("Channel 1", {})
    ch2_cfg = row_dict.get("Channel 2", {})
    ch1_wl = ch1_cfg.get("Excitation wavelength", "")
    ch2_wl = ch2_cfg.get("Excitation wavelength", "")
    ch1_filt = ch1_cfg.get("emission filter", "")
    ch2_filt = ch2_cfg.get("emission filter", "")
    ax_out.set_xlabel(f"{ch1_wl}nm, {ch1_filt} filter")
    ax_out.set_ylabel(f"{ch2_wl}nm, {ch2_filt} filter")

    # Optional post-processing: force scatter points to a single gray color.
    # Useful for multiplexed layouts where we don't have per-fluorophore labeling.
    if point_color_mode == "all_gray":
        gray = "#444444"
        for coll in getattr(ax_out, "collections", []):
            try:
                coll.set_facecolor(gray)
                coll.set_edgecolor(gray)
            except Exception:
                pass

    return fig_out, ax_out


def H_three_channel_3d_scatterplot_with_vectors_and_cones_plot(params_dict, ax=None):
    """
    Generic function H: 3-channel 3D scatterplot (legacy figure2.subpanel_5).
    """
    from src.figure2 import subpanel_5 as legacy_subpanel_5

    figsize = params_dict.get("figsize", (10, 8))
    data_dir = params_dict.get("data_dir", "data/fig2_3color_inh_spatial_control_2p3_10072025")
    single_fp_data_dir = params_dict.get(
        "single_fp_data_dir", "data/fig1_fig2_1color_3mice_singleplane_june20250619"
    )
    row_dict = params_dict.get("row_dict", None)
    shared_data = params_dict.get("shared_data", None)
    if shared_data is None:
        raise ValueError(
            "Function H plotting requires precomputed `shared_data` "
            "(classification must be handled in new_figure_1.py)."
        )

    if ax is None:
        fig_local = plt.figure(figsize=figsize)
        ax_local = fig_local.add_subplot(111, projection="3d")
    else:
        ax_local = ax
        fig_local = ax_local.figure

    fig_out, ax_out = legacy_subpanel_5(
        row_dict=row_dict,
        ax=ax_local,
        data_dir=data_dir,
        single_fp_data_dir=single_fp_data_dir,
        shared_data=shared_data,
    )

    # Standardized channel labels from the active channel configs.
    # Prefer row_dict values when provided; fallback to shared_data channel configs.
    def _resolve_channel_label(channel_key, shared_key):
        cfg_dict = {}
        if isinstance(row_dict, dict):
            cfg_dict = row_dict.get(channel_key, {}) or {}
        if not cfg_dict:
            cfg_dict = shared_data.get(shared_key, {}) if isinstance(shared_data, dict) else {}
        wl = cfg_dict.get("Excitation wavelength", "")
        filt = cfg_dict.get("emission filter", "")
        if isinstance(filt, (list, tuple)) and len(filt) == 2:
            # Convert range-like filters to names when possible.
            try:
                from src.figure2 import FILTER_RANGE_TO_NAME as _F2_FILTER_RANGE_TO_NAME
                filt = _F2_FILTER_RANGE_TO_NAME.get((int(filt[0]), int(filt[1])), filt)
            except Exception:
                pass
        return f"{wl}nm, {filt} filter"

    ax_out.set_xlabel(_resolve_channel_label("Channel 1", "ch1_config"))
    ax_out.set_ylabel(_resolve_channel_label("Channel 2", "ch2_config"))
    ax_out.set_zlabel(_resolve_channel_label("Channel 3", "ch3_config"))

    style = params_dict.get("style", "presentation")
    if style == "manuscript":
        # Manuscript: no legend/title, simplified axis labels, no tick labels.
        if getattr(ax_out, "legend_", None) is not None:
            ax_out.legend_.remove()
        ax_out.set_title("")
        # Bring labels closer since tick labels are hidden.
        ax_out.xaxis.labelpad = -3
        ax_out.yaxis.labelpad = -3
        ax_out.zaxis.labelpad = -3
        ax_out.set_xticklabels([])
        ax_out.set_yticklabels([])
        ax_out.set_zticklabels([])
    else:
        # Presentation: keep channel-specific labels from legacy plot, but cap tick label count.
        ax_out.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_out.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_out.zaxis.set_major_locator(MaxNLocator(nbins=4))

    # Always provide a minimal legend key for point semantics.
    pixel_handle = Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markersize=4,
        markerfacecolor="#666666",
        markeredgecolor="#666666",
        alpha=0.8,
        label="Pixels",
    )
    ax_out.legend(handles=[pixel_handle], loc="upper right", fontsize=8, frameon=True)

    return fig_out, ax_out


def I_three_channel_triangle_projection_plot(params_dict, ax=None):
    """
    Generic function I: 3-channel triangle projection (legacy figure2.subpanel_6).
    """
    from src.figure2 import subpanel_6 as legacy_subpanel_6

    figsize = params_dict.get("figsize", (8, 8))
    data_dir = params_dict.get("data_dir", "data/fig2_3color_inh_spatial_control_2p3_10072025")
    single_fp_data_dir = params_dict.get(
        "single_fp_data_dir", "data/fig1_fig2_1color_3mice_singleplane_june20250619"
    )
    row_dict = params_dict.get("row_dict", None)
    shared_data = params_dict.get("shared_data", None)
    if shared_data is None:
        raise ValueError(
            "Function I plotting requires precomputed `shared_data` "
            "(classification must be handled in new_figure_1.py)."
        )

    if ax is None:
        fig_local, ax_local = plt.subplots(figsize=figsize)
    else:
        ax_local = ax
        fig_local = ax_local.figure

    projection_method = params_dict.get("projection_method", "legacy_rotated")

    if projection_method in {"ternary_basis", "simplex_basis", "nonlinear_star"}:
        # Alternate method: change-of-basis to FP weights, then map to a regular N-gon.
        # - ternary_basis: 3-vector special case (triangle)
        # - simplex_basis: generic N-vector mode (square/diamond for 4, pentagon for 5, ...)
        ax_local.clear()
        ax_out = ax_local
        fig_out = fig_local

        fluorophores = list(shared_data.get("fluorophores", []))
        if len(fluorophores) < 3:
            raise ValueError("simplex/ternary basis projection requires at least 3 fluorophores.")

        # ternary mode intentionally keeps 3 vectors; simplex/star modes keep all available vectors.
        fp_names = fluorophores[:3] if projection_method == "ternary_basis" else fluorophores

        data_vectors_3d = shared_data.get("data_vectors_3d", {})
        if any(fp not in data_vectors_3d for fp in fp_names):
            raise ValueError("shared_data.data_vectors_3d is missing one or more fluorophore vectors.")

        # Basis in measurement space (dim x n_vectors, columns are FP vectors).
        V = np.column_stack([np.asarray(data_vectors_3d[fp], dtype=float) for fp in fp_names])
        dim = V.shape[0]
        n_vectors = V.shape[1]
        V_pinv = np.linalg.pinv(V)

        # Build point matrix using ch1_valid..ch{dim}_valid if available.
        point_cols = []
        for i in range(dim):
            key = f"ch{i+1}_valid"
            col = np.asarray(shared_data.get(key, []), dtype=float)
            point_cols.append(col)
        if len(point_cols) == 0:
            raise ValueError("No channel vectors found in shared_data (expected ch1_valid, ch2_valid, ...).")
        lengths = [len(c) for c in point_cols]
        if len(set(lengths)) != 1:
            raise ValueError("shared_data channel arrays have mismatched lengths.")
        p = np.column_stack(point_cols)

        labels = np.asarray(shared_data.get("pixel_labels", []), dtype=object)
        if p.shape[0] != labels.shape[0]:
            raise ValueError("shared_data pixel arrays and labels have mismatched lengths.")

        # Solve for FP weights per point: w = pinv(V) @ p
        w = (V_pinv @ p.T).T
        if bool(params_dict.get("enforce_nonnegative_weights", True)):
            w = np.clip(w, 0.0, None)

        if projection_method == "nonlinear_star":
            # Direction + magnitude decoupling.
            # - Radius: sum of raw (nonnegative) weights.
            # - Direction: weighted center-of-mass angle on a circle (equally spaced base angles).
            max_points = int(params_dict.get("max_plot_points", 60000))
            if len(w) > max_points > 0:
                idx = np.random.choice(len(w), size=max_points, replace=False)
                w = w[idx]
                labels = labels[idx]

            radius = np.sum(w, axis=1)

            star_start_angle_deg = float(params_dict.get("star_start_angle_deg", 90.0))
            phis = np.radians(star_start_angle_deg + np.arange(n_vectors) * (360.0 / n_vectors))
            cosv = np.cos(phis)
            sinv = np.sin(phis)

            # Weighted center-of-mass direction
            dir_x = w @ cosv
            dir_y = w @ sinv

            theta = np.arctan2(dir_y, dir_x)
            x_coords = radius * np.cos(theta)
            y_coords = radius * np.sin(theta)

            fp_colors = {fp: cfg.fluorophore_colors.get(fp, "#808080") for fp in fp_names}
            point_color_mode = params_dict.get("point_color_mode", "by_label")
            if point_color_mode == "all_gray":
                ax_out.scatter(x_coords, y_coords, s=1, alpha=0.28, c="#444444", zorder=2)
            else:
                for fp in fp_names:
                    mask = labels == fp
                    if np.any(mask):
                        ax_out.scatter(
                            x_coords[mask], y_coords[mask], s=1, alpha=0.28, c=fp_colors[fp], label=fp, zorder=2
                        )
                mask_unclassified = labels == None  # noqa: E711
                if np.any(mask_unclassified):
                    ax_out.scatter(
                        x_coords[mask_unclassified],
                        y_coords[mask_unclassified],
                        s=1,
                        alpha=0.12,
                        c="gray",
                        label="unclassified",
                        zorder=1,
                    )

            # Rays for each base vector direction
            ray_end = params_dict.get("star_ray_end", None)
            if ray_end is None:
                pos_r = radius[radius > 0]
                if len(pos_r) > 0:
                    r_lim = float(np.percentile(pos_r, float(params_dict.get("star_radius_plot_percentile", 99.0))))
                    ray_end = 0.95 * r_lim
                else:
                    ray_end = 1.0
            ray_end = float(ray_end)

            for i, fp in enumerate(fp_names):
                phi = phis[i]
                ax_out.plot([0.0, ray_end * np.cos(phi)], [0.0, ray_end * np.sin(phi)],
                            color=fp_colors[fp], linewidth=2.8, alpha=0.85, zorder=4)

            # Axis limits based on high-percentile radius to avoid hard clipping
            pos_r = radius[radius > 0]
            if len(pos_r) > 0:
                r_lim = float(np.percentile(pos_r, float(params_dict.get("star_radius_plot_percentile", 99.0))))
            else:
                r_lim = 1.0
            lim = float(params_dict.get("plot_limit", 1.2 * r_lim))
            ax_out.set_xlim(-lim, lim)
            ax_out.set_ylim(-lim, lim)
            ax_out.set_aspect("equal", adjustable="box")
            ax_out.set_title("Subpanel 6: Direction + magnitude (non-linear star)")
        else:
            # Normalize to simplex/polygon
            w_sum = np.sum(w, axis=1, keepdims=True)
            valid = w_sum[:, 0] > 0
            w_norm = np.zeros_like(w)
            w_norm[valid] = w[valid] / w_sum[valid]

            # Optional plotting subsample
            max_points = int(params_dict.get("max_plot_points", 60000))
            if len(w_norm) > max_points > 0:
                idx = np.random.choice(len(w_norm), size=max_points, replace=False)
                w_norm = w_norm[idx]
                labels = labels[idx]

            # Regular N-gon vertices (centroid at origin), starting at 90 deg (top).
            theta0 = float(params_dict.get("polygon_start_angle_deg", 90.0))
            radius = float(params_dict.get("polygon_radius", 1.0))
            angles = np.radians(theta0 + np.arange(n_vectors) * (360.0 / n_vectors))
            vertices = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

            xy = w_norm[:, :n_vectors] @ vertices
            x_coords = xy[:, 0]
            y_coords = xy[:, 1]

            # Points
            fp_colors = {fp: cfg.fluorophore_colors.get(fp, "#808080") for fp in fp_names}
            point_color_mode = params_dict.get("point_color_mode", "by_label")
            if point_color_mode == "all_gray":
                ax_out.scatter(x_coords, y_coords, s=1, alpha=0.28, c="#444444", zorder=2)
            else:
                for fp in fp_names:
                    mask = labels == fp
                    if np.any(mask):
                        ax_out.scatter(
                            x_coords[mask], y_coords[mask], s=1, alpha=0.28, c=fp_colors[fp], label=fp, zorder=2
                        )
                mask_unclassified = labels == None  # noqa: E711
                if np.any(mask_unclassified):
                    ax_out.scatter(
                        x_coords[mask_unclassified],
                        y_coords[mask_unclassified],
                        s=1,
                        alpha=0.12,
                        c="gray",
                        label="unclassified",
                        zorder=1,
                    )

            # Polygon outline
            poly_closed = np.vstack([vertices, vertices[0]])
            ax_out.plot(poly_closed[:, 0], poly_closed[:, 1], color="black", linewidth=1.5, alpha=0.8, zorder=3)

            # Vectors from centroid (origin) to each corner
            for i, fp in enumerate(fp_names):
                v = vertices[i]
                ax_out.plot([0.0, v[0]], [0.0, v[1]], color=fp_colors[fp], linewidth=2.8, alpha=0.85, zorder=4)

            lim = float(params_dict.get("plot_limit", 1.2 * radius))
            ax_out.set_xlim(-lim, lim)
            ax_out.set_ylim(-lim, lim)
            ax_out.set_aspect("equal", adjustable="box")
            ax_out.set_title(f"Subpanel 6: Basis projection ({n_vectors}-vector polygon)")
    else:
        fig_out, ax_out = legacy_subpanel_6(
            row_dict=row_dict,
            ax=ax_local,
            data_dir=data_dir,
            single_fp_data_dir=single_fp_data_dir,
            shared_data=shared_data,
        )

        # Optional rigid 2D rotation: rotate all projected geometry so one vector
        # (closest to target axis) aligns exactly with that axis.
        if bool(params_dict.get("align_anchor_vector", True)):
            target_angle_deg = float(params_dict.get("anchor_axis_angle_deg", 90.0))  # 90 deg = "up"

            # Find candidate vectors from origin (the projected FP vectors are lines from origin).
            candidate_lines = []
            for line in list(getattr(ax_out, "lines", [])):
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
                if len(x) == 2 and len(y) == 2 and abs(x[0]) < 1e-9 and abs(y[0]) < 1e-9:
                    v = np.array([x[1], y[1]], dtype=float)
                    if float(np.linalg.norm(v)) > 0:
                        candidate_lines.append((line, v))

            if len(candidate_lines) > 0:
                target_rad = np.radians(target_angle_deg)

                def _wrap_pi(a):
                    return (a + np.pi) % (2 * np.pi) - np.pi

                # Pick anchor vector with smallest angular difference to target axis.
                _, anchor_vec = min(
                    candidate_lines,
                    key=lambda lv: abs(_wrap_pi(np.arctan2(lv[1][1], lv[1][0]) - target_rad)),
                )
                anchor_angle = np.arctan2(anchor_vec[1], anchor_vec[0])
                delta = target_rad - anchor_angle

                c = float(np.cos(delta))
                s = float(np.sin(delta))
                rot = np.array([[c, -s], [s, c]], dtype=float)

                # Rotate line geometry (vectors + triangle edges + any guide lines).
                for line in list(getattr(ax_out, "lines", [])):
                    x = np.asarray(line.get_xdata(), dtype=float)
                    y = np.asarray(line.get_ydata(), dtype=float)
                    if len(x) != len(y) or len(x) == 0:
                        continue
                    pts = np.column_stack([x, y])
                    pts_rot = pts @ rot.T
                    line.set_xdata(pts_rot[:, 0])
                    line.set_ydata(pts_rot[:, 1])

                # Rotate scatter collections (projected points / zones).
                for coll in list(getattr(ax_out, "collections", [])):
                    try:
                        offsets = coll.get_offsets()
                    except Exception:
                        offsets = None
                    if offsets is None:
                        continue
                    off = np.asarray(offsets, dtype=float)
                    if off.ndim != 2 or off.shape[1] < 2 or off.shape[0] == 0:
                        continue
                    rotated_xy = off[:, :2] @ rot.T
                    if off.shape[1] > 2:
                        off_new = off.copy()
                        off_new[:, :2] = rotated_xy
                        coll.set_offsets(off_new)
                    else:
                        coll.set_offsets(rotated_xy)

    # Overlay theoretical predicted vectors (dashed) in the same 2D triangle basis.
    # This keeps the vector source explicit: solid = data, dashed = prediction.
    try:
        # Resolve channel configs from row_dict/shared_data and normalize filter names.
        from src.figure2 import FILTER_RANGE_TO_NAME as _F2_FILTER_RANGE_TO_NAME

        channel_cfgs = []
        for i in [1, 2, 3]:
            cfg_i = {}
            if isinstance(row_dict, dict):
                cfg_i = row_dict.get(f"Channel {i}", {}) or {}
            if not cfg_i and isinstance(shared_data, dict):
                cfg_i = shared_data.get(f"ch{i}_config", {}) or {}
            if not isinstance(cfg_i, dict) or not cfg_i:
                continue
            filt = cfg_i.get("emission filter")
            if isinstance(filt, (list, tuple)) and len(filt) == 2:
                filt_name = _F2_FILTER_RANGE_TO_NAME.get((int(filt[0]), int(filt[1])), filt)
            else:
                filt_name = filt
            channel_cfgs.append(
                {
                    "Excitation wavelength": cfg_i.get("Excitation wavelength"),
                    "emission filter": filt_name,
                    "power_mw": cfg_i.get("power_mw", cfg_i.get("power")),
                }
            )

        fp_names_pred = list(shared_data.get("fluorophores", []))
        if len(channel_cfgs) >= 3 and len(fp_names_pred) > 0:
            predicted = _compute_predicted_signals_per_fp(
                fluorophore_names=fp_names_pred,
                channel_configs=channel_cfgs[:3],
                smoothing_std=5,
            )

            cos_30 = float(np.cos(np.radians(30.0)))
            sin_30 = float(np.sin(np.radians(30.0)))
            transform_matrix = np.array(
                [
                    [0.0, cos_30, -cos_30],
                    [1.0, -sin_30, -sin_30],
                ],
                dtype=float,
            )

            # Determine a reasonable arrow length from existing origin vectors.
            origin_vec_lengths = []
            for line in list(getattr(ax_out, "lines", [])):
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
                if len(x) == 2 and len(y) == 2 and abs(x[0]) < 1e-9 and abs(y[0]) < 1e-9:
                    origin_vec_lengths.append(float(np.hypot(x[1], y[1])))
            arrow_len = float(np.median(origin_vec_lengths)) if len(origin_vec_lengths) > 0 else 1.0

            from matplotlib.patches import FancyArrowPatch
            for fp_name in fp_names_pred:
                vec3 = np.asarray(predicted.get(fp_name, []), dtype=float)
                if vec3.size < 3:
                    continue
                norm3 = float(np.linalg.norm(vec3))
                if norm3 <= 0:
                    continue
                vec3 = vec3 / norm3
                vec2 = transform_matrix @ vec3[:3]
                norm2 = float(np.linalg.norm(vec2))
                if norm2 <= 0:
                    continue
                vec2 = (vec2 / norm2) * arrow_len
                c = cfg.fluorophore_colors.get(fp_name, "#808080")
                pred_arrow = FancyArrowPatch(
                    (0.0, 0.0),
                    (float(vec2[0]), float(vec2[1])),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linestyle="--",
                    linewidth=1.8,
                    color=c,
                    alpha=0.9,
                    zorder=5,
                )
                ax_out.add_patch(pred_arrow)
    except Exception:
        # Keep plotting robust if predictor/metadata resolution fails.
        pass

    # Add arrowheads to any data vectors that are currently lines from origin.
    try:
        from matplotlib.patches import FancyArrowPatch
        for line in list(getattr(ax_out, "lines", [])):
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            if len(x) == 2 and len(y) == 2 and abs(x[0]) < 1e-9 and abs(y[0]) < 1e-9:
                data_arrow = FancyArrowPatch(
                    (0.0, 0.0),
                    (float(x[1]), float(y[1])),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linestyle="-",
                    linewidth=2.2,
                    color=line.get_color(),
                    alpha=0.95,
                    zorder=6,
                )
                ax_out.add_patch(data_arrow)
    except Exception:
        pass

    style = params_dict.get("style", "presentation")
    if style == "manuscript":
        ax_out.set_title("")

    # Remove axis framing and replace with concentric-circle crosshair guide.
    x0, x1 = ax_out.get_xlim()
    y0, y1 = ax_out.get_ylim()
    cx = 0.0
    cy = 0.0
    r_max = 0.95 * max(abs(x0 - cx), abs(x1 - cx), abs(y0 - cy), abs(y1 - cy))
    radii = np.linspace(r_max * 0.25, r_max, 4)

    for rr in radii:
        ax_out.add_patch(
            Circle((cx, cy), rr, fill=False, linewidth=1.0, edgecolor="#B0B0B0", alpha=0.6, zorder=0)
        )
    ax_out.axhline(cy, color="#B0B0B0", linewidth=1.0, alpha=0.6, zorder=0)
    ax_out.axvline(cx, color="#B0B0B0", linewidth=1.0, alpha=0.6, zorder=0)

    ax_out.set_xticks([])
    ax_out.set_yticks([])
    ax_out.set_xlabel("")
    ax_out.set_ylabel("")
    for spine in ax_out.spines.values():
        spine.set_visible(False)
    ax_out.set_aspect("equal", adjustable="box")

    # Explicit legend for point semantics + vector source.
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=4,
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            alpha=0.8,
            label="Pixels",
        ),
        Line2D([0], [0], color="black", linewidth=2.2, linestyle="-", label="Vector from data"),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle="--", label="Theoretical prediction"),
    ]
    ax_out.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)

    return fig_out, ax_out


def J_angle_histogram_plot(params_dict, ax=None):
    """
    Generic function J: stacked angle histogram (balanced pixel subset).

    This is a config-driven port of `src.figure1.subpanel_7`, but it uses the
    pre-balanced `shared_data` pixel subset from `src/new_figure_1.py`.
    """
    shared_data = params_dict.get("shared_data", None)
    if shared_data is None:
        raise ValueError(
            "Function J plotting requires precomputed `shared_data` "
            "(classification must be handled in new_figure_1.py)."
        )

    style = params_dict.get("style", "presentation")
    font_scale = float(params_dict.get("font_scale", 1.0))
    show_legend = bool(params_dict.get("show_legend", True))

    fluorophores = params_dict.get("Fluorophores", list(shared_data.get("fluorophores", [])))
    if not fluorophores:
        raise ValueError("params_dict must contain non-empty 'Fluorophores'.")

    ch1_cfg = params_dict.get("Channel 1", {})
    ch2_cfg = params_dict.get("Channel 2", {})
    if not ch1_cfg or not ch2_cfg:
        raise ValueError("params_dict must contain 'Channel 1' and 'Channel 2' configs.")

    # Map wrapper emission-filter names ("Red", "FarRed") -> shared_data numeric ranges.
    from src.figure2 import FILTER_RANGE_TO_NAME as _F2_FILTER_RANGE_TO_NAME

    name_to_range = {name: list(filter_range) for filter_range, name in _F2_FILTER_RANGE_TO_NAME.items()}

    def _norm_filter_value(val):
        if isinstance(val, str) and val in name_to_range:
            return name_to_range[val]
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return [float(val[0]), float(val[1])]
        return val

    def _pick_shared_channel_values(target_channel_cfg):
        excitation_wl = target_channel_cfg.get("Excitation wavelength")
        emission_filter = target_channel_cfg.get("emission filter")
        if excitation_wl is None:
            raise ValueError("Channel config missing 'Excitation wavelength'")

        desired_filter_norm = _norm_filter_value(emission_filter)

        for short in ["ch1", "ch2", "ch3"]:
            sc = shared_data.get(f"{short}_config", {})
            if not isinstance(sc, dict):
                continue
            if sc.get("Excitation wavelength") == excitation_wl and _norm_filter_value(sc.get("emission filter")) == desired_filter_norm:
                return np.asarray(shared_data.get(f"{short}_valid", []), dtype=float)

        raise ValueError(
            "Could not map wrapper channel to shared_data channel. "
            f"Channel cfg={target_channel_cfg}"
        )

    ch1_vals = _pick_shared_channel_values(ch1_cfg)
    ch2_vals = _pick_shared_channel_values(ch2_cfg)
    labels_valid = np.asarray(shared_data.get("pixel_labels", []), dtype=object)

    if ch1_vals.shape != ch2_vals.shape or ch1_vals.shape[0] != labels_valid.shape[0]:
        raise ValueError(
            "shared_data arrays do not align for J histogram. "
            f"ch1_vals={ch1_vals.shape}, ch2_vals={ch2_vals.shape}, labels_valid={labels_valid.shape}"
        )

    # Histogram settings
    bin_size = int(getattr(cfg, "angle_histogram_bin_size_degrees", 1))
    bins = np.arange(0, 90 + bin_size, bin_size)
    fp_colors = {fp: cfg.fluorophore_colors.get(fp, "#808080") for fp in fluorophores}

    # Compute data vectors and per-reference "kept" threshold.
    data_vectors = {}
    half_angles = {}
    zone_min_distance = getattr(cfg, "classification_zone_min_distance", 500)

    for fp_name in fluorophores:
        mask_fp = labels_valid == fp_name
        if np.any(mask_fp):
            data_vectors[fp_name] = compute_data_vector(ch1_vals[mask_fp], ch2_vals[mask_fp])
        else:
            # Keep vector well-defined so angle computations won't crash.
            data_vectors[fp_name] = np.array([1.0, 0.0], dtype=float)

        try:
            half_angles[fp_name] = compute_classification_zone(
                ch1_vals,
                ch2_vals,
                labels_valid,
                fp_name,
                data_vectors[fp_name],
                percentile=cfg.classification_zone_percentile,
                min_distance=zone_min_distance,
            )
        except Exception:
            half_angles[fp_name] = None

    # Plot: stacked axes, like legacy `subpanel_7`.
    n_fps = len(fluorophores)
    if ax is None:
        figsize = params_dict.get("figsize", (10, 2.6 * n_fps))
        fig, axes = plt.subplots(n_fps, 1, figsize=figsize, sharex=True)
        if n_fps == 1:
            axes = np.array([axes])
    else:
        if n_fps != 1:
            raise ValueError("When providing `ax`, J histogram currently supports only 1 fluorophore.")
        axes = np.array([ax])
        fig = ax.figure

    max_y = 0
    for ax_sub, ref_fp in zip(axes, fluorophores):
        ref_vec = data_vectors[ref_fp]
        offsets_deg_all = compute_angle_to_vector(ch1_vals, ch2_vals, ref_vec)

        # Shade "kept" region (0..threshold) for this reference vector.
        thresh = half_angles.get(ref_fp)
        if thresh is not None:
            ax_sub.axvspan(0, thresh, color=fp_colors.get(ref_fp, "#808080"), alpha=0.15, zorder=0)
            ax_sub.axvline(
                thresh,
                color=fp_colors.get(ref_fp, "#808080"),
                linestyle=":",
                linewidth=2,
                alpha=0.9,
                zorder=3,
            )

        # Overlaid histograms by true label.
        for true_fp in fluorophores:
            mask = labels_valid == true_fp
            fp_offsets = offsets_deg_all[mask]
            fp_offsets = fp_offsets[~np.isnan(fp_offsets)]
            if fp_offsets.size == 0:
                continue

            hist, _ = np.histogram(fp_offsets, bins=bins)
            max_y = max(max_y, int(np.max(hist)))

            ax_sub.bar(
                bins[:-1],
                hist,
                width=bin_size,
                color=fp_colors.get(true_fp, "#808080"),
                alpha=0.55,
                edgecolor="none",
                label=true_fp if show_legend and ref_fp == fluorophores[0] else None,
                zorder=1,
            )

        y_fs = 10 * font_scale if style == "manuscript" else 10 * font_scale
        ax_sub.set_ylabel(f"to {ref_fp}\ncount", fontsize=y_fs)
        ax_sub.grid(True, alpha=0.25, axis="y")
        ax_sub.spines["top"].set_visible(False)
        ax_sub.spines["right"].set_visible(False)

    for ax_sub in axes:
        ax_sub.set_xlim(0, 90)
        if max_y > 0:
            ax_sub.set_ylim(0, max_y * 1.15)

    axes[-1].set_xlabel("Angle to reference vector (degrees)", fontsize=12 * font_scale)

    if show_legend and len(axes) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        pixel_handle = Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=4,
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            alpha=0.8,
            label="Pixels",
        )
        handles = list(handles) + [pixel_handle]
        labels = list(labels) + ["Pixels"]
        axes[0].legend(handles, labels, loc="upper right", fontsize=9 * font_scale, frameon=True)

    # Manuscript: drop the title area by keeping it minimal (no explicit title here).
    plt.tight_layout()
    return fig, axes

