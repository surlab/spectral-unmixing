"""
Figure 5 generation for spectral unmixing methods paper.

This module generates Figure 5, which shows spectral unmixing for many fluorophores
across multiple excitation wavelengths and emission filters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from src import config as cfg
from src.figure1 import (
    load_2p_spectra,
    plot_2p_excitation_spectra,
    plot_1p_emission_spectra,
    apply_smoothing_to_spectrum,
    load_filter_transmission
)
from src.figure_scatterplot_helpers import (
    create_ratio_histogram_base,
    load_2p_spectra_flexible
)

# Figure 5 row dictionary
FIG_5_ROW_DICT = {
    "name": "fig_5",
    "Fluorophores": ["EBFP", "tagBFP", "ECFP", "GCamp Ca+", "GCampCa-", "LSSmOrange", "TdTomato", "mCherry", "LSSmKAte", "mNeptune"],
    "Excitation wavelengths": [750, 800, 870, 1040, 1180, 1240],
    "emission filters": [[400, 440], [445, 475], [475, 495], [500, 550], [550, 580], [590, 620], [645, 695]]
}

# Demo figure row dictionary (with dichroics)
# Based on demo_figure specification:
# FPs: gcamp ca-, gcamp ca+, yfp, tdtomato, mscarlet, mcherry, mneptune
# Excitation wavelengths: [800, 920, 1040, 1080, 1140, 1180]
# Filters: narrow green, yellow, orange, red, far red
# Note: When channels are created, they will pair filters with dichroics
# (e.g., Yellow filter with 514-Transmitted, NarrowGreen filter with 514-Reflected)
DEMO_FIGURE_ROW_DICT = {
    "name": "demo_figure",
    "Fluorophores": ["GCampCa-", "GCamp Ca+", "YFP", "TdTomato", "mScarlet", "mCherry", "mNeptune"],
    "Excitation wavelengths": [800, 920, 1040, 1080, 1140, 1180],
    "emission filters": ["NarrowGreen", "Yellow", "Orange", "Red", "FarRed"]
}

# Fluorophore colors for Figure 5
FIG_5_FP_COLORS = {
    "EBFP": "#00008B",      # dark blue
    "tagBFP": "#4169E1",    # lighter blue (royal blue)
    "ECFP": "#008080",     # teal
    "GCamp Ca+": "#00FF00",     # green
    "GCampCa-": "#32CD32",  # lime green (slightly different from GCamp Ca+)
    "LSSmOrange": "#FF8C00", # orange (dark orange)
    "TdTomato": "#FFD700",  # yellow (gold, same as fig 2)
    "mCherry": "#E31A1C",   # red (same as fig 1)
    "LSSmKAte": "#8B0000",  # dark red
    "mNeptune": "#4B0082",  # purple (same as fig 1)
    "YFP": "#FFFF00",       # yellow (for eYFP)
    "mScarlet": "#FF4500"   # orange-red (for mScarlet)
}


def load_figure5_2p_spectra(fluorophore_name, spectra_dir=None):
    """
    Load 2P excitation spectra for Figure 5 fluorophores.
    
    Wrapper around the flexible loading function in helpers.
    
    Parameters
    ----------
    fluorophore_name : str
        Name of fluorophore
    spectra_dir : str, optional
        Directory containing spectra CSV files. Defaults to dev_scripts/demo_data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Wavelength, Excitation, Emission
    """
    return load_2p_spectra_flexible(fluorophore_name, spectra_dir)


def subpanel_1(row_dict=None, ax=None, wavelength_range=(740, 1250)):
    """
    Generate subpanel 1: 2P excitation spectra for many fluorophores.
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses FIG_5_ROW_DICT
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (will create if None)
    wavelength_range : tuple, optional
        Wavelength range to plot (min, max). Default (950, 1250)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = FIG_5_ROW_DICT
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    fluorophores = row_dict["Fluorophores"]
    excitation_wavelengths = row_dict["Excitation wavelengths"]
    
    # Load spectra for each fluorophore
    spectra_dict = {}
    for fp_name in fluorophores:
        try:
            spectra_dict[fp_name] = load_figure5_2p_spectra(fp_name)
            print(f"Loaded spectra for {fp_name}: {len(spectra_dict[fp_name])} data points, "
                  f"wavelength range: {spectra_dict[fp_name]['Wavelength'].min():.1f}-{spectra_dict[fp_name]['Wavelength'].max():.1f}nm, "
                  f"max excitation: {spectra_dict[fp_name]['Excitation'].max():.4f}")
        except Exception as e:
            print(f"Warning: Could not load spectra for {fp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(spectra_dict) == 0:
        print("ERROR: No spectra were loaded. Cannot generate plot.")
        return fig, ax
    
    # Get colors for fluorophores
    colors = [FIG_5_FP_COLORS.get(fp_name, "#808080") for fp_name in fluorophores if fp_name in spectra_dict]
    
    # Filter to wavelength range and plot
    legend_patches = []
    plotted_count = 0
    for i, (fp_name, df) in enumerate(spectra_dict.items()):
        # Filter to wavelength range
        mask = (df["Wavelength"] >= wavelength_range[0]) & (df["Wavelength"] <= wavelength_range[1])
        df_filtered = df[mask].copy()
        
        # Check if we have data in this range
        if len(df_filtered) == 0:
            print(f"Warning: {fp_name} has no data in wavelength range {wavelength_range}")
            continue
        
        # Check if excitation data exists and is not all zeros
        if df_filtered["Excitation"].sum() == 0:
            print(f"Warning: {fp_name} has no 2P excitation data in wavelength range {wavelength_range}")
            continue
        
        # Apply smoothing
        df_smoothed = apply_smoothing_to_spectrum(df_filtered, smoothing_std=5)
        
        # Normalize excitation to max = 1 for visibility (like figure1)
        smoothed_excitation = df_smoothed["Excitation"].values
        max_excitation = smoothed_excitation.max()
        if max_excitation > 0:
            normalized_excitation = smoothed_excitation / max_excitation
        else:
            normalized_excitation = smoothed_excitation
        
        # Plot and collect handle for legend
        color = FIG_5_FP_COLORS.get(fp_name, "#808080")
        # Format label for display: "GCampCa-" -> "GCamp Ca-"
        display_label = fp_name.replace("GCampCa-", "GCamp Ca-")
        line, = ax.plot(df_smoothed["Wavelength"], normalized_excitation, 
                       color=color, linewidth=2, label=display_label, alpha=0.8)
        legend_patches.append(line)
        plotted_count += 1
    
    if plotted_count == 0:
        print("ERROR: No spectra were plotted. Check if spectra files exist and contain 2P data in range 950-1250nm")
        # Still return the figure so it can be saved (will be empty but won't crash)
    
    # Add vertical lines for excitation wavelengths (prominent, like Figure 1)
    for idx, wl in enumerate(excitation_wavelengths):
        if wavelength_range[0] <= wl <= wavelength_range[1]:
            # Get color and style from config (cycle if more than available)
            line_color = cfg.excitation_line_colors[idx % len(cfg.excitation_line_colors)]
            line_style = cfg.excitation_line_styles[idx % len(cfg.excitation_line_styles)]
            
            ax.axvline(
                wl,
                color=line_color,
                linestyle=line_style,
                linewidth=3,
                alpha=0.7
            )
            # Add text label above the line (horizontal, with box)
            ax.text(wl, ax.get_ylim()[1] * 0.95 + 0.03, f'{wl}nm', 
                   ha='center', va='bottom', fontsize=12, rotation=0,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("2P Excitation (normalized)", fontsize=12)
    ax.set_title("2P Excitation Spectra", fontsize=14, fontweight='bold', y=1.05)
    ax.set_xlim(wavelength_range)
    # Keep x-axis exactly at 0 (avoid matplotlib auto-padding below zero)
    ax.set_ylim(bottom=0)
    ax.margins(y=0)
    
    # Create legend with patches for fluorophores only (not excitation lines)
    # Position at far right, centered vertically
    ax.legend(handles=legend_patches, loc='center right', fontsize=9, 
              bbox_to_anchor=(1.20, 0.5))
    # Clean panel style (like Figure 1)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    
    return fig, ax


def subpanel_1_demo_figure(row_dict=None, ax=None):
    """
    Wrapper for subpanel_1 that limits to excitation wavelengths >= 890 nm for demo_figure.
    
    This is specific to demo_figure and doesn't affect the shared subpanel_1 function
    used by regular Figure 5.
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses DEMO_FIGURE_ROW_DICT
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (will create if None)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = DEMO_FIGURE_ROW_DICT
    
    # Create a modified row_dict that only includes excitation wavelengths >= 890
    modified_row_dict = row_dict.copy()
    original_wavelengths = row_dict["Excitation wavelengths"]
    filtered_wavelengths = [wl for wl in original_wavelengths if wl >= 890]
    modified_row_dict["Excitation wavelengths"] = filtered_wavelengths
    
    # Call the base subpanel_1 function with wavelength range starting at 890
    fig, ax = subpanel_1(modified_row_dict, ax, wavelength_range=(890, 1250))
    
    # Ensure x-axis starts at 890
    ax.set_xlim(890, 1250)
    
    return fig, ax


def subpanel_2(row_dict=None, ax=None, wavelength_range=(400, 700)):
    """
    Generate subpanel 2: 1P emission spectra for many fluorophores with filters.
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses FIG_5_ROW_DICT
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (will create if None)
    wavelength_range : tuple, optional
        Wavelength range to plot (min, max). Default (400, 700)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = FIG_5_ROW_DICT
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    fluorophores = row_dict["Fluorophores"]
    emission_filters = row_dict["emission filters"]
    
    # Load spectra for each fluorophore
    spectra_dict = {}
    for fp_name in fluorophores:
        try:
            spectra_dict[fp_name] = load_figure5_2p_spectra(fp_name)
        except Exception as e:
            print(f"Warning: Could not load spectra for {fp_name}: {e}")
            continue
    
    # Plot emission filters first (as shaded regions)
    # Map filters to nearest FP colors with high alpha
    filter_to_fp_color = {
        (400, 440): FIG_5_FP_COLORS["EBFP"],      # BFP - dark blue
        (445, 475): FIG_5_FP_COLORS["tagBFP"],   # tagBFP - lighter blue
        (475, 495): FIG_5_FP_COLORS["ECFP"],    # ECFP - teal
        (500, 550): FIG_5_FP_COLORS["GCamp Ca+"],   # GCamp Ca+ - green
        (550, 580): FIG_5_FP_COLORS["TdTomato"], # TdTomato - yellow
        (590, 620): FIG_5_FP_COLORS["mCherry"],  # mCherry - red (same as fig 1)
        (645, 695): FIG_5_FP_COLORS["mNeptune"]  # mNeptune - purple (same as fig 1)
    }
    
    # Also map filter names to colors
    filter_name_to_color = {
        "DarkBlue": FIG_5_FP_COLORS["EBFP"],
        "Blue": FIG_5_FP_COLORS["tagBFP"],
        "Cyan": FIG_5_FP_COLORS["ECFP"],
        "NarrowGreen": "#00FF00",  # Green color for narrow green filter
        "Yellow": FIG_5_FP_COLORS["TdTomato"],
        "Orange": FIG_5_FP_COLORS["TdTomato"],
        "Red": FIG_5_FP_COLORS["mCherry"],
        "FarRed": FIG_5_FP_COLORS["mNeptune"]
    }
    
    for filter_spec in emission_filters:
        # Handle both filter names (strings) and filter ranges (lists)
        if isinstance(filter_spec, str):
            # Look up filter name in config to get display range
            if hasattr(cfg, "emission_filter_display_ranges") and filter_spec in cfg.emission_filter_display_ranges:
                filter_range = cfg.emission_filter_display_ranges[filter_spec]
            elif hasattr(cfg, "emission_filter_sets") and filter_spec in cfg.emission_filter_sets:
                filter_config = cfg.emission_filter_sets[filter_spec]
                if isinstance(filter_config, list) and len(filter_config) == 2:
                    filter_range = filter_config
                else:
                    print(f"Warning: Could not determine range for filter {filter_spec}, skipping")
                    continue
            else:
                print(f"Warning: Filter name {filter_spec} not found in config, skipping")
                continue
            color = filter_name_to_color.get(filter_spec, 'gray')
        elif isinstance(filter_spec, list) and len(filter_spec) == 2:
            filter_range = filter_spec
            filter_key = tuple(filter_range)
            color = filter_to_fp_color.get(filter_key, 'gray')
        else:
            print(f"Warning: Unknown filter format {filter_spec}, skipping")
            continue
        
        ax.axvspan(filter_range[0], filter_range[1], alpha=0.3, color=color)
    
    # Get colors for fluorophores and plot emission spectra
    for fp_name, df in spectra_dict.items():
        # Filter to wavelength range
        mask = (df["Wavelength"] >= wavelength_range[0]) & (df["Wavelength"] <= wavelength_range[1])
        df_filtered = df[mask].copy()
        
        # Apply smoothing
        df_smoothed = apply_smoothing_to_spectrum(df_filtered, smoothing_std=10)
        
        # Plot emission spectrum
        color = FIG_5_FP_COLORS.get(fp_name, "#808080")
        # Format label for display: "GCampCa-" -> "GCamp Ca-"
        display_label = fp_name.replace("GCampCa-", "GCamp Ca-")
        ax.plot(df_smoothed["Wavelength"], df_smoothed["Emission"], 
               color=color, linewidth=2, label=display_label, alpha=0.8)
    
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("1P Emission (normalized)", fontsize=12)
    ax.set_title("1P Emission Spectra with Filters", fontsize=14, fontweight='bold')
    ax.set_xlim(wavelength_range)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def compute_predicted_signals_figure5(row_dict, power_mw=20.0):
    """
    Compute predicted signals for all channel combinations in Figure 5.
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary with fluorophores, excitation wavelengths, and emission filters
    power_mw : float
        Laser power in mW (default 20mW). For 2P excitation, signal scales as (power/20)^2.
        
    Returns
    -------
    dict
        Dictionary with structure: {fp_name: {channel_key: signal_value}}
        where channel_key is a string like "750nm_[400,440]"
    """
    fluorophores = row_dict["Fluorophores"]
    excitation_wavelengths = row_dict["Excitation wavelengths"]
    emission_filters = row_dict["emission filters"]
    
    signals = {fp: {} for fp in fluorophores}
    
    # Power correction factor: (power/20)^2 for 2P excitation
    power_factor = (power_mw / 20.0) ** 2
    
    # Load spectra for all fluorophores and apply smoothing
    spectra_dict = {}
    for fp_name in fluorophores:
        df_raw = load_figure5_2p_spectra(fp_name)
        # Apply 5nm Gaussian smoothing to excitation spectra
        df_smoothed = apply_smoothing_to_spectrum(df_raw, smoothing_std=5)
        spectra_dict[fp_name] = df_smoothed
    
    # Process each channel combination
    for exc_wl in excitation_wavelengths:
        for filter_spec in emission_filters:
            # Handle both filter names (strings) and filter ranges (lists)
            if isinstance(filter_spec, str):
                # Look up filter name in config to get range or CSV
                if hasattr(cfg, "emission_filter_sets") and filter_spec in cfg.emission_filter_sets:
                    filter_config = cfg.emission_filter_sets[filter_spec]
                    if isinstance(filter_config, list) and len(filter_config) == 2:
                        filter_min, filter_max = filter_config[0], filter_config[1]
                    elif isinstance(filter_config, str):
                        # CSV file - need to load it to get range
                        from src.figure1 import load_filter_transmission
                        filter_df = load_filter_transmission(filter_spec)
                        if filter_df is not None and len(filter_df) > 0:
                            filter_min = filter_df["Wavelength"].min()
                            filter_max = filter_df["Wavelength"].max()
                        else:
                            print(f"Warning: Could not load filter {filter_spec}, skipping")
                            continue
                    else:
                        print(f"Warning: Unknown filter config format for {filter_spec}, skipping")
                        continue
                else:
                    print(f"Warning: Filter name {filter_spec} not found in config, skipping")
                    continue
                channel_key = f"{exc_wl}nm_{filter_spec}"
            elif isinstance(filter_spec, list) and len(filter_spec) == 2:
                filter_min, filter_max = filter_spec[0], filter_spec[1]
                channel_key = f"{exc_wl}nm_{filter_spec}"
            else:
                print(f"Warning: Unknown filter format {filter_spec}, skipping")
                continue
            
            # Compute signal for each fluorophore
            for fp_name in fluorophores:
                df = spectra_dict[fp_name]
                
                # Get 2P excitation at excitation wavelength
                exc_mask = np.abs(df["Wavelength"] - exc_wl).idxmin()
                exc_value = df.loc[exc_mask, "Excitation"]
                
                # Get 1P emission filtered by emission filter range
                em_mask = (df["Wavelength"] >= filter_min) & (df["Wavelength"] <= filter_max)
                emission_in_range = df.loc[em_mask].copy()
                
                # Calculate wavelength spacing for proper integration
                wavelengths = emission_in_range["Wavelength"].values
                if len(wavelengths) > 1:
                    wavelength_spacings = np.diff(wavelengths)
                    spacings = np.concatenate([[wavelength_spacings[0]], 
                                              (wavelength_spacings[:-1] + wavelength_spacings[1:]) / 2,
                                              [wavelength_spacings[-1]]])
                else:
                    spacings = np.array([1.0])
                
                # Load filter transmission if available (for CSV-based filters)
                filter_transmission_df = None
                if isinstance(filter_spec, str):
                    from src.figure1 import load_filter_transmission
                    filter_transmission_df = load_filter_transmission(filter_spec)
                
                # Apply filter transmission if available
                if filter_transmission_df is not None:
                    # Interpolate filter transmission to match emission wavelengths
                    # Transmission values in CSV are percentages (0-100), convert to fractions (0-1)
                    filter_transmission_interp = np.interp(
                        wavelengths,
                        filter_transmission_df["Wavelength"].values,
                        filter_transmission_df["Transmission"].values,
                        left=0,  # Outside range: 0 transmission
                        right=0  # Outside range: 0 transmission
                    ) / 100.0  # Convert from percentage to fraction
                    
                    # Integrate: multiply emission by filter transmission and wavelength spacing, then sum
                    filtered_emission = (emission_in_range["Emission"].values * filter_transmission_interp * spacings).sum()
                else:
                    # No transmission file - assume 100% transmission within filter range
                    # Integrate: multiply emission by wavelength spacing, then sum
                    filtered_emission = (emission_in_range["Emission"].values * spacings).sum()
                
                # Predicted signal = excitation * filtered emission * power_factor
                signal = exc_value * filtered_emission * power_factor
                signals[fp_name][channel_key] = signal
    
    return signals


def _add_channel_mapping_legend(fig, channel_labels, position='right', fontsize=None, bbox_x=None):
    """
    Add channel mapping legend to figure.
    
    Formats channel labels to be easier to read:
    - Bracketed filter ranges are indented with spaces on second line so numbers stand alone
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add legend to
    channel_labels : list
        List of channel label strings (format: "750nm\n[400, 440]" or "750nm_[400, 440]")
    position : str
        'right' for right side (centered vertically) or 'bottom' for bottom
    fontsize : int, optional
        Font size for legend. If None, uses default (int(18*0.75))
    bbox_x : float, optional
        X position for bbox_to_anchor when position='right'. If None, uses default (1.25)
        
    Returns
    -------
    legend : matplotlib.legend.Legend
        The legend object, which can be adjusted after creation (e.g., legend.set_fontsize())
    """
    from matplotlib.patches import Rectangle
    import re
    
    if fontsize is None:
        fontsize = int(18*0.75)
    
    legend_handles = []
    legend_labels = []
    for i, ch_label in enumerate(channel_labels):
        # Create a small invisible patch for each entry
        patch = Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='none')
        legend_handles.append(patch)
        
        # Format label: if it contains a bracketed filter range, indent it on second line
        # Handle both formats: "750nm\n[400, 440]" and "750nm_[400, 440]"
        formatted_label = ch_label
        
        # Check if label contains bracketed range (e.g., "[400, 440]" or "[550,580]")
        bracket_match = re.search(r'\[([\d\s,]+)\]', ch_label)
        if bracket_match:
            # Split on newline or underscore
            if '\n' in ch_label:
                parts = ch_label.split('\n', 1)
                exc_part = parts[0].strip()
                filter_part = parts[1].strip()
            elif '_' in ch_label:
                parts = ch_label.split('_', 1)
                exc_part = parts[0].strip()
                filter_part = parts[1].strip()
            else:
                # Try to extract bracket part
                exc_part = ch_label[:bracket_match.start()].strip()
                filter_part = bracket_match.group(0)
            
            # Format with indentation on second line: "750nm\n        [400, 440]"
            # Use 8 spaces for indentation (more visible than 4)
            formatted_label = f"{exc_part}\n        {filter_part}"
        else:
            # No bracket found, use as-is but ensure it's formatted nicely
            if '\n' in ch_label:
                parts = ch_label.split('\n', 1)
                formatted_label = f"{parts[0]}\n        {parts[1]}"
        
        legend_labels.append(f"{i+1}: {formatted_label}")
    
    if position == 'right':
        # Vertical list on the right, centered vertically
        # Increase fontsize for better readability
        readable_fontsize = max(fontsize, 12)  # At least 12pt for readability
        # Use provided bbox_x or default to 0.97, but allow override for further right
        # Default moved left by 2 bar widths (0.2 units) from subpanel 3a perspective, then right by 0.02
        if bbox_x is None:
            bbox_x = 0.97
        legend = fig.legend(legend_handles, legend_labels, 
                  loc='center right', ncol=1, 
                  fontsize=readable_fontsize, frameon=True, 
                  framealpha=0.0,  # Fully transparent background
                  columnspacing=0.5,  # Reduce spacing between columns
                  handletextpad=0.3,  # Reduce spacing between handle and text
                  borderpad=0.3,  # Reduce padding around legend
                  bbox_to_anchor=(bbox_x, 0.5))
        legend.set_zorder(1)  # Lower z-order so bars appear on top
    else:
        # Horizontal at the bottom
        readable_fontsize = max(fontsize, 12)  # At least 12pt for readability
        legend = fig.legend(legend_handles, legend_labels, 
                  loc='lower center', ncol=min(10, len(channel_labels)), 
                  fontsize=readable_fontsize, frameon=True, 
                  bbox_to_anchor=(0.5, -0.05))
    
    return legend


def subpanel_3(row_dict=None, ax=None, min_signal_threshold=0.2, 
                legend_fontsize=None, legend_bbox_x=None):
    """
    Generate subpanel 3: Visualization of predicted unmixing ratios for all channels.
    
    Generic version matching Figure 1 subpanel 4 structure:
    - Creates N+1 vertically stacked subplots (N = number of fluorophores)
    - Each fluorophore gets its own subplot (top to bottom)
    - Last (bottom) subplot is overlay with all FPs
    - Bars touch (histogram style, width=1.0)
    - Filters out channels where all FPs are below threshold
    - Legend optional (no legend for Figure 5, with legend for Figure 2)
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses FIG_5_ROW_DICT
    ax : matplotlib.axes.Axes or array, optional
        Axes to plot on (will create subplots if None)
    min_signal_threshold : float
        Minimum signal threshold. Channels where all FPs are below this are filtered out.
    legend_fontsize : int, optional
        Font size for legend. If None, no legend is added (Figure 5 default)
    legend_bbox_x : float, optional
        X position for legend bbox_to_anchor. If None, uses default (1.35)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = FIG_5_ROW_DICT
    
    fluorophores = row_dict["Fluorophores"]
    excitation_wavelengths = row_dict["Excitation wavelengths"]
    emission_filters = row_dict["emission filters"]
    
    # Compute predicted signals for all channels
    signals = compute_predicted_signals_figure5(row_dict)
    
    # Create channel labels and get signals for each channel
    channel_keys = []
    for exc_wl in excitation_wavelengths:
        for filter_range in emission_filters:
            channel_key = f"{exc_wl}nm_{filter_range}"
            channel_keys.append(channel_key)
    
    # Extract signal values for each fluorophore (raw, not normalized yet)
    fp_data_raw = {}
    for fp_name in fluorophores:
        fp_signals = np.array([signals[fp_name].get(ch, 0) for ch in channel_keys])
        fp_data_raw[fp_name] = fp_signals
    
    # Filter out channels where all FPs are below threshold
    valid_channel_mask = np.zeros(len(channel_keys), dtype=bool)
    for i in range(len(channel_keys)):
        max_signal = max([fp_data_raw[fp][i] for fp in fluorophores if fp in fp_data_raw], default=0)
        if max_signal >= min_signal_threshold:
            valid_channel_mask[i] = True
    
    # Filter channels
    filtered_channel_keys = [ch for i, ch in enumerate(channel_keys) if valid_channel_mask[i]]
    filtered_fp_data_raw = {}
    for fp_name in fluorophores:
        if fp_name in fp_data_raw:
            filtered_fp_data_raw[fp_name] = fp_data_raw[fp_name][valid_channel_mask]
    
    # Normalize each FP's signals to max = 1 (so brightest channel for each FP is 1)
    fp_data = {}
    for fp_name in fluorophores:
        if fp_name in filtered_fp_data_raw:
            fp_signals = filtered_fp_data_raw[fp_name]
            max_signal = fp_signals.max()
            if max_signal > 0:
                fp_data[fp_name] = fp_signals / max_signal
            else:
                fp_data[fp_name] = fp_signals
    
    # Sort channels by FP preference
    fp_order = ["EBFP", "tagBFP", "ECFP", "GCamp Ca+", "GCampCa-", "LSSmOrange", 
                "TdTomato", "mCherry", "LSSmKAte", "mNeptune"]
    
    # Find best FP for each channel
    channel_best_fp = []
    for i in range(len(filtered_channel_keys)):
        best_fp_idx = 0
        best_signal = 0
        for j, fp_name in enumerate(fluorophores):
            if fp_name in fp_data:
                signal_val = fp_data[fp_name][i]
                if signal_val > best_signal:
                    best_signal = signal_val
                    best_fp_idx = j
        channel_best_fp.append(best_fp_idx)
    
    # Sort channels by best FP (using fp_order for ordering)
    def get_sort_key(i):
        best_fp_idx = channel_best_fp[i]
        if best_fp_idx < len(fluorophores):
            best_fp_name = fluorophores[best_fp_idx]
            if best_fp_name in fp_order:
                fp_priority = fp_order.index(best_fp_name)
            else:
                fp_priority = 999
        else:
            fp_priority = 999
        max_signal = max([fp_data[fp][i] for fp in fluorophores if fp in fp_data], default=0)
        return (fp_priority, -max_signal)
    
    sorted_indices = sorted(range(len(filtered_channel_keys)), key=get_sort_key)
    
    # Reorder data according to sorted indices
    sorted_channel_keys = [filtered_channel_keys[i] for i in sorted_indices]
    sorted_fp_data = {}
    for fp_name in fluorophores:
        if fp_name in fp_data:
            sorted_fp_data[fp_name] = fp_data[fp_name][sorted_indices]
    
    # Create short channel labels for x-axis
    channel_labels = []
    for ch_key in sorted_channel_keys:
        parts = ch_key.split("_")
        exc_part = parts[0]  # e.g., "750nm"
        filter_part = parts[1] if len(parts) > 1 else ""  # e.g., "[400, 440]"
        label = f"{exc_part}\n{filter_part}"
        channel_labels.append(label)
    
    # Use base function to create plots
    if ax is not None:
        if isinstance(ax, np.ndarray):
            axes = ax
            fig = axes[0].figure
            # Plot on provided axes (reuse existing figure)
            # Note: base function creates new figure, so we need to handle this differently
            # For now, if ax is provided, we'll still create new figure but could be improved
            pass
    
    fig, axes, channel_labels = create_ratio_histogram_base(
        row_dict, sorted_channel_keys, sorted_fp_data, channel_labels, 
        fluorophores, FIG_5_FP_COLORS, figsize_width=20, label_every_other=True)
    
    # Add legend if requested (for Figure 2)
    if legend_fontsize is not None:
        if legend_bbox_x is None:
            legend_bbox_x = 1.35  # Default position
        legend = _add_channel_mapping_legend(fig, channel_labels, position='right', 
                                    fontsize=legend_fontsize, bbox_x=legend_bbox_x)
        plt.subplots_adjust(hspace=0, right=0.75)  # Extra space for legend
    else:
        # No legend for subpanel 3 (Figure 5)
        plt.subplots_adjust(hspace=0)  # No extra right space needed
    
    plt.tight_layout()
    
    return fig, axes


def subpanel_3a(row_dict=None, ax=None, legend_fontsize=None, legend_bbox_x=None):
    """
    Generate subpanel 3a: Simplified version showing only the best channel for each FP.
    
    For each fluorophore, finds the channel where that FP most exceeds the nearest other FP.
    This results in 10 channels (one for each of the 10 fluorophores).
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses FIG_5_ROW_DICT
    ax : matplotlib.axes.Axes or array, optional
        Axes to plot on (will create subplots if None)
    legend_fontsize : int, optional
        Font size for legend. If None, uses default (3x larger: int(18*0.75*3))
    legend_bbox_x : float, optional
        X position for legend bbox_to_anchor. If None, uses default (1.41)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = FIG_5_ROW_DICT
    
    fluorophores = row_dict["Fluorophores"]
    excitation_wavelengths = row_dict["Excitation wavelengths"]
    emission_filters = row_dict["emission filters"]
    
    # Compute predicted signals for all channels
    signals = compute_predicted_signals_figure5(row_dict)
    
    # Create channel labels and get signals for each channel
    channel_keys = []
    for exc_wl in excitation_wavelengths:
        for filter_range in emission_filters:
            channel_key = f"{exc_wl}nm_{filter_range}"
            channel_keys.append(channel_key)
    
    # Extract signal values for each fluorophore (raw, not normalized yet)
    fp_data_raw = {}
    for fp_name in fluorophores:
        fp_signals = np.array([signals[fp_name].get(ch, 0) for ch in channel_keys])
        fp_data_raw[fp_name] = fp_signals
    
    # Normalize each FP's signals to max = 1 (relative signals)
    fp_data_relative = {}
    for fp_name in fluorophores:
        if fp_name in fp_data_raw:
            fp_signals = fp_data_raw[fp_name]
            max_signal = fp_signals.max()
            if max_signal > 0:
                fp_data_relative[fp_name] = fp_signals / max_signal
            else:
                fp_data_relative[fp_name] = fp_signals
    
    # For each FP, find the best channel where:
    # 1. That FP has the highest relative signal (among all FPs)
    # 2. Among those channels, pick the one where min(target_fp_relative / other_fp_relative) is highest
    best_channels = {}  # {fp_name: channel_index}
    
    for fp_name in fluorophores:
        if fp_name not in fp_data_relative:
            continue
        
        # Find channels where this FP has the highest relative signal
        candidate_channels = []
        for ch_idx in range(len(channel_keys)):
            target_relative = fp_data_relative[fp_name][ch_idx]
            
            # Check if this FP has the highest relative signal in this channel
            is_highest = True
            for other_fp in fluorophores:
                if other_fp != fp_name and other_fp in fp_data_relative:
                    other_relative = fp_data_relative[other_fp][ch_idx]
                    if other_relative > target_relative:
                        is_highest = False
                        break
            
            if is_highest and target_relative > 0:
                candidate_channels.append(ch_idx)
        
        # Among candidate channels, find the one with highest min(target_fp_relative / other_fp_relative)
        best_channel_idx = None
        best_min_ratio = -np.inf
        
        for ch_idx in candidate_channels:
            target_relative = fp_data_relative[fp_name][ch_idx]
            
            # Calculate min(target_fp_relative / other_fp_relative) across all other FPs
            min_ratio = np.inf
            for other_fp in fluorophores:
                if other_fp != fp_name and other_fp in fp_data_relative:
                    other_relative = fp_data_relative[other_fp][ch_idx]
                    if other_relative > 0:
                        ratio = target_relative / other_relative
                        min_ratio = min(min_ratio, ratio)
                    else:
                        # If other FP has 0 signal, ratio is infinite (best case)
                        min_ratio = np.inf
            
            # Update best channel if this one has higher minimum ratio
            if min_ratio > best_min_ratio:
                best_min_ratio = min_ratio
                best_channel_idx = ch_idx
        
        if best_channel_idx is not None:
            best_channels[fp_name] = best_channel_idx
    
    # Get unique channel indices (in case multiple FPs share the same best channel)
    unique_channel_indices = sorted(set(best_channels.values()))
    
    # Create mapping from channel index to FP (which FP selected this channel)
    # If multiple FPs share a channel, use the first one in FP order
    fp_order = ["EBFP", "tagBFP", "ECFP", "GCamp Ca+", "GCampCa-", "LSSmOrange", 
                "TdTomato", "mCherry", "LSSmKAte", "mNeptune"]
    channel_to_fp = {}
    for ch_idx in unique_channel_indices:
        # Find which FP(s) selected this channel
        selecting_fps = [fp for fp, idx in best_channels.items() if idx == ch_idx]
        # Sort by FP order and take the first
        selecting_fps.sort(key=lambda x: fp_order.index(x) if x in fp_order else 999)
        if selecting_fps:
            channel_to_fp[ch_idx] = selecting_fps[0]
    
    # Sort channels by FP order
    def get_channel_sort_key(ch_idx):
        if ch_idx in channel_to_fp:
            fp_name = channel_to_fp[ch_idx]
            if fp_name in fp_order:
                return fp_order.index(fp_name)
        return 999
    
    sorted_channel_indices = sorted(unique_channel_indices, key=get_channel_sort_key)
    sorted_channel_keys = [channel_keys[i] for i in sorted_channel_indices]
    
    # Extract data for selected channels only (in sorted order)
    selected_fp_data_raw = {}
    for fp_name in fluorophores:
        if fp_name in fp_data_raw:
            selected_fp_data_raw[fp_name] = fp_data_raw[fp_name][sorted_channel_indices]
    
    # Normalize each FP's signals to max = 1 (so brightest channel for each FP is 1)
    fp_data = {}
    for fp_name in fluorophores:
        if fp_name in selected_fp_data_raw:
            fp_signals = selected_fp_data_raw[fp_name]
            max_signal = fp_signals.max()
            if max_signal > 0:
                fp_data[fp_name] = fp_signals / max_signal
            else:
                fp_data[fp_name] = fp_signals
    
    # Use fp_data directly (already sorted)
    sorted_fp_data = fp_data
    
    # Create short channel labels for x-axis
    channel_labels = []
    # Create mapping from channel key to FP for legend
    channel_key_to_fp = {}
    for i, ch_key in enumerate(sorted_channel_keys):
        parts = ch_key.split("_")
        exc_part = parts[0]  # e.g., "750nm"
        filter_part = parts[1] if len(parts) > 1 else ""  # e.g., "[400, 440]"
        label = f"{exc_part}\n{filter_part}"
        channel_labels.append(label)
        
        # Map channel key to its best FP
        ch_idx = sorted_channel_indices[i]
        if ch_idx in channel_to_fp:
            channel_key_to_fp[ch_key] = channel_to_fp[ch_idx]
    
    # Use base function to create plots
    if ax is not None:
        if isinstance(ax, np.ndarray):
            axes = ax
            fig = axes[0].figure
            # Plot on provided axes (reuse existing figure)
            # Note: base function creates new figure, so we need to handle this differently
            # For now, if ax is provided, we'll still create new figure but could be improved
            pass
    
    fig, axes, channel_labels = create_ratio_histogram_base(
        row_dict, sorted_channel_keys, sorted_fp_data, channel_labels, 
        fluorophores, FIG_5_FP_COLORS, figsize_width=12, label_every_other=False)
    
    # Create enhanced channel labels for legend with FP information
    enhanced_channel_labels = []
    for ch_key in sorted_channel_keys:
        parts = ch_key.split("_")
        exc_part = parts[0]  # e.g., "750nm"
        filter_part = parts[1] if len(parts) > 1 else ""  # e.g., "[400, 440]"
        
        # Add FP name if this channel is best for a specific FP
        if ch_key in channel_key_to_fp:
            fp_name = channel_key_to_fp[ch_key]
            # Format FP name for display (e.g., "GCampCa-" -> "GCamp Ca-")
            display_fp_name = fp_name.replace("GCampCa-", "GCamp Ca-")
            label = f"{exc_part} {filter_part}\n(best for {display_fp_name})"
        else:
            label = f"{exc_part}\n{filter_part}"
        enhanced_channel_labels.append(label)
    
    # Add channel mapping legend on the right, centered vertically
    # Use default values from 5.3a if not specified
    if legend_fontsize is None:
        legend_fontsize = int(18*0.75*3)  # 3x larger
    if legend_bbox_x is None:
        legend_bbox_x = 1.41  # Default position from 5.3a
    
    legend = _add_channel_mapping_legend(fig, enhanced_channel_labels, position='right', 
                                fontsize=legend_fontsize, bbox_x=legend_bbox_x)
    
    # Adjust legend font size if needed (wrapper can adjust after creation)
    # Legend can be adjusted after creation if needed: legend.set_fontsize(new_size)
    
    # Remove spacing between subplots
    plt.subplots_adjust(hspace=0, right=0.75)  # Extra space for legend on right
    plt.tight_layout()
    
    return fig, axes


def subpanel_5(row_dict=None, ax=None):
    """
    Generate subpanel 5: Angle plots for each fluorophore.
    
    For each FP, plots vertical lines showing angles to all other FPs.
    X-axis: angle (0-90 degrees)
    Y-axis: -10 to 90 (to make target at 0 obvious)
    Separate subplot for each FP.
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses FIG_5_ROW_DICT
    ax : matplotlib.axes.Axes or array, optional
        Axes to plot on (will create subplots if None)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = FIG_5_ROW_DICT
    
    fluorophores = row_dict["Fluorophores"]
    
    # Compute predicted signals for all channels
    signals = compute_predicted_signals_figure5(row_dict)
    
    # Get all channel keys
    excitation_wavelengths = row_dict["Excitation wavelengths"]
    emission_filters = row_dict["emission filters"]
    channel_keys = []
    for exc_wl in excitation_wavelengths:
        for filter_range in emission_filters:
            channel_key = f"{exc_wl}nm_{filter_range}"
            channel_keys.append(channel_key)
    
    # Build vectors for each FP (from predicted signals across all channels)
    fp_vectors = {}
    for fp_name in fluorophores:
        vector = np.array([signals[fp_name].get(ch, 0) for ch in channel_keys])
        # Normalize to unit vector
        magnitude = np.linalg.norm(vector)
        if magnitude > 0:
            fp_vectors[fp_name] = vector / magnitude
        else:
            fp_vectors[fp_name] = vector
    
    # Compute angles between each FP and all other FPs
    fp_angles = {}  # {fp_name: {other_fp: angle}}
    for fp_name in fluorophores:
        fp_angles[fp_name] = {}
        vec1 = fp_vectors[fp_name]
        
        for other_fp in fluorophores:
            if other_fp == fp_name:
                continue
            
            vec2 = fp_vectors[other_fp]
            
            # Compute angle between vectors using dot product
            dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)
            
            # Map to 0-90 range (take minimum of angle and 180-angle)
            angle_deg = min(angle_deg, 180 - angle_deg)
            
            fp_angles[fp_name][other_fp] = angle_deg
    
    # Create subplots: one for each FP (rotated 90 degrees - horizontal layout)
    n_subplots = len(fluorophores)
    if ax is None:
        fig, axes = plt.subplots(1, n_subplots, figsize=(2 * n_subplots, 10), sharey=True)
    else:
        if isinstance(ax, np.ndarray):
            axes = ax
            fig = axes[0].figure
        else:
            fig, axes = plt.subplots(1, n_subplots, figsize=(2 * n_subplots, 10), sharey=True)
    
    # Get colors
    colors = [FIG_5_FP_COLORS.get(fp_name, "#808080") for fp_name in fluorophores]
    
    # Collect all legend handles and labels for shared legend
    all_legend_handles = []
    all_legend_labels = []
    
    # Plot for each FP
    for fp_idx, fp_name in enumerate(fluorophores):
        ax_sub = axes[fp_idx]
        display_label = fp_name.replace("GCampCa-", "GCamp Ca-")
        
        # Find closest FP (smallest angle to target, which is at 0)
        # The target FP itself is at 0, so we want the other FP with smallest angle
        closest_fp = None
        closest_angle = 90.0
        second_closest_fp = None
        second_closest_angle = 90.0
        all_angles = []
        for other_fp in fluorophores:
            if other_fp == fp_name:
                continue
            angle = fp_angles[fp_name][other_fp]
            all_angles.append((angle, other_fp))
        
        # Sort by angle to find closest and second closest
        all_angles.sort(key=lambda x: x[0])
        if len(all_angles) >= 1:
            closest_angle, closest_fp = all_angles[0]
        if len(all_angles) >= 2:
            second_closest_angle, second_closest_fp = all_angles[1]
        
        # Plot horizontal lines for angles to all other FPs (double thickness and length)
        for other_fp_idx, other_fp in enumerate(fluorophores):
            if other_fp == fp_name:
                continue
            
            angle = fp_angles[fp_name][other_fp]
            other_color = colors[other_fp_idx]
            other_display_label = other_fp.replace("GCampCa-", "GCamp Ca-")
            
            # Double thickness: linewidth=8 (was 4), 2x longer again for better visibility
            # Lines from -3 to 5 (was -1.5 to 2.5, now 2x longer)
            line = ax_sub.plot([-3, 5], [angle, angle], color=other_color, linestyle='-', 
                          linewidth=8, alpha=0.7, label=other_display_label)[0]
            
            # Collect handles for shared legend (only once per FP)
            if other_display_label not in all_legend_labels:
                all_legend_handles.append(line)
                all_legend_labels.append(other_display_label)
        
        # Add target line at 0 degrees using FP's color (solid line, 4x as thick as other lines)
        # For each subplot, use the FP's own color for the target line
        fp_color = colors[fp_idx]
        ax_sub.plot([-3, 5], [0, 0], color=fp_color, linestyle='-', linewidth=32, alpha=0.7)
        
        # Create legend entry for target (only once, gray line twice as thick)
        if fp_idx == 0:
            # Create a custom line for legend with gray color, twice as thick as other lines
            target_line = Line2D([0], [0], color='gray', linestyle='-', 
                                linewidth=16, alpha=0.8, label='Target (0°)')
            all_legend_handles.append(target_line)
            all_legend_labels.append('Target (0°)')
        
        # Set labels and limits
        # Rotated 90 degrees: y-axis is angle, x-axis extended to accommodate longer lines
        ax_sub.set_ylim(0, 90)  # Y-axis: 0 to 90 (target at 0)
        ax_sub.set_xlim(-3, 5)  # X-axis: extended range for 2x longer lines (was -1.5 to 2.5, now -3 to 5)
        # Title aligned at the bottom (position at y=0, angled 15 degrees down)
        ax_sub.text(0.5, 0, display_label, fontsize=24, fontweight='bold', 
                   ha='center', va='top', rotation=-15)
        ax_sub.set_xlabel("", fontsize=12)  # No x-label needed for horizontal lines
        ax_sub.set_xticks([])  # Hide x-axis ticks
        ax_sub.tick_params(axis='y', labelsize=36)  # 3x bigger
        ax_sub.set_ylabel("")  # Remove ylabel from all except leftmost
        ax_sub.grid(True, alpha=0.3, axis='y')
        ax_sub.spines["top"].set_visible(False)
        ax_sub.spines["right"].set_visible(False)
        ax_sub.spines["bottom"].set_visible(False)
        
        # Label the closest fluorophore (positioned like second closest labels, relative to their line)
        if closest_fp is not None:
            closest_display_label = closest_fp.replace("GCampCa-", "GCamp Ca-")
            # Position above the line at closest_angle, on the left side to avoid overlap
            x_offset = 0.3 + (fp_idx % 3) * 0.2  # Left side positioning
            ax_sub.text(x_offset, closest_angle + 2, closest_display_label, 
                       fontsize=16, ha='center', va='bottom', 
                       rotation=0, weight='normal',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Label the second closest fluorophore (using size/position that was good for third best)
        if second_closest_fp is not None:
            second_display_label = second_closest_fp.replace("GCampCa-", "GCamp Ca-")
            # Position above the line, on the right side to avoid overlap with closest label
            x_offset_second = 0.7 - (fp_idx % 3) * 0.2  # Right side positioning
            ax_sub.text(x_offset_second, second_closest_angle + 2, second_display_label, 
                       fontsize=16, ha='center', va='bottom', 
                       rotation=0, weight='normal',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Set y-axis label only on leftmost subplot (shared axis)
    axes[0].set_ylabel("Angle (degrees)", fontsize=36)
    
    # Reorder legend to match fluorophore order (EBFP at top)
    # Create mapping from display labels to handles
    label_to_handle = dict(zip(all_legend_labels, all_legend_handles))
    
    # Reorder according to fluorophore order
    ordered_handles = []
    ordered_labels = []
    
    # Add FPs in fluorophore order
    for fp_name in fluorophores:
        display_label = fp_name.replace("GCampCa-", "GCamp Ca-")
        if display_label in label_to_handle:
            ordered_handles.append(label_to_handle[display_label])
            ordered_labels.append(display_label)
    
    # Add target at the end if it exists
    if 'Target (0°)' in label_to_handle:
        ordered_handles.append(label_to_handle['Target (0°)'])
        ordered_labels.append('Target (0°)')
    
    # Create shared legend at the right side (vertical list, moved further right outside plot)
    fig.legend(ordered_handles, ordered_labels, 
              loc='center right', ncol=1, 
              fontsize=30, frameon=True, bbox_to_anchor=(1.22, 0.5))
    
    plt.tight_layout()
    
    return fig, axes


def save_all_subpanels(row_dict=None, output_dir="results/Figure5"):
    """
    Generate and save all subpanels for Figure 5.
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses FIG_5_ROW_DICT
    output_dir : str
        Directory to save figures
    """
    import os
    
    if row_dict is None:
        row_dict = FIG_5_ROW_DICT
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Figure 5 Subpanels")
    print("="*60)
    
    # Subpanel 1: 2P Excitation Spectra
    print("\nSubpanel 1: 2P Excitation Spectra...")
    try:
        # For demo_figure, use wrapper that limits to wavelengths >= 890 nm
        if row_dict is not None and row_dict.get("name") == "demo_figure":
            fig1, ax1 = subpanel_1_demo_figure(row_dict)
        else:
            fig1, ax1 = subpanel_1(row_dict)
        filepath1 = os.path.join(output_dir, "subpanel_1_2p_excitation_spectra.png")
        fig1.savefig(filepath1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved: {filepath1}")
    except Exception as e:
        print(f"  Error in subpanel 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Subpanel 2: 1P Emission Spectra
    print("\nSubpanel 2: 1P Emission Spectra...")
    try:
        fig2, ax2 = subpanel_2(row_dict)
        filepath2 = os.path.join(output_dir, "subpanel_2_1p_emission_spectra.png")
        fig2.savefig(filepath2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {filepath2}")
    except Exception as e:
        print(f"  Error in subpanel 2: {e}")
        import traceback
        traceback.print_exc()
    
    # Subpanel 3: Predicted Unmixing Ratios
    print("\nSubpanel 3: Predicted Unmixing Ratios...")
    try:
        fig3, ax3 = subpanel_3(row_dict)
        filepath3 = os.path.join(output_dir, "subpanel_3_predicted_unmixing_ratios.png")
        fig3.savefig(filepath3, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"Saved: {filepath3}")
    except Exception as e:
        print(f"  Error in subpanel 3: {e}")
        import traceback
        traceback.print_exc()
    
    # Subpanel 3a: Simplified - Best Channel per FP
    print("\nSubpanel 3a: Best Channel per FP...")
    try:
        fig3a, ax3a = subpanel_3a(row_dict)
        filepath3a = os.path.join(output_dir, "subpanel_3a_best_channel_per_fp.png")
        fig3a.savefig(filepath3a, dpi=300, bbox_inches='tight')
        plt.close(fig3a)
        print(f"Saved: {filepath3a}")
    except Exception as e:
        print(f"  Error in subpanel 3a: {e}")
        import traceback
        traceback.print_exc()
    
    # Subpanel 5: Angle plots for each FP
    print("\nSubpanel 5: Angle plots for each FP...")
    try:
        fig5, ax5 = subpanel_5(row_dict)
        filepath5 = os.path.join(output_dir, "subpanel_5_angle_plots.png")
        fig5.savefig(filepath5, dpi=300, bbox_inches='tight')
        plt.close(fig5)
        print(f"Saved: {filepath5}")
    except Exception as e:
        print(f"  Error in subpanel 5: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Figure 5 subpanels generation complete!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Generate and save all subpanels
    save_all_subpanels()

