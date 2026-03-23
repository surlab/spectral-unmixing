"""
Figure 2 generation for spectral unmixing methods paper.

This module generates Figure 2, which shows spectral unmixing for 4 fluorophores
(GCamp, TdTomato, mCherry, mNeptune) across multiple excitation wavelengths and emission filters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import os
import glob
import tifffile as tf
from src import config as cfg
from src.figure1 import (
    load_2p_spectra,
    plot_2p_excitation_spectra,
    plot_1p_emission_spectra,
    apply_smoothing_to_spectrum,
    load_filter_transmission,
    load_channel_data,
    compute_predicted_channel_signals,
    _get_all_acquisition_pairs,
    extract_pockels_from_filename,
    find_image_folder
)
from src.figure5 import (
    compute_predicted_signals_figure5
)
import src.figure5 as fig5_module
from src.figure_scatterplot_helpers import (
    compute_data_vector,
    vector_angle,
    classify_pixel_by_angle,
    load_2p_spectra_flexible
)

# Figure 2 fluorophores - single source of truth
# Note: GCamp is only used in Figure 3, not Figure 2
FIG_2_Fluorophores = ["TdTomato", "mCherry", "mNeptune"]

# Figure 2 main row dictionary (flexible format like Figure 5)
FIG_2_ROW_DICT = {
    "name": "fig_2",
    "Fluorophores": FIG_2_Fluorophores,
    "Excitation wavelengths": [800, 1040, 1080, 1180, 1240],  # Added 1080nm (sorted)
    "emission filters": [[550, 580], [590, 620], [645, 695]]  # Removed [500, 550] (GCamp filter)
}

# Figure 2 best channel row dictionary (for subpanels 2.2, 4, 5, 6)
# ch1: 1040, orange = [550,580]
# ch2: 1180, red = [590,620] (note: description says 1040 but best chan says 1180, using best chan)
# ch3: 1240, far red = [645,695]
FIG_2_BEST_CHAN_ROW_DICT = {
    "name": "fig_2_best_chan",
    "Fluorophores": FIG_2_Fluorophores,
    "Channel 1": {
        "Excitation wavelength": 1040,
        "emission filter": [550, 580]  # Orange filter
    },
    "Channel 2": {
        "Excitation wavelength": 1180,
        "emission filter": [590, 620]  # Red filter
    },
    "Channel 3": {
        "Excitation wavelength": 1240,
        "emission filter": [645, 695]  # Far red filter
    }
}

# Row dicts for subpanel 8.2 subsets
# Subset 1: all 20mW broad
FIG_2_SUBSET_1_ROW_DICT = {
    "name": "fig_2_subset_1_all_20mW_broad",
    "Fluorophores": FIG_2_Fluorophores,
    "Excitation wavelengths": [800, 1040, 1180, 1240],
    "emission filters": [[500, 700]]  # Broad filter (all wavelengths)
}

# Subset 2: best row dict (same as best chan)
FIG_2_SUBSET_2_ROW_DICT = FIG_2_BEST_CHAN_ROW_DICT.copy()
FIG_2_SUBSET_2_ROW_DICT["name"] = "fig_2_subset_2_best_chan"

# Subset 3: 1080 + orange, red, far red (emission based)
FIG_2_SUBSET_3_ROW_DICT = {
    "name": "fig_2_subset_3_1080_emission_based",
    "Fluorophores": FIG_2_Fluorophores,
    "Excitation wavelengths": [1080],
    "emission filters": [[550, 580], [590, 620], [645, 695]]  # Orange, red, far red
}

# Subset 4: 800nm red, far red, 1040 red, far red
FIG_2_SUBSET_4_ROW_DICT = {
    "name": "fig_2_subset_4_mixed",
    "Fluorophores": FIG_2_Fluorophores,
    "Excitation wavelengths": [800, 1040],
    "emission filters": [[590, 620], [645, 695]]  # Red, far red
}

# Fluorophore colors for Figure 2
# Note: GCamp is only used in Figure 3, not Figure 2
FIG_2_FP_COLORS = {
    "TdTomato": "#B8860B",   # dark yellow (dark goldenrod, matching fig 5 filter shading)
    "mCherry": "#E31A1C",    # red (same as fig 1)
    "mNeptune": "#4B0082"    # purple (same as fig 1)
}

# Map emission filter ranges to filter names (for data loading)
# Orange filter: [550, 580] -> "Orange"
# Red filter: [590, 620] -> "Red"  
# Far red filter: [645, 695] -> "FarRed"
FILTER_RANGE_TO_NAME = {
    (550, 580): "Orange",  # Orange filter (TdTomato range)
    (590, 620): "Red",  # Red filter (mCherry range)
    (645, 695): "FarRed"  # Far red filter (mNeptune range)
}


def load_figure2_2p_spectra(fluorophore_name, spectra_dir=None):
    """
    Load 2P excitation spectra for Figure 2 fluorophores.
    
    Uses the flexible loading function from helpers which handles case-insensitive
    column names and various naming variations.
    
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
    # For now, GCamp is not available (saved for fig3)
    if fluorophore_name == "GCamp":
        # Placeholder - will be implemented for fig3
        raise ValueError("GCamp spectra not yet available (saved for fig3)")
    
    # Use the flexible loading function from helpers
    # This handles case-insensitive column names and various naming variations
    return load_2p_spectra_flexible(fluorophore_name, spectra_dir)


def subpanel_1(row_dict=None, ax=None, wavelength_range=(780, 1250)):
    """
    Generate subpanel 1: 2P excitation spectra with excitation wavelengths.
    
    Same as Figure 1 subpanel 1 but adds TdTomato, 800, 1040, and 1180 excitation,
    and orange filter (550-580).
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses FIG_2_ROW_DICT
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    wavelength_range : tuple of float, optional
        Wavelength range to plot (min, max). Default (780, 1250)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = FIG_2_ROW_DICT
    
    fluorophores = row_dict["Fluorophores"]
    excitation_wavelengths = row_dict["Excitation wavelengths"]
    
    # Filter out GCamp if not available
    available_fluorophores = []
    for fp in fluorophores:
        if fp == "GCamp":
            print("Warning: Skipping GCamp (saved for fig3)")
            continue
        available_fluorophores.append(fp)
    
    if len(available_fluorophores) == 0:
        raise ValueError("No available fluorophores for subpanel 1")
    
    # Create channel labels for excitation wavelengths
    channel_labels = [f"{wl} nm" for wl in excitation_wavelengths]
    
    # Temporarily override cfg.fluorophore_colors to use Figure 2 colors
    original_colors = {}
    for fp in available_fluorophores:
        if fp in cfg.fluorophore_colors:
            original_colors[fp] = cfg.fluorophore_colors[fp]
        if fp in FIG_2_FP_COLORS:
            cfg.fluorophore_colors[fp] = FIG_2_FP_COLORS[fp]
    
    try:
        fig, ax = plot_2p_excitation_spectra(
            fluorophore_names=available_fluorophores,
            excitation_wavelengths=excitation_wavelengths,
            channel_labels=channel_labels,
            wavelength_range=wavelength_range,
            smoothing_std=5,
            ax=ax,
            load_spectra_func=load_2p_spectra_flexible  # Use flexible loader for TdTomato support
        )
        
        # Update labeling to match Figure 5 style: text labels above lines (not in legend)
        # Remove excitation lines from legend and add text labels above them
        # Get current legend handles and filter out excitation line handles
        current_legend = ax.get_legend()
        if current_legend is not None:
            # Remove the legend temporarily to rebuild it
            current_legend.remove()
        
        # Re-add excitation wavelength lines with text labels above (like Figure 5)
        # Custom color mapping for Figure 2: 1080nm=black, 1040nm=yellow, 800nm=red
        wavelength_color_map = {
            800: '#FF0000',   # Red (for red+purple, using red as primary)
            1040: '#FFFF00',  # Yellow
            1080: '#000000',  # Black
            1180: cfg.excitation_line_colors[0] if len(cfg.excitation_line_colors) > 0 else '#808080',  # Default
            1240: cfg.excitation_line_colors[1] if len(cfg.excitation_line_colors) > 1 else '#808080'   # Default
        }
        
        for idx, wl in enumerate(excitation_wavelengths):
            if wavelength_range[0] <= wl <= wavelength_range[1]:
                # Get color from custom mapping, fallback to config
                line_color = wavelength_color_map.get(wl, cfg.excitation_line_colors[idx % len(cfg.excitation_line_colors)])
                line_style = cfg.excitation_line_styles[idx % len(cfg.excitation_line_styles)]
                
                # Draw the line (if not already drawn, or redraw to ensure it's visible)
                ax.axvline(
                    wl,
                    color=line_color,
                    linestyle=line_style,
                    linewidth=3,
                    alpha=0.7
                )
                # Add text label above the line (like Figure 5)
                ax.text(wl, ax.get_ylim()[1] * 0.95 + 0.03, f'{wl}nm', 
                       ha='center', va='bottom', fontsize=16, rotation=0,  # 2x larger as requested
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Re-add legend with FP spectra color patches (not excitation lines) at bottom center
        # Create patches with fluorophore names directly
        from matplotlib.patches import Patch
        legend_patches = []
        legend_labels = []
        
        # Get colors and names from available_fluorophores
        for fp_name in available_fluorophores:
            color = FIG_2_FP_COLORS.get(fp_name, cfg.fluorophore_colors.get(fp_name, "#808080"))
            legend_patches.append(Patch(facecolor=color, alpha=0.3))
            legend_labels.append(fp_name)  # Use fluorophore name as label
        
        if legend_patches:
            ax.legend(handles=legend_patches, labels=legend_labels, loc='lower center', fontsize=9, ncol=len(legend_patches))
        
        # Update title: raise by 0.07 and make 1.5x larger
        current_title = ax.get_title()
        if current_title:
            current_fontsize = ax.title.get_fontsize()
            new_fontsize = current_fontsize * 1.5
            ax.set_title(current_title, fontsize=new_fontsize, y=1.0 + 0.07)  # Raise by 0.07 (not 0.7)
    finally:
        # Restore original colors
        for fp, color in original_colors.items():
            cfg.fluorophore_colors[fp] = color
    
    return fig, ax


def subpanel_2(row_dict=None, ax=None, wavelength_range=(500, 700)):
    """
    Generate subpanel 2: 1P emission spectra overlaid with emission filters.
    
    Uses the elegant function from Figure 1, but with TdTomato color from Figure 5.
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses FIG_2_ROW_DICT
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    wavelength_range : tuple of float, optional
        Wavelength range to plot (min, max). Default (500, 700)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = FIG_2_ROW_DICT
    
    fluorophores = row_dict["Fluorophores"]
    emission_filters = row_dict["emission filters"]
    
    # Filter out GCamp if not available
    available_fluorophores = []
    for fp in fluorophores:
        if fp == "GCamp":
            print("Warning: Skipping GCamp (saved for fig3)")
            continue
        available_fluorophores.append(fp)
    
    if len(available_fluorophores) == 0:
        raise ValueError("No available fluorophores for subpanel 2")
    
    # Convert filter ranges to filter names (like Figure 1 uses)
    # Map filter ranges to filter names from config
    filter_range_to_name = {
        (550, 580): "Orange",
        (590, 620): "Red",
        (645, 695): "FarRed"
    }
    
    # Filter out GCamp filter range (500-550) and convert to filter names
    gcamp_filter_range = [500, 550]
    filter_names = []
    channel_labels = []
    for filter_range in emission_filters:
        if isinstance(filter_range, list) and len(filter_range) == 2:
            # Skip GCamp filter range
            if filter_range == gcamp_filter_range:
                continue
            # Convert to filter name
            filter_key = tuple(filter_range)
            if filter_key in filter_range_to_name:
                filter_name = filter_range_to_name[filter_key]
                filter_names.append(filter_name)
                # Get display name from config
                display_name = cfg.filter_display_names.get(filter_name, filter_name)
                channel_labels.append(display_name)
            else:
                # Fallback: use range as-is if not in mapping
                filter_names.append(filter_range)
        else:
            filter_names.append(filter_range)
    
    # Temporarily override cfg.fluorophore_colors to use Figure 2 colors
    # But use TdTomato color from Figure 5 (as requested)
    from src.figure5 import FIG_5_FP_COLORS
    original_colors = {}
    for fp in available_fluorophores:
        if fp in cfg.fluorophore_colors:
            original_colors[fp] = cfg.fluorophore_colors[fp]
        # Use Figure 2 colors, except TdTomato uses Figure 5 color
        if fp == "TdTomato":
            cfg.fluorophore_colors[fp] = FIG_5_FP_COLORS["TdTomato"]  # "#FFD700" from Figure 5
        elif fp in FIG_2_FP_COLORS:
            cfg.fluorophore_colors[fp] = FIG_2_FP_COLORS[fp]
    
    try:
        # Use the elegant function from Figure 1
        # Pass filter names (like "Orange", "Red", "FarRed") instead of ranges
        # X-axis should start at 525 (override wavelength_range if needed)
        adjusted_wavelength_range = (525, wavelength_range[1])
        fig, ax = plot_1p_emission_spectra(
            fluorophore_names=available_fluorophores,
            emission_filters=filter_names,  # Pass filter names (like Figure 1)
            channel_labels=channel_labels,  # Pass display names
            wavelength_range=adjusted_wavelength_range,  # Start at 525
            smoothing_std=10,
            ax=ax,
            use_display_ranges=True,  # Use display ranges for visualization
            load_spectra_func=load_2p_spectra_flexible  # Use flexible loader
        )
    finally:
        # Restore original colors
        for fp, color in original_colors.items():
            cfg.fluorophore_colors[fp] = color
    
    return fig, ax


def compute_predicted_signals_figure2(row_dict, power_mw=20.0):
    """
    Compute predicted signals for all channel combinations in Figure 2.
    
    Similar to compute_predicted_signals_figure5 but adapted for Figure 2 format.
    
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
        where channel_key is a string like "1040nm_[550,580]"
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
        if fp_name == "GCamp":
            continue  # Skip GCamp for now
        try:
            df_raw = load_figure2_2p_spectra(fp_name)
            # Apply 5nm Gaussian smoothing to excitation spectra
            df_smoothed = apply_smoothing_to_spectrum(df_raw, smoothing_std=5)
            spectra_dict[fp_name] = df_smoothed
        except:
            continue
    
    # Process each channel combination
    for exc_wl in excitation_wavelengths:
        for filter_range in emission_filters:
            filter_min, filter_max = filter_range
            channel_key = f"{exc_wl}nm_{filter_range}"
            
            # Compute signal for each fluorophore
            for fp_name in fluorophores:
                if fp_name not in spectra_dict:
                    signals[fp_name][channel_key] = 0
                    continue
                
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
                
                # For Figure 2, assume 100% transmission within filter range (no filter transmission file)
                # Integrate: multiply emission by wavelength spacing, then sum
                filtered_emission = (emission_in_range["Emission"].values * spacings).sum()
                
                # Predicted signal = excitation * filtered emission * power_factor
                signal = exc_value * filtered_emission * power_factor
                signals[fp_name][channel_key] = signal
    
    return signals


def subpanel_2_1(row_dict=None, ax=None, min_signal_threshold=0.2):
    """
    Generate subpanel 2.1: Emission ratios like Figure 5 subpanel 3.
    
    Shows predicted unmixing ratios for all channel combinations.
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses FIG_2_ROW_DICT
    ax : matplotlib.axes.Axes or array, optional
        Axes to plot on (will create subplots if None)
    min_signal_threshold : float
        Minimum signal threshold. Channels where all FPs are below this are filtered out.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = FIG_2_ROW_DICT
    
    # Convert Figure 2 row dict format to Figure 5 format for reuse
    # Filter out GCamp
    fluorophores = [fp for fp in row_dict["Fluorophores"] if fp != "GCamp"]
    
    if len(fluorophores) == 0:
        raise ValueError("No available fluorophores for subpanel 2.1")
    
    # Create Figure 5-style row dict
    fig5_style_dict = {
        "Fluorophores": fluorophores,
        "Excitation wavelengths": row_dict["Excitation wavelengths"],
        "emission filters": row_dict["emission filters"]
    }
    
    # Use Figure 5's subpanel 3 but with Figure 2 colors
    # We'll need to temporarily override colors
    original_colors = {}
    for fp in fluorophores:
        if fp in FIG_2_FP_COLORS:
            # Temporarily add to FIG_5_FP_COLORS if not present
            from src.figure5 import FIG_5_FP_COLORS
            original_colors[fp] = FIG_5_FP_COLORS.get(fp)
            FIG_5_FP_COLORS[fp] = FIG_2_FP_COLORS[fp]
    
    try:
        # Call subpanel_3 with legend enabled for Figure 2.1
        # Use same font size as subpanel_3a but adjust position further right
        fig, axes = fig5_module.subpanel_3(fig5_style_dict, ax=ax, min_signal_threshold=min_signal_threshold,
                                           legend_fontsize=int(18*0.75), legend_bbox_x=1.35)
        
        # Adjust figure width to make bars 3x narrower (make x-axis 3x narrower)
        # Get actual number of channels from x-axis ticks
        x_ticks = axes[-1].get_xticks()
        n_channels = len([t for t in x_ticks if not np.isnan(t)])
        
        # For Figure 2, we want bars to be 3x narrower
        # Current figure width is 20 (from subpanel_3 which uses figsize_width=20)
        # To make bars 3x narrower, we need to reduce figure width by 3x
        # But we also need to maintain aspect ratio, so we adjust both width and x-axis limits
        
        if n_channels > 0:
            # Get current figure size
            current_width, current_height = fig.get_size_inches()
            
            # Reduce width by 3x to make bars appear 3x narrower
            new_width = current_width / 3.0
            fig.set_size_inches(new_width, current_height)
            
            # Remove all padding - set x-axis limits to exactly match the bars
            # Bars are at positions 0 to n_channels-1, so set limits to match exactly
            for ax_sub in axes:
                ax_sub.set_xlim(-0.5, n_channels - 0.5)  # Standard bar chart limits (no extra padding)
            
            # Re-adjust layout after size change
            plt.tight_layout()
    finally:
        # Restore original colors
        from src.figure5 import FIG_5_FP_COLORS
        for fp, color in original_colors.items():
            if color is None:
                FIG_5_FP_COLORS.pop(fp, None)
            else:
                FIG_5_FP_COLORS[fp] = color
    
    return fig, axes


def subpanel_2_2(row_dict=None, ax=None):
    """
    Generate subpanel 2.2: Emission ratios for best channels only (like Figure 5 subpanel 3a).
    
    Just use the row dict with these 3:
    - orange filter, 1040 nm
    - red filter, 1180 nm  
    - far red filter, 1240 nm
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses FIG_2_BEST_CHAN_ROW_DICT
    ax : matplotlib.axes.Axes or array, optional
        Axes to plot on (will create subplots if None)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = FIG_2_BEST_CHAN_ROW_DICT
    
    # Convert best chan row dict to Figure 5 format
    # Extract channels and convert to excitation wavelengths + emission filters
    fluorophores = [fp for fp in row_dict["Fluorophores"] if fp != "GCamp"]
    
    if len(fluorophores) == 0:
        raise ValueError("No available fluorophores for subpanel 2.2")
    
    excitation_wavelengths = []
    emission_filters = []
    
    for channel_key in ["Channel 1", "Channel 2", "Channel 3"]:
        if channel_key in row_dict:
            ch_config = row_dict[channel_key]
            exc_wl = ch_config["Excitation wavelength"]
            filter_range = ch_config["emission filter"]
            if exc_wl not in excitation_wavelengths:
                excitation_wavelengths.append(exc_wl)
            if filter_range not in emission_filters:
                emission_filters.append(filter_range)
    
    # Create Figure 5-style row dict
    fig5_style_dict = {
        "Fluorophores": fluorophores,
        "Excitation wavelengths": excitation_wavelengths,
        "emission filters": emission_filters
    }
    
    # Use Figure 5's subpanel 3a but with Figure 2 colors
    original_colors = {}
    for fp in fluorophores:
        if fp in FIG_2_FP_COLORS:
            from src.figure5 import FIG_5_FP_COLORS
            original_colors[fp] = FIG_5_FP_COLORS.get(fp)
            FIG_5_FP_COLORS[fp] = FIG_2_FP_COLORS[fp]
    
    try:
        # Adjust figure width to make bars 3x narrower (make x-axis 3x narrower)
        # We'll call subpanel_3a first, then adjust
        
        # Call subpanel_3a with legend position adjusted much further right for Figure 2.2
        # Since we'll reduce figure width by 3x, we need legend much further right
        fig, axes = fig5_module.subpanel_3a(fig5_style_dict, ax=ax,
                                            legend_fontsize=int(18*0.75*3), legend_bbox_x=2.5)
        
        # Get actual number of channels from x-axis ticks
        x_ticks = axes[-1].get_xticks()
        n_channels = len([t for t in x_ticks if not np.isnan(t)])
        
        # For Figure 2, we want bars to be 3x narrower
        # Current figure width is 12 (from subpanel_3a which uses figsize_width=12)
        # To make bars 3x narrower, we need to reduce figure width by 3x
        
        if n_channels > 0:
            # Get current figure size
            current_width, current_height = fig.get_size_inches()
            
            # Reduce width by 3x to make bars appear 3x narrower
            new_width = current_width / 3.0
            fig.set_size_inches(new_width, current_height)
            
            # Remove all padding - set x-axis limits to exactly match the bars
            # Bars are at positions 0 to n_channels-1, so set limits to match exactly
            for ax_sub in axes:
                ax_sub.set_xlim(-0.5, n_channels - 0.5)  # Standard bar chart limits (no extra padding)
            
            # Adjust subplots_adjust to make more space for legend on the right
            # Since figure is now 3x narrower, we need more right space for legend
            plt.subplots_adjust(hspace=0, right=0.50)  # More space for legend
            
            # Re-adjust layout after size change
            plt.tight_layout()
    finally:
        # Restore original colors
        from src.figure5 import FIG_5_FP_COLORS
        for fp, color in original_colors.items():
            if color is None:
                FIG_5_FP_COLORS.pop(fp, None)
            else:
                FIG_5_FP_COLORS[fp] = color
    
    return fig, axes


def subpanel_3(row_dict=None, ax=None, data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619"):
    """
    Generate subpanel 3: Actual vs theoretical angle scatterplot.
    
    Similar to Figure 1 subpanel 9.3 but includes TdTomato and colors by all 4 fluorophores.
    Each point is 1 FP, 2 acquisitions.
    
    Parameters
    ----------
    row_dict : dict, optional
        Row configuration dictionary. If None, uses FIG_2_ROW_DICT
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    data_dir : str
        Path to data directory
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Get all acquisition pairs (including TdTomato, not excluding it)
    # For Figure 2, we need to use different wavelengths and filters
    print("Subpanel 3: Finding all acquisition pairs (including TdTomato)...")
    
    # Figure 2 uses different excitation wavelengths and filters
    # Include 1080 to match Figure 1 pairs, plus Figure 2 specific ones: [800, 1040, 1080, 1180, 1240]
    # Filters: ['BR2', 'Orange', 'Red', 'FarRed'] (includes all Figure 1 filters)
    fig2_excitation_wls = [800, 1040, 1080, 1180, 1240]  # Added 1080 to include Figure 1 pairs
    fig2_filters = ['BR2', 'Orange', 'Red', 'FarRed']
    
    all_pairs = _get_all_acquisition_pairs(
        data_dir, 
        exclude_fluorophores=None, 
        avoid_bidirectional=True,
        excitation_wls=fig2_excitation_wls,
        filters=fig2_filters
    )
    print(f"  Found {len(all_pairs)} valid acquisition pairs")
    
    # Debug: Check which fluorophores are in the pairs
    fp_in_pairs = set()
    for pair in all_pairs:
        fp_in_pairs.add(pair['fp1'])
        fp_in_pairs.add(pair['fp2'])
    print(f"  Fluorophores in pairs: {sorted(fp_in_pairs)}")
    
    actual_angles = []
    predicted_angles = []
    fp_names_list = []
    
    # Use Figure 2 colors for all fluorophores
    # Map various name variations to standard names
    fp_colors = {
        'GCamp': FIG_2_FP_COLORS.get('GCamp', '#00FF00'),
        'TdTomato': FIG_2_FP_COLORS.get('TdTomato', '#B8860B'),
        'tdTomato': FIG_2_FP_COLORS.get('TdTomato', '#B8860B'),  # lowercase variant
        'TDTomato': FIG_2_FP_COLORS.get('TdTomato', '#B8860B'),  # all caps variant
        'mCherry': FIG_2_FP_COLORS.get('mCherry', '#E31A1C'),
        'mNeptune': FIG_2_FP_COLORS.get('mNeptune', '#4B0082')
    }
    
    # Brightness filtering: stricter criterion - at least one fluorophore needs 3000 pixels > 500
    brightness_threshold = 500
    min_pixels_above_threshold = 3000  # Stricter: 3000 pixels above threshold
    
    for pair_idx, pair in enumerate(all_pairs):
        if (pair_idx + 1) % 10 == 0:
            print(f"  Processing pair {pair_idx + 1}/{len(all_pairs)}...")
        
        fp1 = pair['fp1']
        fp2 = pair['fp2']
        ch1_wl = pair['ch1_wl']
        ch1_filter = pair['ch1_filter']
        ch2_wl = pair['ch2_wl']
        ch2_filter = pair['ch2_filter']
        fp1_ch1 = pair['fp1_ch1']
        fp1_ch2 = pair['fp1_ch2']
        fp2_ch1 = pair['fp2_ch1']
        fp2_ch2 = pair['fp2_ch2']
        
        # Check brightness for fp1: at least one channel needs many bright pixels
        fp1_ch1_bright = np.sum(fp1_ch1 > brightness_threshold)
        fp1_ch2_bright = np.sum(fp1_ch2 > brightness_threshold)
        fp1_has_bright = (fp1_ch1_bright >= min_pixels_above_threshold or 
                         fp1_ch2_bright >= min_pixels_above_threshold)
        
        # Check brightness for fp2: at least one channel needs many bright pixels
        fp2_ch1_bright = np.sum(fp2_ch1 > brightness_threshold)
        fp2_ch2_bright = np.sum(fp2_ch2 > brightness_threshold)
        fp2_has_bright = (fp2_ch1_bright >= min_pixels_above_threshold or 
                         fp2_ch2_bright >= min_pixels_above_threshold)
        
        # Skip pair if neither fluorophore has enough bright pixels
        if not fp1_has_bright and not fp2_has_bright:
            continue
        
        # Compute data vectors
        data_vectors = {}
        try:
            data_vectors[fp1] = compute_data_vector(fp1_ch1, fp1_ch2)
            data_vectors[fp2] = compute_data_vector(fp2_ch1, fp2_ch2)
        except:
            continue  # Skip if can't compute vectors
        
        if len(data_vectors) != 2:
            continue
        
        # Create a row_dict-like structure for computing predicted vectors
        row_dict_like = {
            'Fluorophores': [fp1, fp2],
            'Channel 1': {'Excitation wavelength': ch1_wl, 'emission filter': ch1_filter},
            'Channel 2': {'Excitation wavelength': ch2_wl, 'emission filter': ch2_filter}
        }
        
        # Get predicted vectors - use actual Pockels values from the pair
        # compute_predicted_channel_signals will convert Pockels to power using the calibration table
        # and apply the gain factor (power/20)^2 for 2P excitation
        ch1_pockels = pair.get('ch1_pockels')  # Use actual Pockels value from pair
        ch2_pockels = pair.get('ch2_pockels')  # Use actual Pockels value from pair
        
        try:
            # Use flexible spectra loader for TdTomato and other fluorophores
            from src.figure_scatterplot_helpers import load_2p_spectra_flexible
            print(f"DEBUG subpanel_3: Computing predicted signals for pair: {fp1} vs {fp2}")
            print(f"DEBUG: ch1={ch1_wl}nm {ch1_filter}, ch2={ch2_wl}nm {ch2_filter}")
            print(f"DEBUG: row_dict_like fluorophores: {row_dict_like['Fluorophores']}")
            predicted_signals = compute_predicted_channel_signals(
                row_dict_like, 
                ch1_pockels=ch1_pockels, 
                ch2_pockels=ch2_pockels,
                load_spectra_func=load_2p_spectra_flexible
            )
            print(f"DEBUG: Successfully computed predicted signals. Keys: {list(predicted_signals.keys())}")
            
            predicted_vectors = {}
            for fp_name in [fp1, fp2]:
                print(f"DEBUG: Processing predicted vector for '{fp_name}'")
                if fp_name not in predicted_signals:
                    print(f"ERROR: '{fp_name}' not in predicted_signals. Available: {list(predicted_signals.keys())}")
                    raise KeyError(f"'{fp_name}' not in predicted_signals")
                ch1_signal = predicted_signals[fp_name]["Channel 1"]
                ch2_signal = predicted_signals[fp_name]["Channel 2"]
                print(f"DEBUG: '{fp_name}' signals - ch1: {ch1_signal:.6f}, ch2: {ch2_signal:.6f}")
                pred_vec = np.array([ch1_signal, ch2_signal])
                pred_vec = pred_vec / np.linalg.norm(pred_vec)
                predicted_vectors[fp_name] = pred_vec
                print(f"DEBUG: '{fp_name}' normalized vector: {pred_vec}")
        except Exception as e:
            print(f"ERROR computing predicted signals for pair {fp1} vs {fp2}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # For each FP, compute angle to nearest axis
        for fp_name in [fp1, fp2]:
            # Actual angle (data vector)
            vec_data = data_vectors[fp_name]
            angle_rad_data = np.arctan2(vec_data[1], vec_data[0])
            angle_deg_data = np.degrees(angle_rad_data)
            angle_deg_data = np.abs(angle_deg_data) % 180
            angle_deg_data = min(angle_deg_data, 180 - angle_deg_data)  # Angle to nearest axis
            
            # Predicted angle (predicted vector)
            vec_pred = predicted_vectors[fp_name]
            angle_rad_pred = np.arctan2(vec_pred[1], vec_pred[0])
            angle_deg_pred = np.degrees(angle_rad_pred)
            angle_deg_pred = np.abs(angle_deg_pred) % 180
            angle_deg_pred = min(angle_deg_pred, 180 - angle_deg_pred)  # Angle to nearest axis
            
            actual_angles.append(angle_deg_data)
            predicted_angles.append(angle_deg_pred)
            fp_names_list.append(fp_name)
    
    # Debug: Count points per FP
    fp_counts = {}
    for fp_name in fp_names_list:
        fp_counts[fp_name] = fp_counts.get(fp_name, 0) + 1
    print(f"  Points per fluorophore: {fp_counts}")
    
    # Plot points: all colored by FP
    for i, (actual, predicted, fp_name) in enumerate(zip(actual_angles, predicted_angles, fp_names_list)):
        # Get color for this FP (use Figure 2 color if available, otherwise gray)
        # Try exact match first, then case-insensitive
        color = fp_colors.get(fp_name, None)
        if color is None:
            # Try case-insensitive match
            for key, val in fp_colors.items():
                if key.lower() == fp_name.lower():
                    color = val
                    break
        if color is None:
            color = 'gray'
        ax.scatter(actual, predicted, color=color, s=30, alpha=0.5, zorder=1)
    
    # Plot diagonal line (perfect agreement, no label)
    if len(actual_angles) > 0:
        max_angle = max(max(actual_angles), max(predicted_angles))
        ax.plot([0, max_angle], [0, max_angle], 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel("Actual Angle (degrees)", fontsize=12)
    ax.set_ylabel("Predicted Angle (degrees)", fontsize=12)
    ax.set_title("Subpanel 3: Actual vs Predicted Angle", fontsize=12, fontweight='bold')
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # Create legend with FP colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=fp_colors.get(fp, 'gray'), 
              markersize=8, markeredgecolor='black', markeredgewidth=1, label=fp)
        for fp in ['TdTomato', 'mCherry', 'mNeptune'] if fp in fp_colors
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return fig, ax


def _make_3channel_rgb_overlay(ch1_image, ch2_image, ch3_image, 
                                ch1_color_hex, ch2_color_hex, ch3_color_hex, 
                                norm_percentile=None):
    """
    Create an RGB overlay by tinting three channels and adding them together.
    
    Parameters
    ----------
    ch1_image : ndarray
        Channel 1 image (2D)
    ch2_image : ndarray
        Channel 2 image (2D)
    ch3_image : ndarray
        Channel 3 image (2D)
    ch1_color_hex : str
        Hex color for channel 1 (e.g., "#FFFF00" for yellow)
    ch2_color_hex : str
        Hex color for channel 2 (e.g., "#FF0000" for red)
    ch3_color_hex : str
        Hex color for channel 3 (e.g., "#0000FF" for blue)
    norm_percentile : float, optional
        If None, no normalization (preserve ratios). If provided, normalize each channel
        to this percentile (default: None, no normalization)
        
    Returns
    -------
    ndarray
        RGB image array (height, width, 3) with values in [0, 1]
    """
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    ch1_rgb = hex_to_rgb(ch1_color_hex)
    ch2_rgb = hex_to_rgb(ch2_color_hex)
    ch3_rgb = hex_to_rgb(ch3_color_hex)

    # No normalization - preserve the actual ratio between channels
    if norm_percentile is None:
        # Convert to float and scale to 0-1 range based on data type
        ch1_float = ch1_image.astype(np.float64)
        ch2_float = ch2_image.astype(np.float64)
        ch3_float = ch3_image.astype(np.float64)
        
        # Scale to 0-1 range but preserve relative intensities between channels
        # Use the maximum value across all channels to preserve ratios
        max_val = max(np.max(ch1_float), np.max(ch2_float), np.max(ch3_float), 1.0)
        ch1_norm = ch1_float / max_val
        ch2_norm = ch2_float / max_val
        ch3_norm = ch3_float / max_val
    else:
        # Old normalization behavior (if percentile is specified)
        denom1 = np.percentile(ch1_image, norm_percentile) if np.any(ch1_image) else 1.0
        denom2 = np.percentile(ch2_image, norm_percentile) if np.any(ch2_image) else 1.0
        denom3 = np.percentile(ch3_image, norm_percentile) if np.any(ch3_image) else 1.0
        denom1 = denom1 if denom1 > 0 else 1.0
        denom2 = denom2 if denom2 > 0 else 1.0
        denom3 = denom3 if denom3 > 0 else 1.0
        ch1_norm = np.clip(ch1_image.astype(float) / denom1, 0, 1)
        ch2_norm = np.clip(ch2_image.astype(float) / denom2, 0, 1)
        ch3_norm = np.clip(ch3_image.astype(float) / denom3, 0, 1)

    # Create RGB arrays for each channel
    rgb1 = np.zeros((ch1_image.shape[0], ch1_image.shape[1], 3), dtype=float)
    rgb1[:, :, 0] = ch1_norm * ch1_rgb[0]
    rgb1[:, :, 1] = ch1_norm * ch1_rgb[1]
    rgb1[:, :, 2] = ch1_norm * ch1_rgb[2]

    rgb2 = np.zeros((ch2_image.shape[0], ch2_image.shape[1], 3), dtype=float)
    rgb2[:, :, 0] = ch2_norm * ch2_rgb[0]
    rgb2[:, :, 1] = ch2_norm * ch2_rgb[1]
    rgb2[:, :, 2] = ch2_norm * ch2_rgb[2]

    rgb3 = np.zeros((ch3_image.shape[0], ch3_image.shape[1], 3), dtype=float)
    rgb3[:, :, 0] = ch3_norm * ch3_rgb[0]
    rgb3[:, :, 1] = ch3_norm * ch3_rgb[1]
    rgb3[:, :, 2] = ch3_norm * ch3_rgb[2]

    # Combine all three channels
    return np.clip(rgb1 + rgb2 + rgb3, 0, 1)


def _load_channel_image_for_overlay(data_dir, fp_name, excitation_wl, filter_range, channel_num=1):
    """
    Load a channel image for overlay, making max projection if it's a stack.
    
    Parameters
    ----------
    data_dir : str
        Base data directory
    fp_name : str
        Fluorophore name
    excitation_wl : int
        Excitation wavelength in nm
    filter_range : list
        Filter range [min, max] - will be converted to filter name
    channel_num : int
        Channel number (1 or 2), default 1
        
    Returns
    -------
    ndarray
        2D image (max projection if stack)
    """
    # Convert filter range to filter name
    filter_key = tuple(filter_range)
    filter_name = FILTER_RANGE_TO_NAME.get(filter_key)
    if filter_name is None:
        raise ValueError(f"Unknown filter range: {filter_range}")
    
    folder = find_image_folder(data_dir, fp_name, excitation_wl, filter_name)
    if folder is None:
        raise ValueError(f"Could not find folder for {fp_name}")
    
    # Filter names match filename prefixes exactly
    if filter_name not in ["BR2", "Red", "FarRed", "Orange"]:
        raise ValueError(f"Unknown filter name: {filter_name}. Available: BR2, Red, FarRed, Orange")
    
    filter_prefix = f"{filter_name}EmFilt"
    
    # Find aligned files matching excitation wavelength and filter
    pattern1 = os.path.join(folder, f"{filter_prefix}_{excitation_wl}nm*.tif")
    pattern2 = os.path.join(folder, f"{filter_prefix}_{excitation_wl}nm*.ome.tif")
    
    files = glob.glob(pattern1) + glob.glob(pattern2)
    files = list(set(files))  # Remove duplicates
    
    # If not found, try more flexible pattern
    if len(files) == 0:
        pattern3 = os.path.join(folder, f"*{filter_prefix}*{excitation_wl}nm*.tif")
        pattern4 = os.path.join(folder, f"*{filter_prefix}*{excitation_wl}nm*.ome.tif")
        files = glob.glob(pattern3) + glob.glob(pattern4)
        files = list(set(files))
    
    if len(files) == 0:
        all_tif_files = glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.ome.tif"))
        example_files = [os.path.basename(f) for f in all_tif_files[:10]]
        example_str = ", ".join(example_files) if example_files else "none found"
        raise ValueError(f"Could not find file for {fp_name}, {excitation_wl}nm, {filter_name} filter "
                       f"(prefix: {filter_prefix}) in {folder}. "
                       f"Example files found: {example_str}")
    
    # If multiple matches, prefer the most specific match
    if len(files) > 1:
        preferred = [f for f in files if f"{filter_prefix}_{excitation_wl}nm" in os.path.basename(f)]
        if len(preferred) > 0:
            files = preferred
        files = sorted(files)[:1]  # Take first if still multiple
    
    img = tf.imread(files[0])
    
    # Handle different image shapes and make max projection if stack
    if len(img.shape) == 3:
        if img.shape[0] < img.shape[2]:  # Likely (z, height, width) - stack
            # Make max projection along z-axis
            img = np.max(img, axis=0)
        elif img.shape[2] < img.shape[0]:  # Likely (height, width, channels)
            # Multi-channel image - extract requested channel
            if img.shape[2] >= channel_num:
                img = img[:, :, channel_num - 1]
            else:
                img = img[:, :, 0]
        else:
            # Ambiguous - assume (z, height, width) and make max projection
            img = np.max(img, axis=0)
    elif len(img.shape) == 2:
        pass  # Already 2D
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    
    return img


def _load_image_from_fig2_dir(data_dir, excitation_wl, filter_range):
    """
    Load image directly from fig2 directory structure.
    
    Files are directly in the directory with pattern:
    FilterEmFilt_ExcitationWavelengthnm_PockelsValuepoc_PMTpmt.tif
    e.g., OrangeEmFilt_1040nm_185poc_600pmt.tif
    
    Parameters
    ----------
    data_dir : str
        Path to fig2 data directory (files directly in this directory)
    excitation_wl : int
        Excitation wavelength in nm
    filter_range : list
        Filter range [min, max] - will be converted to filter name
        
    Returns
    -------
    ndarray
        2D image (max projection if stack)
    """
    # Convert filter range to filter name
    filter_key = tuple(filter_range)
    filter_name = FILTER_RANGE_TO_NAME.get(filter_key)
    if filter_name is None:
        raise ValueError(f"Unknown filter range: {filter_range}")
    
    # Filter names match filename prefixes exactly
    if filter_name not in ["BR2", "Red", "FarRed", "Orange"]:
        raise ValueError(f"Unknown filter name: {filter_name}. Available: BR2, Red, FarRed, Orange")
    
    filter_prefix = f"{filter_name}EmFilt"
    
    # Find files matching excitation wavelength and filter (directly in data_dir)
    pattern1 = os.path.join(data_dir, f"{filter_prefix}_{excitation_wl}nm*.tif")
    pattern2 = os.path.join(data_dir, f"{filter_prefix}_{excitation_wl}nm*.ome.tif")
    
    files = glob.glob(pattern1) + glob.glob(pattern2)
    files = list(set(files))  # Remove duplicates
    
    if len(files) == 0:
        all_tif_files = glob.glob(os.path.join(data_dir, "*.tif")) + glob.glob(os.path.join(data_dir, "*.ome.tif"))
        example_files = [os.path.basename(f) for f in all_tif_files[:10]]
        example_str = ", ".join(example_files) if example_files else "none found"
        raise ValueError(f"Could not find file for {excitation_wl}nm, {filter_name} filter "
                       f"(prefix: {filter_prefix}) in {data_dir}. "
                       f"Example files found: {example_str}")
    
    # If multiple matches, prefer the most specific match
    if len(files) > 1:
        preferred = [f for f in files if f"{filter_prefix}_{excitation_wl}nm" in os.path.basename(f)]
        if len(preferred) > 0:
            files = preferred
        files = sorted(files)[:1]  # Take first if still multiple
    
    img = tf.imread(files[0])
    
    # Handle different image shapes and make max projection if stack
    if len(img.shape) == 3:
        if img.shape[0] < img.shape[2]:  # Likely (z, height, width) - stack
            # Make max projection along z-axis
            img = np.max(img, axis=0)
        elif img.shape[2] < img.shape[0]:  # Likely (height, width, channels)
            # Multi-channel image - take first channel
            img = img[:, :, 0]
        else:
            # Ambiguous - assume (z, height, width) and make max projection
            img = np.max(img, axis=0)
    elif len(img.shape) == 2:
        pass  # Already 2D
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    
    return img


def subpanel_4(row_dict=None, ax=None, data_dir="data/fig2_3color_inh_spatial_control_2p3_10072025"):
    """
    Generate subpanel 4: 3-color image overlay based on best channel row dict.
    
    Creates an RGB overlay where:
    - Channel 1 (Orange/1040nm) -> Yellow
    - Channel 2 (Red/1180nm) -> Red
    - Channel 3 (FarRed/1240nm) -> Blue
    
    Makes max projections of stacks if needed.
    Uses acquisitions directly from the fig2 data directory.
    
    Parameters
    ----------
    row_dict : dict, optional
        Best channel row configuration dictionary. If None, uses FIG_2_BEST_CHAN_ROW_DICT
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    data_dir : str
        Path to fig2 data directory (files directly in this directory)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = FIG_2_BEST_CHAN_ROW_DICT
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Get channel configurations
    ch1_config = row_dict["Channel 1"]
    ch2_config = row_dict["Channel 2"]
    ch3_config = row_dict["Channel 3"]
    
    print(f"Subpanel 4: Loading images from {data_dir}...")
    
    # Load images for each channel (with max projection if stacks)
    # Files are directly in the fig2 directory, not in fluorophore subfolders
    try:
        ch1_image = _load_image_from_fig2_dir(
            data_dir,
            ch1_config["Excitation wavelength"], 
            ch1_config["emission filter"]
        )
        print(f"  Loaded Ch1 ({ch1_config['Excitation wavelength']}nm, {ch1_config['emission filter']}): {ch1_image.shape}")
        
        ch2_image = _load_image_from_fig2_dir(
            data_dir,
            ch2_config["Excitation wavelength"],
            ch2_config["emission filter"]
        )
        print(f"  Loaded Ch2 ({ch2_config['Excitation wavelength']}nm, {ch2_config['emission filter']}): {ch2_image.shape}")
        
        ch3_image = _load_image_from_fig2_dir(
            data_dir,
            ch3_config["Excitation wavelength"],
            ch3_config["emission filter"]
        )
        print(f"  Loaded Ch3 ({ch3_config['Excitation wavelength']}nm, {ch3_config['emission filter']}): {ch3_image.shape}")
    except Exception as e:
        print(f"Error loading images: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Ensure all images have the same shape
    if not (ch1_image.shape == ch2_image.shape == ch3_image.shape):
        # Resize to match the smallest dimensions
        min_h = min(ch1_image.shape[0], ch2_image.shape[0], ch3_image.shape[0])
        min_w = min(ch1_image.shape[1], ch2_image.shape[1], ch3_image.shape[1])
        ch1_image = ch1_image[:min_h, :min_w]
        ch2_image = ch2_image[:min_h, :min_w]
        ch3_image = ch3_image[:min_h, :min_w]
        print(f"  Resized all images to: {ch1_image.shape}")
    
    # Create RGB overlay with specified colors
    # Orange/1040 -> Yellow, Red/1180 -> Red, FarRed/1240 -> Blue
    # Use percentile normalization (99th percentile) to normalize each channel independently
    # This brightens dimmer channels by using the full dynamic range of each channel
    overlay_rgb = _make_3channel_rgb_overlay(
        ch1_image, ch2_image, ch3_image,
        ch1_color_hex="#FFFF00",  # Yellow for Orange/1040
        ch2_color_hex="#FF0000",  # Red for Red/1180
        ch3_color_hex="#0000FF",  # Blue for FarRed/1240
        norm_percentile=99.0  # Normalize each channel to its 99th percentile independently
    )
    
    ax.imshow(overlay_rgb)
    ax.set_title("Subpanel 4: 3-Color Overlay", fontsize=12, fontweight='bold')
    ax.axis("off")
    
    return fig, ax


def _compute_data_vector_3d(ch1_data, ch2_data, ch3_data, lower_percentile=None, upper_percentile=None):
    """
    Compute unit vector from 3D data by filtering pixels and computing mean direction.
    
    Parameters
    ----------
    ch1_data : np.ndarray
        Channel 1 pixel intensities
    ch2_data : np.ndarray
        Channel 2 pixel intensities
    ch3_data : np.ndarray
        Channel 3 pixel intensities
    lower_percentile : float, optional
        Lower percentile to filter (default from config)
    upper_percentile : float, optional
        Upper percentile to filter (default from config)
        
    Returns
    -------
    np.ndarray
        Unit vector [ch1_component, ch2_component, ch3_component]
    """
    if lower_percentile is None:
        lower_percentile = cfg.pixel_intensity_lower_percentile
    if upper_percentile is None:
        upper_percentile = cfg.pixel_intensity_upper_percentile
    
    # Filter pixels based on intensity
    ch1_lower = np.percentile(ch1_data, lower_percentile)
    ch1_upper = np.percentile(ch1_data, upper_percentile)
    ch2_lower = np.percentile(ch2_data, lower_percentile)
    ch2_upper = np.percentile(ch2_data, upper_percentile)
    ch3_lower = np.percentile(ch3_data, lower_percentile)
    ch3_upper = np.percentile(ch3_data, upper_percentile)
    
    # Keep pixels in the "middle chunk"
    mask = ((ch1_data >= ch1_lower) & (ch1_data <= ch1_upper) &
            (ch2_data >= ch2_lower) & (ch2_data <= ch2_upper) &
            (ch3_data >= ch3_lower) & (ch3_data <= ch3_upper))
    
    ch1_filtered = ch1_data[mask]
    ch2_filtered = ch2_data[mask]
    ch3_filtered = ch3_data[mask]
    
    # Compute mean direction (mean of normalized vectors)
    # IMPORTANT: cast to float64 BEFORE squaring to avoid uint16 overflow
    ch1_f = ch1_filtered.astype(np.float64)
    ch2_f = ch2_filtered.astype(np.float64)
    ch3_f = ch3_filtered.astype(np.float64)

    # Normalize each pixel vector
    magnitudes = np.sqrt(ch1_f**2 + ch2_f**2 + ch3_f**2)
    valid_mask = magnitudes > 0
    ch1_normalized = ch1_f[valid_mask] / magnitudes[valid_mask]
    ch2_normalized = ch2_f[valid_mask] / magnitudes[valid_mask]
    ch3_normalized = ch3_f[valid_mask] / magnitudes[valid_mask]
    
    # Mean normalized vector
    mean_ch1 = np.mean(ch1_normalized)
    mean_ch2 = np.mean(ch2_normalized)
    mean_ch3 = np.mean(ch3_normalized)
    
    # Normalize to unit vector
    mean_magnitude = np.sqrt(mean_ch1**2 + mean_ch2**2 + mean_ch3**2)
    if mean_magnitude > 0:
        unit_vector = np.array([mean_ch1 / mean_magnitude, mean_ch2 / mean_magnitude, mean_ch3 / mean_magnitude])
    else:
        unit_vector = np.array([1.0, 0.0, 0.0])  # Default if no valid pixels
    
    return unit_vector


def _compute_angle_3d(pixel_vec, fp_vec):
    """
    Compute angle between a pixel vector and a fluorophore vector in 3D space.
    
    Parameters
    ----------
    pixel_vec : np.ndarray
        Pixel vector [ch1, ch2, ch3]
    fp_vec : np.ndarray
        Fluorophore unit vector [ch1, ch2, ch3]
        
    Returns
    -------
    float
        Angle in degrees (0-90)
    """
    # Normalize pixel vector
    # IMPORTANT: cast to float64 BEFORE squaring to avoid uint16 overflow
    pixel_vec = np.asarray(pixel_vec, dtype=np.float64)
    fp_vec = np.asarray(fp_vec, dtype=np.float64)

    pixel_magnitude = np.sqrt(np.sum(pixel_vec**2))
    if pixel_magnitude == 0:
        return 90.0  # Return 90 degrees if pixel is at origin
    
    pixel_normalized = pixel_vec / pixel_magnitude
    
    # Compute dot product (both vectors are normalized)
    # fp_vec should already be unit, but clip for numerical stability
    dot_product = np.clip(np.dot(pixel_normalized, fp_vec), -1.0, 1.0)
    
    # Compute angle in radians, then convert to degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    # Ensure angle is in 0-90 range
    angle_deg = min(angle_deg, 180 - angle_deg)
    
    return angle_deg


def _compute_classification_zone_3d(ch1_data, ch2_data, ch3_data, fp_labels, fp_name, vector, 
                                    percentile=90, min_distance=500):
    """
    Compute symmetric angle zone in 3D that contains a given percentile of pixels.
    
    Parameters
    ----------
    ch1_data : np.ndarray
        Channel 1 pixel intensities
    ch2_data : np.ndarray
        Channel 2 pixel intensities
    ch3_data : np.ndarray
        Channel 3 pixel intensities
    fp_labels : np.ndarray
        Labels indicating which fluorophore each pixel belongs to
    fp_name : str
        Name of fluorophore to compute zone for
    vector : np.ndarray
        Reference unit vector for this fluorophore (3D)
    percentile : float
        Percentile to include (default 90)
    min_distance : float
        Minimum distance from origin to include (default 500)
        
    Returns
    -------
    float or None
        Half-angle in degrees (symmetric zone extends ±half_angle from vector)
        Returns None if insufficient data
    """
    # Filter to pixels from this fluorophore
    fp_mask = fp_labels == fp_name
    if not np.any(fp_mask):
        return None
    
    ch1_fp = ch1_data[fp_mask]
    ch2_fp = ch2_data[fp_mask]
    ch3_fp = ch3_data[fp_mask]
    
    # Filter by minimum distance
    distances = np.sqrt(ch1_fp.astype(np.float64)**2 + ch2_fp.astype(np.float64)**2 + ch3_fp.astype(np.float64)**2)
    bright_mask = distances >= min_distance
    
    if not np.any(bright_mask):
        return None
    
    ch1_bright = ch1_fp[bright_mask]
    ch2_bright = ch2_fp[bright_mask]
    ch3_bright = ch3_fp[bright_mask]
    
    # Compute angles between each pixel and the vector
    angles = []
    for i in range(len(ch1_bright)):
        pixel_vec = np.array([ch1_bright[i], ch2_bright[i], ch3_bright[i]])
        angle = _compute_angle_3d(pixel_vec, vector)
        angles.append(angle)
    
    angles = np.array(angles)
    valid_angles = angles[~np.isnan(angles)]
    
    if len(valid_angles) == 0:
        return None
    
    # Find the half-angle such that [0, half_angle] contains percentile% of pixel angles
    sorted_angles = np.sort(valid_angles)
    target_count = int(np.ceil(len(sorted_angles) * percentile / 100.0))
    
    # The half-angle is the target_count-th smallest angle
    half_angle = sorted_angles[target_count - 1] if target_count <= len(sorted_angles) else sorted_angles[-1]
    
    print(f"  compute_classification_zone_3d({fp_name}): {len(valid_angles)} valid angles")
    print(f"    angles: min={np.min(valid_angles):.2f}°, max={np.max(valid_angles):.2f}°, median={np.median(valid_angles):.2f}°")
    print(f"    target_count={target_count} out of {len(sorted_angles)}, half_angle={half_angle:.2f}°")
    
    return half_angle


def _classify_pixel_3d(ch1_val, ch2_val, ch3_val, vectors_dict, classification_zones):
    """
    Classify a pixel by finding which vector it's closest to (by angle in 3D).
    
    Parameters
    ----------
    ch1_val : float
        Channel 1 intensity
    ch2_val : float
        Channel 2 intensity
    ch3_val : float
        Channel 3 intensity
    vectors_dict : dict
        Dictionary mapping fluorophore names to unit vectors (3D)
    classification_zones : dict
        Dictionary mapping fluorophore names to half-angle thresholds
        
    Returns
    -------
    str or None
        Name of closest fluorophore within its classification zone, or None
    """
    if ch1_val == 0 and ch2_val == 0 and ch3_val == 0:
        return None
    
    pixel_vec = np.array([ch1_val, ch2_val, ch3_val])
    
    # Compute angle to each vector and check if within classification zone
    min_angle = float('inf')
    closest_fp = None
    
    for fp_name, vec in vectors_dict.items():
        angle = _compute_angle_3d(pixel_vec, vec)
        
        # Check if within classification zone
        half_angle = classification_zones.get(fp_name)
        if half_angle is not None and angle <= half_angle:
            if angle < min_angle:
                min_angle = angle
                closest_fp = fp_name
    
    return closest_fp


def _compute_fig2_classifications(row_dict=None,
                                  data_dir="data/fig2_3color_inh_spatial_control_2p3_10072025",
                                  single_fp_data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619"):
    """
    Shared function to compute pixel classifications for Figure 2 subpanels 5, 6, and 7.
    
    Loads Fig2 images, computes vectors/zones from single-FP data, and classifies ALL pixels.
    This avoids redundant computation across the three subpanels.
    
    Parameters
    ----------
    row_dict : dict, optional
        Best channel row configuration dictionary. If None, uses FIG_2_BEST_CHAN_ROW_DICT
    data_dir : str
        Path to fig2 data directory
    single_fp_data_dir : str
        Path to single fluorophore data directory
        
    Returns
    -------
    dict with keys:
        'ch1_valid', 'ch2_valid', 'ch3_valid': All valid pixel values (filtered by distance, non-zero)
        'pixel_labels': Classification for each pixel (FP name or None)
        'data_vectors_3d': Dict mapping FP names to 3D unit vectors
        'classification_zones_3d': Dict mapping FP names to half-angle thresholds
        'fluorophores': List of fluorophore names
        'ch1_config', 'ch2_config', 'ch3_config': Channel configurations
        'max_value': Maximum value for axis limits
    """
    if row_dict is None:
        row_dict = FIG_2_BEST_CHAN_ROW_DICT
    
    fluorophores = row_dict["Fluorophores"]
    ch1_config = row_dict["Channel 1"]
    ch2_config = row_dict["Channel 2"]
    ch3_config = row_dict["Channel 3"]
    
    # Load Fig2 images (same as subpanel 4/5/7)
    ch1_image = _load_image_from_fig2_dir(data_dir, ch1_config["Excitation wavelength"], ch1_config["emission filter"])
    ch2_image = _load_image_from_fig2_dir(data_dir, ch2_config["Excitation wavelength"], ch2_config["emission filter"])
    ch3_image = _load_image_from_fig2_dir(data_dir, ch3_config["Excitation wavelength"], ch3_config["emission filter"])
    
    # Ensure all images have same shape
    if not (ch1_image.shape == ch2_image.shape == ch3_image.shape):
        min_h = min(ch1_image.shape[0], ch2_image.shape[0], ch3_image.shape[0])
        min_w = min(ch1_image.shape[1], ch2_image.shape[1], ch3_image.shape[1])
        ch1_image = ch1_image[:min_h, :min_w]
        ch2_image = ch2_image[:min_h, :min_w]
        ch3_image = ch3_image[:min_h, :min_w]
    
    ch1_flat = ch1_image.flatten().astype(np.float64)
    ch2_flat = ch2_image.flatten().astype(np.float64)
    ch3_flat = ch3_image.flatten().astype(np.float64)
    
    # Background subtraction: use bottom 10% of pixels (same approach as Figure 1)
    # Combine all channels to determine which pixels are background
    combined_intensity = ch1_flat + ch2_flat + ch3_flat
    background_threshold = np.percentile(combined_intensity, 10)
    background_mask = combined_intensity <= background_threshold
    
    # Calculate background as average of bottom 10% pixels for each channel
    if np.any(background_mask):
        bg_ch1 = np.mean(ch1_flat[background_mask])
        bg_ch2 = np.mean(ch2_flat[background_mask])
        bg_ch3 = np.mean(ch3_flat[background_mask])
    else:
        # Fallback if no background pixels found
        bg_ch1 = 0.0
        bg_ch2 = 0.0
        bg_ch3 = 0.0
    
    print(f"  Background values: Ch1={bg_ch1:.2f}, Ch2={bg_ch2:.2f}, Ch3={bg_ch3:.2f}")
    
    # Subtract background and clip negative values to 0
    ch1_bg_subtracted = np.clip(ch1_flat - bg_ch1, 0, None)
    ch2_bg_subtracted = np.clip(ch2_flat - bg_ch2, 0, None)
    ch3_bg_subtracted = np.clip(ch3_flat - bg_ch3, 0, None)
    
    # Filter by distance (after background subtraction)
    max_value = 3000
    max_distance = max_value * np.sqrt(3)
    distances = np.sqrt(ch1_bg_subtracted**2 + ch2_bg_subtracted**2 + ch3_bg_subtracted**2)
    distance_mask = distances <= max_distance
    ch1_filtered = ch1_bg_subtracted[distance_mask]
    ch2_filtered = ch2_bg_subtracted[distance_mask]
    ch3_filtered = ch3_bg_subtracted[distance_mask]
    
    # Drop origin pixels
    nonzero_mask = (ch1_filtered != 0) | (ch2_filtered != 0) | (ch3_filtered != 0)
    ch1_valid = ch1_filtered[nonzero_mask]
    ch2_valid = ch2_filtered[nonzero_mask]
    ch3_valid = ch3_filtered[nonzero_mask]
    
    # Load single FP data and compute vectors/zones
    ch1_filter_name = FILTER_RANGE_TO_NAME.get(tuple(ch1_config["emission filter"]))
    ch2_filter_name = FILTER_RANGE_TO_NAME.get(tuple(ch2_config["emission filter"]))
    ch3_filter_name = FILTER_RANGE_TO_NAME.get(tuple(ch3_config["emission filter"]))
    if ch1_filter_name is None or ch2_filter_name is None or ch3_filter_name is None:
        raise ValueError("Could not convert filter ranges to names")
    
    all_ch1_data = []
    all_ch2_data = []
    all_ch3_data = []
    fp_labels = []
    for fp_name in fluorophores:
        ch1_data, _ = load_channel_data(single_fp_data_dir, fp_name, ch1_config["Excitation wavelength"], ch1_filter_name, channel_num=1, subsample_factor=None)
        ch2_data, _ = load_channel_data(single_fp_data_dir, fp_name, ch2_config["Excitation wavelength"], ch2_filter_name, channel_num=1, subsample_factor=None)
        ch3_data, _ = load_channel_data(single_fp_data_dir, fp_name, ch3_config["Excitation wavelength"], ch3_filter_name, channel_num=1, subsample_factor=None)
        all_ch1_data.append(ch1_data)
        all_ch2_data.append(ch2_data)
        all_ch3_data.append(ch3_data)
        fp_labels.extend([fp_name] * len(ch1_data))
    
    # Compute 3D vectors
    data_vectors_3d = {}
    for i, fp_name in enumerate(fluorophores):
        data_vectors_3d[fp_name] = _compute_data_vector_3d(all_ch1_data[i], all_ch2_data[i], all_ch3_data[i])
    
    # Compute classification zones from single-FP data
    ch1_combined = np.concatenate(all_ch1_data)
    ch2_combined = np.concatenate(all_ch2_data)
    ch3_combined = np.concatenate(all_ch3_data)
    fp_labels_array = np.array(fp_labels, dtype=object)
    
    distances_combined = np.sqrt(ch1_combined**2 + ch2_combined**2 + ch3_combined**2)
    combined_mask = distances_combined <= max_distance
    ch1_combined = ch1_combined[combined_mask]
    ch2_combined = ch2_combined[combined_mask]
    ch3_combined = ch3_combined[combined_mask]
    fp_labels_array = fp_labels_array[combined_mask]
    
    zone_min_distance = getattr(cfg, "classification_zone_min_distance", 500)
    classification_zones_3d = {}
    for fp_name in fluorophores:
        classification_zones_3d[fp_name] = _compute_classification_zone_3d(
            ch1_combined, ch2_combined, ch3_combined, fp_labels_array,
            fp_name, data_vectors_3d[fp_name],
            percentile=cfg.classification_zone_percentile,
            min_distance=zone_min_distance
        )
    
    # Classify ALL pixels
    pixel_labels = []
    for i in range(len(ch1_valid)):
        pixel_labels.append(_classify_pixel_3d(ch1_valid[i], ch2_valid[i], ch3_valid[i], data_vectors_3d, classification_zones_3d))
    pixel_labels = np.array(pixel_labels, dtype=object)
    
    return {
        'ch1_valid': ch1_valid,
        'ch2_valid': ch2_valid,
        'ch3_valid': ch3_valid,
        'pixel_labels': pixel_labels,
        'data_vectors_3d': data_vectors_3d,
        'classification_zones_3d': classification_zones_3d,
        'fluorophores': fluorophores,
        'ch1_config': ch1_config,
        'ch2_config': ch2_config,
        'ch3_config': ch3_config,
        'max_value': max_value
    }


def subpanel_5(row_dict=None, ax=None, 
               data_dir="data/fig2_3color_inh_spatial_control_2p3_10072025",
               single_fp_data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619",
               shared_data=None):
    """
    Generate subpanel 5: 3D scatterplot with pixel classifications, vectors, and shaded cones.
    
    Uses shared classification data if provided (from _compute_fig2_classifications),
    otherwise computes it. Colors pixels by classification, overlays vectors, and draws
    shaded cones for classification zones.
    
    Parameters
    ----------
    row_dict : dict, optional
        Best channel row configuration dictionary. If None, uses FIG_2_BEST_CHAN_ROW_DICT
    ax : matplotlib.axes.Axes, optional
        3D axes to plot on (will create if None)
    data_dir : str
        Path to fig2 data directory
    single_fp_data_dir : str
        Path to single fluorophore data directory
    shared_data : dict, optional
        Pre-computed classification data from _compute_fig2_classifications
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes (3D)
    """
    if row_dict is None:
        row_dict = FIG_2_BEST_CHAN_ROW_DICT
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Use shared data if provided, otherwise compute
    if shared_data is None:
        print("Subpanel 5: Computing classifications...")
        shared_data = _compute_fig2_classifications(row_dict, data_dir, single_fp_data_dir)
    
    ch1_valid = shared_data['ch1_valid']
    ch2_valid = shared_data['ch2_valid']
    ch3_valid = shared_data['ch3_valid']
    pixel_labels = shared_data['pixel_labels']
    data_vectors_3d = shared_data['data_vectors_3d']
    classification_zones_3d = shared_data['classification_zones_3d']
    fluorophores = shared_data['fluorophores']
    ch1_config = shared_data['ch1_config']
    ch2_config = shared_data['ch2_config']
    ch3_config = shared_data['ch3_config']
    max_value = shared_data['max_value']
    
    # Subsample for plotting (same as before)
    max_distance = max_value * np.sqrt(3)
    distances = np.sqrt(ch1_valid**2 + ch2_valid**2 + ch3_valid**2)
    bin_width = 100
    n_bins = int(np.ceil(max_distance / bin_width))
    samples_per_bin = 300
    
    ch1_plot_list = []
    ch2_plot_list = []
    ch3_plot_list = []
    labels_plot_list = []
    
    for bin_idx in range(n_bins):
        bin_max = (bin_idx + 1) * bin_width
        prev_bin_max = bin_idx * bin_width
        
        if bin_idx == 0:
            bin_mask = distances < bin_max
        else:
            bin_mask = (distances >= prev_bin_max) & (distances < bin_max)
        
        if np.any(bin_mask):
            ch1_bin = ch1_valid[bin_mask]
            ch2_bin = ch2_valid[bin_mask]
            ch3_bin = ch3_valid[bin_mask]
            labels_bin = pixel_labels[bin_mask]
            
            n_in_bin = len(ch1_bin)
            n_take = min(samples_per_bin, n_in_bin)
            
            if n_in_bin > n_take:
                indices = np.random.choice(n_in_bin, n_take, replace=False)
                ch1_plot_list.append(ch1_bin[indices])
                ch2_plot_list.append(ch2_bin[indices])
                ch3_plot_list.append(ch3_bin[indices])
                labels_plot_list.append(labels_bin[indices])
            else:
                ch1_plot_list.append(ch1_bin)
                ch2_plot_list.append(ch2_bin)
                ch3_plot_list.append(ch3_bin)
                labels_plot_list.append(labels_bin)
    
    if len(ch1_plot_list) > 0:
        ch1_plot = np.concatenate(ch1_plot_list)
        ch2_plot = np.concatenate(ch2_plot_list)
        ch3_plot = np.concatenate(ch3_plot_list)
        labels_plot = np.concatenate(labels_plot_list)
    else:
        ch1_plot = ch1_valid
        ch2_plot = ch2_valid
        ch3_plot = ch3_valid
        labels_plot = pixel_labels
    
    print(f"  Plotting {len(ch1_plot)} points")
    
    fp_colors = FIG_2_FP_COLORS
    
    # Draw shaded cones FIRST (so they appear behind points)
    vector_scale = max_value * 0.7  # Scale vectors to 70% of max_value
    for fp_name in fluorophores:
        if fp_name not in data_vectors_3d:
            continue
        vec = data_vectors_3d[fp_name]
        half_angle = classification_zones_3d.get(fp_name)
        if half_angle is None:
            continue
        
        color = fp_colors.get(fp_name, "#808080")
        
        # Draw cone: create a mesh of points within the cone
        # Cone extends from origin along vector, with half_angle opening
        n_radial = 20
        n_angular = 20
        cone_length = vector_scale
        
        # Generate points on cone surface
        angles_rad = np.linspace(0, np.radians(half_angle), n_angular)
        radii = np.linspace(0, cone_length, n_radial)
        
        # Create meshgrid
        R, Theta = np.meshgrid(radii, angles_rad)
        
        # For each angle, create a circle perpendicular to the vector
        # We need to find two perpendicular vectors to the main vector
        vec_normalized = vec / np.linalg.norm(vec)
        
        # Find a perpendicular vector (arbitrary choice)
        if abs(vec_normalized[0]) < 0.9:
            perp1 = np.array([1, 0, 0]) - np.dot(np.array([1, 0, 0]), vec_normalized) * vec_normalized
        else:
            perp1 = np.array([0, 1, 0]) - np.dot(np.array([0, 1, 0]), vec_normalized) * vec_normalized
        perp1 = perp1 / np.linalg.norm(perp1)
        
        # Second perpendicular vector (cross product)
        perp2 = np.cross(vec_normalized, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        # Generate cone points
        cone_points = []
        for r in radii:
            for theta in angles_rad:
                # Point along vector at distance r
                point_along_vec = r * vec_normalized
                # Offset perpendicular to vector
                offset_magnitude = r * np.tan(theta)
                offset = offset_magnitude * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                cone_point = point_along_vec + offset
                if np.all(cone_point >= 0) and np.all(cone_point <= max_value):
                    cone_points.append(cone_point)
        
        if len(cone_points) > 0:
            cone_points = np.array(cone_points)
            # Draw cone as semi-transparent scatter
            ax.scatter(cone_points[:, 0], cone_points[:, 1], cone_points[:, 2],
                      c=color, alpha=0.1, s=1, zorder=0)
    
    # Plot points colored by classification
    for fp_name in fluorophores:
        if fp_name not in fp_colors:
            continue
        mask = labels_plot == fp_name
        if np.any(mask):
            color = fp_colors[fp_name]
            ax.scatter(ch1_plot[mask], ch2_plot[mask], ch3_plot[mask], 
                      s=1, alpha=0.3, c=color, label=fp_name, zorder=2)
    
    # Plot unclassified points in gray
    unclassified_mask = labels_plot == None
    if np.any(unclassified_mask):
        ax.scatter(ch1_plot[unclassified_mask], ch2_plot[unclassified_mask], ch3_plot[unclassified_mask],
                  s=1, alpha=0.1, c='gray', label='unclassified', zorder=2)
    
    # Draw vectors (from single FP data)
    for fp_name in fluorophores:
        if fp_name not in data_vectors_3d:
            continue
        vec = data_vectors_3d[fp_name]
        color = fp_colors.get(fp_name, "#808080")
        # Draw vector as line from origin
        vec_scaled = vec * vector_scale
        ax.plot([0, vec_scaled[0]], [0, vec_scaled[1]], [0, vec_scaled[2]],
               color=color, linewidth=3, alpha=0.8, zorder=3)
    
    # Set axis limits
    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)
    ax.set_zlim(0, max_value)
    
    # Set axis labels
    ax.set_xlabel(f"Ch1 ({ch1_config['Excitation wavelength']}nm, {ch1_config['emission filter']})", fontsize=10)
    ax.set_ylabel(f"Ch2 ({ch2_config['Excitation wavelength']}nm, {ch2_config['emission filter']})", fontsize=10)
    ax.set_zlabel(f"Ch3 ({ch3_config['Excitation wavelength']}nm, {ch3_config['emission filter']})", fontsize=10)
    
    ax.set_title("Subpanel 5: 3D Scatterplot (Colored by Classification)", fontsize=12, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper left', fontsize=8)
    
    # Set viewing angle
    ax.view_init(elev=10, azim=45)
    
    return fig, ax


def subpanel_6(row_dict=None, ax=None, 
               data_dir="data/fig2_3color_inh_spatial_control_2p3_10072025",
               single_fp_data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619",
               shared_data=None):
    """
    Generate subpanel 6: Triangle projection with pixel classifications, vectors, and shaded zones.
    
    Projects 3D pixel vectors onto a 2D triangle. Uses shared classification data if provided,
    otherwise computes it. Colors pixels by classification, overlays vectors, and draws
    shaded zones for classification regions.
    
    Parameters
    ----------
    row_dict : dict, optional
        Best channel row configuration dictionary. If None, uses FIG_2_BEST_CHAN_ROW_DICT
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (will create if None)
    data_dir : str
        Path to fig2 data directory
    single_fp_data_dir : str
        Path to single fluorophore data directory
    shared_data : dict, optional
        Pre-computed classification data from _compute_fig2_classifications
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if row_dict is None:
        row_dict = FIG_2_BEST_CHAN_ROW_DICT
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Use shared data if provided, otherwise compute
    if shared_data is None:
        print("Subpanel 6: Computing classifications...")
        shared_data = _compute_fig2_classifications(row_dict, data_dir, single_fp_data_dir)
    
    ch1_valid = shared_data['ch1_valid']
    ch2_valid = shared_data['ch2_valid']
    ch3_valid = shared_data['ch3_valid']
    pixel_labels = shared_data['pixel_labels']
    data_vectors_3d = shared_data['data_vectors_3d']
    classification_zones_3d = shared_data['classification_zones_3d']
    fluorophores = shared_data['fluorophores']
    max_value = shared_data['max_value']
    
    # Subsample for plotting
    max_distance = max_value * np.sqrt(3)
    distances = np.sqrt(ch1_valid**2 + ch2_valid**2 + ch3_valid**2)
    bin_width = 100
    n_bins = int(np.ceil(max_distance / bin_width))
    samples_per_bin = 300
    
    ch1_plot_list = []
    ch2_plot_list = []
    ch3_plot_list = []
    labels_plot_list = []
    
    for bin_idx in range(n_bins):
        bin_max = (bin_idx + 1) * bin_width
        prev_bin_max = bin_idx * bin_width
        
        if bin_idx == 0:
            bin_mask = distances < bin_max
        else:
            bin_mask = (distances >= prev_bin_max) & (distances < bin_max)
        
        if np.any(bin_mask):
            ch1_bin = ch1_valid[bin_mask]
            ch2_bin = ch2_valid[bin_mask]
            ch3_bin = ch3_valid[bin_mask]
            labels_bin = pixel_labels[bin_mask]
            
            n_in_bin = len(ch1_bin)
            n_take = min(samples_per_bin, n_in_bin)
            
            if n_in_bin > n_take:
                indices = np.random.choice(n_in_bin, n_take, replace=False)
                ch1_plot_list.append(ch1_bin[indices])
                ch2_plot_list.append(ch2_bin[indices])
                ch3_plot_list.append(ch3_bin[indices])
                labels_plot_list.append(labels_bin[indices])
            else:
                ch1_plot_list.append(ch1_bin)
                ch2_plot_list.append(ch2_bin)
                ch3_plot_list.append(ch3_bin)
                labels_plot_list.append(labels_bin)
    
    if len(ch1_plot_list) > 0:
        ch1_plot = np.concatenate(ch1_plot_list)
        ch2_plot = np.concatenate(ch2_plot_list)
        ch3_plot = np.concatenate(ch3_plot_list)
        labels_plot = np.concatenate(labels_plot_list)
    else:
        ch1_plot = ch1_valid
        ch2_plot = ch2_valid
        ch3_plot = ch3_valid
        labels_plot = pixel_labels
    
    print(f"  Projecting {len(ch1_plot)} points to triangle")
    
    # Transformation matrix for triangle projection
    # [(0, 1), (cos(30), -sin(30)), (-cos(30), -sin(30))]
    # = [(0, 1), (0.866, -0.5), (-0.866, -0.5)]
    cos_30 = np.cos(np.radians(30))  # ≈ 0.866
    sin_30 = np.sin(np.radians(30))  # = 0.5
    
    transform_matrix = np.array([
        [0, cos_30, -cos_30],      # x coordinates
        [1, -sin_30, -sin_30]      # y coordinates
    ])
    
    # Project 3D pixel vectors to 2D triangle coordinates
    pixel_vectors_3d = np.column_stack([ch1_plot, ch2_plot, ch3_plot])
    triangle_coords = pixel_vectors_3d @ transform_matrix.T
    
    x_coords = triangle_coords[:, 0]
    y_coords = triangle_coords[:, 1]
    
    print(f"  Triangle coordinate ranges: x=[{np.min(x_coords):.2f}, {np.max(x_coords):.2f}], "
          f"y=[{np.min(y_coords):.2f}, {np.max(y_coords):.2f}]")
    
    fp_colors = FIG_2_FP_COLORS
    
    # Draw shaded zones FIRST (so they appear behind points)
    # Project vectors to triangle coordinates
    vector_scale = max_value * 0.7
    for fp_name in fluorophores:
        if fp_name not in data_vectors_3d:
            continue
        vec_3d = data_vectors_3d[fp_name]
        half_angle = classification_zones_3d.get(fp_name)
        if half_angle is None:
            continue
        
        color = fp_colors.get(fp_name, "#808080")
        
        # Project vector to triangle coordinates
        vec_3d_scaled = vec_3d * vector_scale
        vec_triangle = (vec_3d_scaled.reshape(1, 3) @ transform_matrix.T)[0]
        
        # Create a shaded region around the vector in triangle space
        # Sample points within the classification zone
        n_samples = 100
        zone_points_3d = []
        
        # Generate points within the cone (3D)
        for i in range(n_samples):
            # Random distance along vector
            r = np.random.uniform(0, vector_scale)
            # Random angle within half_angle
            theta = np.random.uniform(0, np.radians(half_angle))
            
            # Point along vector
            point_along_vec = r * vec_3d
            # Perpendicular offset
            # Find perpendicular vectors
            if abs(vec_3d[0]) < 0.9:
                perp1 = np.array([1, 0, 0]) - np.dot(np.array([1, 0, 0]), vec_3d) * vec_3d
            else:
                perp1 = np.array([0, 1, 0]) - np.dot(np.array([0, 1, 0]), vec_3d) * vec_3d
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(vec_3d, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            offset_magnitude = r * np.tan(theta)
            phi = np.random.uniform(0, 2 * np.pi)
            offset = offset_magnitude * (np.cos(phi) * perp1 + np.sin(phi) * perp2)
            zone_point = point_along_vec + offset
            
            if np.all(zone_point >= 0) and np.all(zone_point <= max_value):
                zone_points_3d.append(zone_point)
        
        if len(zone_points_3d) > 0:
            zone_points_3d = np.array(zone_points_3d)
            # Project to triangle
            zone_triangle = zone_points_3d @ transform_matrix.T
            # Draw as semi-transparent scatter
            ax.scatter(zone_triangle[:, 0], zone_triangle[:, 1],
                      c=color, alpha=0.15, s=5, zorder=0, edgecolors='none')
    
    # Plot points colored by classification
    # [(0, 1), (cos(30), -sin(30)), (-cos(30), -sin(30))]
    # = [(0, 1), (0.866, -0.5), (-0.866, -0.5)]
    cos_30 = np.cos(np.radians(30))  # ≈ 0.866
    sin_30 = np.sin(np.radians(30))  # = 0.5
    
    # Transformation matrix: 2x3 matrix
    # Each column represents where each channel maps to
    # Column 0 (ch1): (0, 1) - top vertex
    # Column 1 (ch2): (cos(30), -sin(30)) - bottom right vertex
    # Column 2 (ch3): (-cos(30), -sin(30)) - bottom left vertex
    transform_matrix = np.array([
        [0, cos_30, -cos_30],      # x coordinates
        [1, -sin_30, -sin_30]      # y coordinates
    ])
    
    # Project 3D vectors to 2D triangle coordinates
    # For each pixel [ch1, ch2, ch3], compute:
    # x = 0*ch1 + cos(30)*ch2 - cos(30)*ch3 = cos(30)*(ch2 - ch3)
    # y = 1*ch1 - sin(30)*ch2 - sin(30)*ch3 = ch1 - sin(30)*(ch2 + ch3)
    pixel_vectors_3d = np.column_stack([ch1_plot, ch2_plot, ch3_plot])
    triangle_coords = pixel_vectors_3d @ transform_matrix.T
    
    x_coords = triangle_coords[:, 0]
    y_coords = triangle_coords[:, 1]
    
    print(f"  Projected {len(x_coords)} points to triangle")
    print(f"  Triangle coordinate ranges: x=[{np.min(x_coords):.2f}, {np.max(x_coords):.2f}], "
          f"y=[{np.min(y_coords):.2f}, {np.max(y_coords):.2f}]")
    
    # Get colors for each fluorophore
    fp_colors = FIG_2_FP_COLORS
    
    # Plot points colored by classification
    for fp_name in fluorophores:
        if fp_name not in fp_colors:
            continue
        mask = labels_plot == fp_name
        if np.any(mask):
            color = fp_colors[fp_name]
            ax.scatter(x_coords[mask], y_coords[mask], 
                      s=1, alpha=0.3, c=color, label=fp_name, zorder=2)
    
    # Plot unclassified points in gray
    unclassified_mask = labels_plot == None
    if np.any(unclassified_mask):
        ax.scatter(x_coords[unclassified_mask], y_coords[unclassified_mask],
                  s=1, alpha=0.1, c='gray', label='unclassified', zorder=2)
    
    # Draw vectors (projected to triangle coordinates)
    for fp_name in fluorophores:
        if fp_name not in data_vectors_3d:
            continue
        vec_3d = data_vectors_3d[fp_name]
        color = fp_colors.get(fp_name, "#808080")
        # Project vector to triangle
        vec_3d_scaled = vec_3d * vector_scale
        vec_triangle = (vec_3d_scaled.reshape(1, 3) @ transform_matrix.T)[0]
        # Draw vector as line from origin
        ax.plot([0, vec_triangle[0]], [0, vec_triangle[1]],
               color=color, linewidth=3, alpha=0.8, zorder=3)
    
    # Draw triangle outline
    triangle_vertices = np.array([
        [0, 1],                    # Top vertex (ch1)
        [cos_30, -sin_30],         # Bottom right (ch2)
        [-cos_30, -sin_30],        # Bottom left (ch3)
        [0, 1]                     # Close triangle
    ])
    ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], 
            'k-', linewidth=1, alpha=0.3, zorder=1)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Set labels
    ax.set_xlabel("Triangle X", fontsize=10)
    ax.set_ylabel("Triangle Y", fontsize=10)
    ax.set_title("Subpanel 6: Triangle Projection (Colored by Classification)", fontsize=12, fontweight='bold')
    
    # Add legend
    ax.legend(loc='best', fontsize=8)
    
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return fig, ax


def subpanel_7(row_dict=None, axes=None,
               data_dir="data/fig2_3color_inh_spatial_control_2p3_10072025",
               single_fp_data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619",
               shared_data=None):
    """
    Generate subpanel 7: Overlapping histograms of angles between pixels and each FP vector.

    Uses shared classification data if provided, otherwise computes it. For each reference
    vector (one subplot per FP, stacked vertically), plots the distribution of angle-to-vector
    for pixels colored by their predicted fluorophore label.

    Angles are in 3D and constrained to 0–90 degrees.

    Parameters
    ----------
    row_dict : dict, optional
        Best channel row configuration dictionary. If None, uses FIG_2_BEST_CHAN_ROW_DICT
    axes : array-like of matplotlib.axes.Axes, optional
        Axes to plot on (will create if None)
    data_dir : str
        Path to fig2 data directory
    single_fp_data_dir : str
        Path to single fluorophore data directory
    shared_data : dict, optional
        Pre-computed classification data from _compute_fig2_classifications

    Returns
    -------
    fig, axes
    """
    if row_dict is None:
        row_dict = FIG_2_BEST_CHAN_ROW_DICT

    # Use shared data if provided, otherwise compute
    if shared_data is None:
        print("Subpanel 7: Computing classifications...")
        shared_data = _compute_fig2_classifications(row_dict, data_dir, single_fp_data_dir)

    fluorophores = shared_data['fluorophores']
    ch1_valid = shared_data['ch1_valid']
    ch2_valid = shared_data['ch2_valid']
    ch3_valid = shared_data['ch3_valid']
    pixel_labels = shared_data['pixel_labels']
    data_vectors_3d = shared_data['data_vectors_3d']
    classification_zones_3d = shared_data['classification_zones_3d']

    n_fps = len(fluorophores)
    if axes is None:
        fig, axes = plt.subplots(n_fps, 1, figsize=(10, 2.6 * n_fps), sharex=True)
        if n_fps == 1:
            axes = np.array([axes])
    else:
        if isinstance(axes, np.ndarray):
            fig = axes[0].figure
        else:
            axes = np.array(list(axes))
            fig = axes[0].figure

    # --- Histogram settings ---
    bin_size = getattr(cfg, "angle_histogram_bin_size_degrees", 1)
    bins = np.arange(0, 90 + bin_size, bin_size)
    fp_colors = FIG_2_FP_COLORS

    # Precompute inter-FP angles from single-FP vectors (for vertical reference lines, like Fig 5.5)
    inter_fp_angles = {fp: {} for fp in fluorophores}
    for fp_a in fluorophores:
        for fp_b in fluorophores:
            if fp_a == fp_b:
                continue
            inter_fp_angles[fp_a][fp_b] = _compute_angle_3d(data_vectors_3d[fp_a], data_vectors_3d[fp_b])

    max_y = 0
    for ax_sub, ref_fp in zip(axes, fluorophores):
        ref_vec = data_vectors_3d[ref_fp]

        # Vertical lines: angle from this FP vector to the other FP vectors (computed from single-FP data)
        for other_fp, angle_deg in inter_fp_angles.get(ref_fp, {}).items():
            ax_sub.axvline(
                angle_deg,
                color=fp_colors.get(other_fp, "#808080"),
                linestyle="-",
                linewidth=3,
                alpha=0.7,
                zorder=2
            )

        # Compute angle-to-vector for all valid pixels
        angles_all = np.zeros(len(ch1_valid), dtype=np.float64)
        for i in range(len(ch1_valid)):
            angles_all[i] = _compute_angle_3d(np.array([ch1_valid[i], ch2_valid[i], ch3_valid[i]]), ref_vec)

        # Shade "accepted" region for this ref FP (0..half_angle)
        half_angle = classification_zones_3d.get(ref_fp)
        if half_angle is not None:
            ax_sub.axvspan(0, half_angle, color=fp_colors.get(ref_fp, "#808080"), alpha=0.15, zorder=0)
            ax_sub.axvline(half_angle, color=fp_colors.get(ref_fp, "#808080"), linestyle=":", linewidth=2, alpha=0.9, zorder=3)

        # Overlay histograms by pixel label (plus unclassified)
        for label in list(fluorophores) + [None]:
            mask = pixel_labels == label
            if not np.any(mask):
                continue
            vals = angles_all[mask]
            hist, _ = np.histogram(vals, bins=bins)
            max_y = max(max_y, int(np.max(hist)))

            color = "gray" if label is None else fp_colors.get(label, "#808080")
            alpha = 0.25 if label is None else 0.55
            legend_label = "unclassified" if label is None else str(label)

            ax_sub.bar(
                bins[:-1], hist, width=bin_size,
                color=color, alpha=alpha, edgecolor="none",
                label=legend_label if ref_fp == fluorophores[0] else None,
                zorder=1
            )

        ax_sub.set_ylabel(f"to {ref_fp}\ncount", fontsize=10)
        ax_sub.grid(True, alpha=0.25, axis="y")
        ax_sub.spines["top"].set_visible(False)
        ax_sub.spines["right"].set_visible(False)

    for ax_sub in axes:
        ax_sub.set_xlim(0, 90)
        if max_y > 0:
            ax_sub.set_ylim(0, max_y * 1.15)

    axes[-1].set_xlabel("Angle to reference vector (degrees)", fontsize=12)
    axes[0].set_title("Subpanel 7: pixel angular offsets to each FP vector (Fig2 best acquisitions)", fontsize=12, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        axes[0].legend(handles, labels, loc="upper right", fontsize=9, frameon=True)

    plt.tight_layout()
    return fig, axes


def save_all_subpanels(row_dict=None, best_chan_row_dict=None, 
                       output_dir="results/Figure2"):
    """
    Generate and save all Figure 2 subpanels.
    
    Parameters
    ----------
    row_dict : dict, optional
        Main row configuration dictionary. If None, uses FIG_2_ROW_DICT
    best_chan_row_dict : dict, optional
        Best channel row configuration dictionary. If None, uses FIG_2_BEST_CHAN_ROW_DICT
    output_dir : str
        Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if row_dict is None:
        row_dict = FIG_2_ROW_DICT
    if best_chan_row_dict is None:
        best_chan_row_dict = FIG_2_BEST_CHAN_ROW_DICT
    
    print("\n" + "="*60)
    print("Generating Figure 2 Subpanels")
    print("="*60)
    
    # Subpanel 1: Excitation spectra
    print("\nSubpanel 1: 2P excitation spectra...")
    try:
        fig1, ax1 = subpanel_1(row_dict)
        filepath1 = os.path.join(output_dir, "subpanel_1.png")
        fig1.savefig(filepath1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"  Saved: {filepath1}")
    except Exception as e:
        print(f"  Error in subpanel 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Subpanel 2: Emission spectra (all combinations)
    print("\nSubpanel 2: 1P emission spectra (all combinations)...")
    try:
        fig2, ax2 = subpanel_2(row_dict)
        filepath2 = os.path.join(output_dir, "subpanel_2.png")
        fig2.savefig(filepath2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: {filepath2}")
    except Exception as e:
        print(f"  Error in subpanel 2: {e}")
        import traceback
        traceback.print_exc()
    
    # Subpanel 2.1: Emission ratios (all combinations)
    print("\nSubpanel 2.1: Emission ratios (all combinations)...")
    try:
        fig2_1, ax2_1 = subpanel_2_1(row_dict)
        filepath2_1 = os.path.join(output_dir, "subpanel_2_1.png")
        fig2_1.savefig(filepath2_1, dpi=300, bbox_inches='tight')
        plt.close(fig2_1)
        print(f"  Saved: {filepath2_1}")
    except Exception as e:
        print(f"  Error in subpanel 2.1: {e}")
        import traceback
        traceback.print_exc()
    
    # Subpanel 2.2: Emission ratios (best channels only)
    print("\nSubpanel 2.2: Emission ratios (best channels)...")
    try:
        fig2_2, ax2_2 = subpanel_2_2(best_chan_row_dict)
        filepath2_2 = os.path.join(output_dir, "subpanel_2_2.png")
        fig2_2.savefig(filepath2_2, dpi=300, bbox_inches='tight')
        plt.close(fig2_2)
        print(f"  Saved: {filepath2_2}")
    except Exception as e:
        print(f"  Error in subpanel 2.2: {e}")
        import traceback
        traceback.print_exc()
    
    # Subpanel 3: Actual vs theoretical angle scatterplot
    print("\nSubpanel 3: Actual vs theoretical angle scatterplot...")
    try:
        fig3, ax3 = subpanel_3(row_dict)
        filepath3 = os.path.join(output_dir, "subpanel_3.png")
        fig3.savefig(filepath3, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"  Saved: {filepath3}")
    except Exception as e:
        print(f"  Error in subpanel 3: {e}")
        import traceback
        traceback.print_exc()
    
        # Subpanel 4: 3-color image overlay
        print("\nSubpanel 4: 3-color image overlay...")
        try:
            fig4, ax4 = subpanel_4(best_chan_row_dict)
            filepath4 = os.path.join(output_dir, "subpanel_4.png")
            fig4.savefig(filepath4, dpi=300, bbox_inches='tight')
            plt.close(fig4)
            print(f"  Saved: {filepath4}")
        except Exception as e:
            print(f"  Error in subpanel 4: {e}")
            import traceback
            traceback.print_exc()
        
    # Compute shared classification data once for subpanels 5, 6, and 7
    print("\nComputing shared classification data for subpanels 5, 6, and 7...")
    try:
        shared_data = _compute_fig2_classifications(best_chan_row_dict)
        print("  Shared data computed successfully")
    except Exception as e:
        print(f"  Error computing shared data: {e}")
        import traceback
        traceback.print_exc()
        shared_data = None
    
    # Subpanel 5: 3D scatterplot
    print("\nSubpanel 5: 3D scatterplot...")
    try:
        fig5, ax5 = subpanel_5(best_chan_row_dict, shared_data=shared_data)
        filepath5 = os.path.join(output_dir, "subpanel_5.png")
        fig5.savefig(filepath5, dpi=300, bbox_inches='tight')
        plt.close(fig5)
        print(f"  Saved: {filepath5}")
    except Exception as e:
        print(f"  Error in subpanel 5: {e}")
        import traceback
        traceback.print_exc()
    
    # Subpanel 6: Triangle projection
    print("\nSubpanel 6: Triangle projection...")
    try:
        fig6, ax6 = subpanel_6(best_chan_row_dict, shared_data=shared_data)
        filepath6 = os.path.join(output_dir, "subpanel_6.png")
        fig6.savefig(filepath6, dpi=300, bbox_inches='tight')
        plt.close(fig6)
        print(f"  Saved: {filepath6}")
    except Exception as e:
        print(f"  Error in subpanel 6: {e}")
        import traceback
        traceback.print_exc()
    
    # Subpanel 7: Angle histograms
    print("\nSubpanel 7: Angle histograms...")
    try:
        fig7, axes7 = subpanel_7(best_chan_row_dict, shared_data=shared_data)
        filepath7 = os.path.join(output_dir, "subpanel_7.png")
        fig7.savefig(filepath7, dpi=300, bbox_inches='tight')
        plt.close(fig7)
        print(f"  Saved: {filepath7}")
    except Exception as e:
        print(f"  Error in subpanel 7: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Figure 2 subpanels generation complete!")
    print("="*60)


if __name__ == "__main__":
    save_all_subpanels()

