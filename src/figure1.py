"""
Figure 1 generation for spectral unmixing methods paper.

This module generates all subpanels for Figure 1, which compares different
spectral unmixing strategies (excitation-based, emission-based, and dual domain).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch, Arc
from scipy import ndimage
import glob
import tifffile as tf
import re

from src import config as cfg
from src import demo_plotting as demo_plot
from src import data_io as io
from src import computation as comp
from src.figure_scatterplot_helpers import (
    compute_data_vector,
    vector_angle,
    filter_by_distance,
    bin_and_subsample_by_distance,
    classify_pixel_by_angle,
    compute_classification_zone,
    compute_actual_variance_perpendicular
)

# Cache zoom centers per fluorophore within a single run, so all images for that
# fluorophore zoom to the same point.
_FIGURE1_ZOOM_CENTER_CACHE = {}


def _pick_bright_zoom_center(ch1_image, ch2_image, percentile=95):
    """
    Pick a bright (y, x) location using combined intensity image (ch1 + ch2).
    Uses 95th percentile to ignore top 5% (saturated pixels).
    
    Parameters
    ----------
    ch1_image : ndarray
        Channel 1 image
    ch2_image : ndarray
        Channel 2 image
    percentile : float or None
        If float, picks the brightest pixel within the bottom `percentile` percent
        (default: 95, ignores top 5%). If None, picks the global maximum.
    """
    combined = ch1_image.astype(float) + ch2_image.astype(float)

    # Light smoothing to avoid picking single hot pixels
    combined_smooth = ndimage.gaussian_filter(combined, sigma=2)

    if percentile is None:
        # Global maximum (brightest)
        y, x = np.unravel_index(np.argmax(combined_smooth), combined_smooth.shape)
    else:
        # Use percentile threshold (default 95th percentile, ignoring top 5%)
        # We want pixels <= 95th percentile (bottom 95%), not >= (top 5%)
        thresh = np.percentile(combined_smooth, percentile)
        ys, xs = np.where(combined_smooth <= thresh)
        if ys.size == 0:
            # Fallback to global max
            y, x = np.unravel_index(np.argmax(combined_smooth), combined_smooth.shape)
        else:
            # Pick the brightest among candidates (brightest in bottom percentile)
            cand_values = combined_smooth[ys, xs]
            best_idx = np.argmax(cand_values)
            y, x = int(ys[best_idx]), int(xs[best_idx])
    
    return int(y), int(x)


def _crop_zoom(image, center_yx, zoom_factor):
    """
    Crop a square around center_yx. zoom_factor=10 means side ~ min(H, W)/10.
    """
    h, w = image.shape[:2]
    side = max(8, int(min(h, w) / float(zoom_factor)))
    half = side // 2
    cy, cx = center_yx
    y0 = max(0, cy - half)
    y1 = min(h, y0 + side)
    x0 = max(0, cx - half)
    x1 = min(w, x0 + side)

    # adjust to fixed size if we clipped at edges
    if (y1 - y0) < side:
        y0 = max(0, y1 - side)
    if (x1 - x0) < side:
        x0 = max(0, x1 - side)
    return image[y0:y1, x0:x1]


def _make_tinted_overlay(ch1_image, ch2_image, ch1_color_hex, ch2_color_hex, norm_percentile=None):
    """
    Create an RGB overlay by tinting each channel and adding them together.
    
    Parameters
    ----------
    ch1_image : ndarray
        Channel 1 image
    ch2_image : ndarray
        Channel 2 image
    ch1_color_hex : str
        Hex color for channel 1 (e.g., "#FF0000" for red)
    ch2_color_hex : str
        Hex color for channel 2 (e.g., "#0000FF" for blue)
    norm_percentile : float, optional
        If None, no normalization (preserve ratios). If provided, normalize each channel
        to this percentile (default: None, no normalization)
    """
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    ch1_rgb = hex_to_rgb(ch1_color_hex)
    ch2_rgb = hex_to_rgb(ch2_color_hex)

    # No normalization - preserve the actual ratio between channels
    if norm_percentile is None:
        # Convert to float and scale to 0-1 range based on data type
        ch1_float = ch1_image.astype(np.float64)
        ch2_float = ch2_image.astype(np.float64)
        
        # Scale to 0-1 range but preserve relative intensities between channels
        # Use the maximum value across both channels to preserve ratios
        max_val = max(np.max(ch1_float), np.max(ch2_float), 1.0)
        ch1_norm = ch1_float / max_val
        ch2_norm = ch2_float / max_val
    else:
        # Old normalization behavior (if percentile is specified)
        denom1 = np.percentile(ch1_image, norm_percentile) if np.any(ch1_image) else 1.0
        denom2 = np.percentile(ch2_image, norm_percentile) if np.any(ch2_image) else 1.0
        denom1 = denom1 if denom1 > 0 else 1.0
        denom2 = denom2 if denom2 > 0 else 1.0
        ch1_norm = np.clip(ch1_image.astype(float) / denom1, 0, 1)
        ch2_norm = np.clip(ch2_image.astype(float) / denom2, 0, 1)

    rgb1 = np.zeros((ch1_image.shape[0], ch1_image.shape[1], 3), dtype=float)
    rgb1[:, :, 0] = ch1_norm * ch1_rgb[0]
    rgb1[:, :, 1] = ch1_norm * ch1_rgb[1]
    rgb1[:, :, 2] = ch1_norm * ch1_rgb[2]

    rgb2 = np.zeros((ch2_image.shape[0], ch2_image.shape[1], 3), dtype=float)
    rgb2[:, :, 0] = ch2_norm * ch2_rgb[0]
    rgb2[:, :, 1] = ch2_norm * ch2_rgb[1]
    rgb2[:, :, 2] = ch2_norm * ch2_rgb[2]

    return np.clip(rgb1 + rgb2, 0, 1)


def _subpanel_overlay_zoom(row_dict, fp_index, ax=None, data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619"):
    """
    Common implementation for subpanels 5 and 6: zoomed 2-channel overlay for a given FP source.
    fp_index=0 -> first FP, fp_index=1 -> second FP.
    """
    fluorophores = row_dict["Fluorophores"]
    if fp_index < 0 or fp_index >= len(fluorophores):
        raise ValueError(f"fp_index={fp_index} is out of range for fluorophores={fluorophores}")
    fp_name = fluorophores[fp_index]

    ch1_config = row_dict["Channel 1"]
    ch2_config = row_dict["Channel 2"]

    def _load_ch1_from_folder(excitation_wl, filter_name):
        folder = find_image_folder(data_dir, fp_name, excitation_wl, filter_name)
        if folder is None:
            raise ValueError(
                f"Could not find folder for {fp_name}"
            )
        
        # Filter names match filename prefixes exactly (no mapping needed)
        # Just add "EmFilt" suffix to get the full prefix
        if filter_name not in ["BR2", "Red", "FarRed", "Orange"]:
            raise ValueError(f"Unknown filter name: {filter_name}. Available: BR2, Red, FarRed, Orange")
        
        filter_prefix = f"{filter_name}EmFilt"
        
        # Find aligned files matching excitation wavelength and filter
        # Aligned files don't have Ch1/Ch2 in filename - search for .tif and .ome.tif files
        # Pattern must match: filter_prefix, then excitation wavelength, in that order
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
            # List some example files for debugging
            all_tif_files = glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.ome.tif"))
            example_files = [os.path.basename(f) for f in all_tif_files[:10]]
            example_str = ", ".join(example_files) if example_files else "none found"
            raise ValueError(f"Could not find file for {fp_name}, {excitation_wl}nm, {filter_name} filter "
                           f"(prefix: {filter_prefix}) in {folder}. "
                           f"Example files found: {example_str}. "
                           f"Only searching for aligned files directly in FP directory.")
        
        # If multiple matches, prefer the most specific match
        if len(files) > 1:
            preferred = [f for f in files if f"{filter_prefix}_{excitation_wl}nm" in os.path.basename(f)]
            if len(preferred) > 0:
                files = preferred
            files = sorted(files)[:1]  # Take first if still multiple
        img = tf.imread(files[0])
        
        # Handle different image shapes
        if len(img.shape) == 3:
            if img.shape[0] < img.shape[2]:  # Likely (z, height, width)
                img = img[0, :, :]  # Take first z-slice
            elif img.shape[2] < img.shape[0]:  # Likely (height, width, channels)
                # Multi-channel image - extract Ch1 (index 0)
                if img.shape[2] >= 1:
                    img = img[:, :, 0]
                else:
                    img = img[:, :, 0]
            else:
                img = img[0, :, :]
        elif len(img.shape) == 2:
            pass  # Already 2D
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        return img

    ch1_image = _load_ch1_from_folder(ch1_config["Excitation wavelength"], ch1_config["emission filter"])
    ch2_image = _load_ch1_from_folder(ch2_config["Excitation wavelength"], ch2_config["emission filter"])

    # Determine colors: Channel 1 -> red, Channel 2 -> blue
    # For emission-based: 1080-Red -> red, 1080-Far red -> blue
    ch1_color = "#FF0000"  # Red
    ch2_color = "#0000FF"  # Blue

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    zoom_factor = getattr(cfg, "figure1_zoom_factor", 10)
    if fp_name in _FIGURE1_ZOOM_CENTER_CACHE:
        center_yx = _FIGURE1_ZOOM_CENTER_CACHE[fp_name]
    else:
        # mCherry: use 95th percentile (ignore top 5% saturated)
        # mNeptune: use global maximum (brightest)
        pick_percentile = None if fp_name == "mNeptune" else 95
        center_yx = _pick_bright_zoom_center(ch1_image, ch2_image, percentile=pick_percentile)
        _FIGURE1_ZOOM_CENTER_CACHE[fp_name] = center_yx

    ch1_crop = _crop_zoom(ch1_image, center_yx, zoom_factor=zoom_factor)
    ch2_crop = _crop_zoom(ch2_image, center_yx, zoom_factor=zoom_factor)
    
    # Background subtraction: use bottom 10% of pixels (same mask for both channels)
    # Combine both channels to determine which pixels are background
    combined_crop = ch1_crop.astype(float) + ch2_crop.astype(float)
    background_threshold = np.percentile(combined_crop, 10)
    background_mask = combined_crop <= background_threshold
    
    # Calculate background as average of bottom 10% pixels
    if np.any(background_mask):
        bg_ch1 = np.mean(ch1_crop[background_mask])
        bg_ch2 = np.mean(ch2_crop[background_mask])
    else:
        # Fallback if no background pixels found
        bg_ch1 = 0.0
        bg_ch2 = 0.0
    
    # Subtract background and clip negative values to 0
    ch1_crop_bg_subtracted = np.clip(ch1_crop.astype(float) - bg_ch1, 0, None)
    ch2_crop_bg_subtracted = np.clip(ch2_crop.astype(float) - bg_ch2, 0, None)
    
    # No normalization - preserve ratios between channels
    overlay_rgb_image = _make_tinted_overlay(ch1_crop_bg_subtracted, ch2_crop_bg_subtracted, ch1_color, ch2_color, norm_percentile=None)

    ax.imshow(overlay_rgb_image)
    ax.set_title(f"{fp_name} (zoom {zoom_factor}x)", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    return fig, ax


# Configuration dictionaries for each row
Row1_dict = {
    "name": "excitation based",
    "Fluorophores": ["mCherry", "mNeptune"],
    "Channel 1": {
        "Excitation wavelength": 1080,
        "emission filter": "BR2"
    },
    "Channel 2": {
        "Excitation wavelength": 1240,
        "emission filter": "BR2"
    }
}

Row2_dict = {
    "name": "emission based",
    "Fluorophores": ["mCherry", "mNeptune"],
    "Channel 1": {
        "Excitation wavelength": 1080,
        "emission filter": "Red"
    },
    "Channel 2": {
        "Excitation wavelength": 1080,
        "emission filter": "FarRed"
    }
}

Row3_dict = {
    "name": "dual domain",
    "Fluorophores": ["mCherry", "mNeptune"],
    "Channel 1": {
        "Excitation wavelength": 1080,
        "emission filter": "Red"
    },
    "Channel 2": {
        "Excitation wavelength": 1240,
        "emission filter": "FarRed"
    }
}

row_list = [Row1_dict, Row2_dict, Row3_dict]


def load_2p_spectra(fluorophore_name, spectra_dir=None):
    """
    Load 2P excitation spectra from CSV file.
    
    Parameters
    ----------
    fluorophore_name : str
        Name of fluorophore (e.g., 'mCherry', 'mNeptune')
    spectra_dir : str, optional
        Directory containing spectra CSV files. Defaults to dev_scripts/demo_data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Wavelength, Excitation, Emission
        Excitation column contains 2P excitation spectra values
    """
    if spectra_dir is None:
        spectra_dir = os.path.join("dev_scripts", "demo_data")
    
    # Map fluorophore names to CSV filenames.
    # Note: 2P data is in the column headers, not the filename.
    filename_map = {
        "TdTomato": "TdTomato.csv",
        "mCherry": "mCherry.csv",
        "mNeptune": "mNeptune.csv"
    }
    
    if fluorophore_name not in filename_map:
        raise ValueError(f"2P spectra not available for {fluorophore_name}")
    
    csv_path = os.path.join(spectra_dir, filename_map[fluorophore_name])
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"2P spectra file not found: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Standardize column names (handle leading space in wavelength column)
    df.columns = df.columns.str.strip()
    
    # Extract relevant columns based on fluorophore name (case-insensitive).
    # Your demo CSVs use e.g. `tdTomato 2p` / `tdTomato em` (lowercase "tdTomato"),
    # while fluorophore_name is `TdTomato`.
    def _pick_col_starting_with(lower_prefix):
        matches = [c for c in df.columns if str(c).lower().startswith(lower_prefix)]
        if not matches:
            return None
        # Prefer exact-length matches first (more stable if there are multiple columns)
        matches.sort(key=lambda x: len(str(x)))
        return matches[0]

    wavelength_col = _pick_col_starting_with("wavelength")
    excitation_col = _pick_col_starting_with(f"{fluorophore_name.lower()} 2p")
    emission_col = _pick_col_starting_with(f"{fluorophore_name.lower()} em")

    if wavelength_col is None:
        raise ValueError(f"Column 'wavelength' not found in {csv_path}")
    if excitation_col is None:
        raise ValueError(
            f"2P excitation column for '{fluorophore_name}' not found in {csv_path}. "
            f"Looked for prefix '{fluorophore_name.lower()} 2p'."
        )
    if emission_col is None:
        raise ValueError(
            f"Emission column for '{fluorophore_name}' not found in {csv_path}. "
            f"Looked for prefix '{fluorophore_name.lower()} em'."
        )
    
    # Create standardized DataFrame
    spectra_df = pd.DataFrame({
        "Wavelength": df[wavelength_col],
        "Excitation": df[excitation_col],
        "Emission": df[emission_col]
    })
    
    # Fill NaN values with 0
    spectra_df = spectra_df.fillna(0)
    
    return spectra_df


def apply_smoothing_to_spectrum(df, smoothing_std=5):
    """
    Apply Gaussian smoothing to the Excitation column of a spectrum DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Wavelength, Excitation, Emission
    smoothing_std : float, optional
        Standard deviation for Gaussian smoothing in nm. Default 5
        
    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with smoothed Excitation column
    """
    df_smoothed = df.copy()
    
    if smoothing_std > 0:
        # Get wavelength spacing for smoothing
        wavelength_spacing = np.mean(np.diff(df["Wavelength"]))
        
        # Convert std from nm to number of points
        sigma_points = smoothing_std / wavelength_spacing
        
        # Apply Gaussian smoothing to excitation
        smoothed_excitation = ndimage.gaussian_filter1d(
            df["Excitation"].values, 
            sigma=sigma_points
        )
        df_smoothed["Excitation"] = smoothed_excitation
    
    return df_smoothed


def plot_2p_excitation_spectra(fluorophore_names, excitation_wavelengths=None, 
                                channel_labels=None, wavelength_range=(950, 1250), 
                                smoothing_std=5, ax=None, load_spectra_func=None):
    """
    Plot 2P excitation spectra for fluorophores with vertical laser lines.
    
    This is a new function based on ex_em_spectra but modified for 2P excitation
    spectra in the 950-1250 nm range.
    
    Parameters
    ----------
    fluorophore_names : list of str
        List of fluorophore names to plot (e.g., ['mCherry', 'mNeptune'])
    excitation_wavelengths : list of float, optional
        List of excitation wavelengths to plot as vertical lines
    channel_labels : list of str, optional
        List of labels for excitation wavelengths (e.g., ['Channel 1 excitation wavelength', 
        'Channel 2 excitation wavelength']). If None, uses generic labels.
    wavelength_range : tuple of float, optional
        Wavelength range to plot (min, max). Default (950, 1250)
    smoothing_std : float, optional
        Standard deviation for Gaussian smoothing in nm. Default 5
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    
    # Use provided loader or default
    if load_spectra_func is None:
        load_spectra_func = load_2p_spectra
    
    # Load spectra for each fluorophore
    spectra_dict = {}
    for fp_name in fluorophore_names:
        spectra_dict[fp_name] = load_spectra_func(fp_name)
    
    # Get colors for fluorophores from config
    colors = [cfg.fluorophore_colors.get(fp_name, "#808080") for fp_name in fluorophore_names]
    
    # Filter to wavelength range and plot
    legend_patches = []
    for i, (fp_name, df) in enumerate(spectra_dict.items()):
        # Filter to wavelength range
        mask = (df["Wavelength"] >= wavelength_range[0]) & (df["Wavelength"] <= wavelength_range[1])
        df_filtered = df[mask].copy()
        
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
            color=colors[i],
            linewidth=2
        )
        ax.fill_between(
            df_filtered["Wavelength"],
            normalized_excitation,
            alpha=0.3,
            color=colors[i]
        )
        
        # Create patch for legend matching the shaded fill (alpha=0.3)
        legend_patches.append(
            Patch(facecolor=colors[i], label=f"{fp_name} 2P Excitation", alpha=0.3)
        )
    
    # Plot vertical lines for excitation wavelengths and collect line handles
    line_handles = []
    if excitation_wavelengths is not None:
        for idx, exc_wl in enumerate(excitation_wavelengths):
            if wavelength_range[0] <= exc_wl <= wavelength_range[1]:
                # Determine label
                if channel_labels is not None and idx < len(channel_labels):
                    label = channel_labels[idx]
                else:
                    label = f"Channel {idx + 1} excitation wavelength"
                
                # Get color and style from config (cycle if more than 2 channels)
                # Default config colors map Channel 1 -> mCherry color, Channel 2 -> mNeptune color
                line_color = cfg.excitation_line_colors[idx % len(cfg.excitation_line_colors)]
                line_style = cfg.excitation_line_styles[idx % len(cfg.excitation_line_styles)]
                
                line = ax.axvline(
                    exc_wl,
                    color=line_color,
                    linestyle=line_style,
                    linewidth=3,
                    alpha=0.7,
                    label=label
                )
                line_handles.append(line)
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized 2P Excitation")
    ax.set_title("2P Excitation Spectra")
    ax.set_xlim(wavelength_range)
    # Keep x-axis exactly at 0 (avoid matplotlib auto-padding below zero)
    ax.set_ylim(bottom=0)
    ax.margins(y=0)
    
    # Create legend with patches for fluorophores and lines for channels
    all_handles = legend_patches + line_handles
    ax.legend(handles=all_handles, loc='upper left')
    # Clean panel style
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return fig, ax


def subpanel_1(row_dict, ax=None):
    """
    Generate subpanel 1: 2P excitation spectra with excitation wavelengths.
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fluorophores = row_dict["Fluorophores"]
    
    # Extract excitation wavelengths and channel labels from channels
    excitation_wavelengths = []
    channel_labels = []
    for channel_key in ["Channel 1", "Channel 2"]:
        if channel_key in row_dict:
            exc_wl = row_dict[channel_key]["Excitation wavelength"]
            excitation_wavelengths.append(exc_wl)
            channel_labels.append(f"{channel_key} excitation wavelength")
    
    fig, ax = plot_2p_excitation_spectra(
        fluorophore_names=fluorophores,
        excitation_wavelengths=excitation_wavelengths,
        channel_labels=channel_labels,
        wavelength_range=(950, 1250),
        smoothing_std=5,
        ax=ax
    )
    
    return fig, ax


def plot_1p_emission_spectra(fluorophore_names, emission_filters=None,
                              channel_labels=None, wavelength_range=(550, 710),
                              smoothing_std=10, ax=None, use_display_ranges=False, load_spectra_func=None):
    """
    Plot 1P emission spectra for fluorophores with emission filter overlays.
    
    Parameters
    ----------
    fluorophore_names : list of str
        List of fluorophore names to plot (e.g., ['mCherry', 'mNeptune'])
    emission_filters : list of list, optional
        List of emission filter ranges [[min1, max1], [min2, max2], ...]
    channel_labels : list of str, optional
        List of labels for emission filters (e.g., ['Channel 1 emission filter', 
        'Channel 2 emission filter']). If None, uses generic labels.
    wavelength_range : tuple of float, optional
        Wavelength range to plot (min, max). Default (500, 750)
    smoothing_std : float, optional
        Standard deviation for Gaussian smoothing in nm. Default 10
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    use_display_ranges : bool, optional
        If True, use display-only ranges for filter visualization
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    
    # Use provided loader or default
    if load_spectra_func is None:
        load_spectra_func = load_2p_spectra
    
    # Load spectra for each fluorophore (reuse load_2p_spectra which also loads emission)
    spectra_dict = {}
    for fp_name in fluorophore_names:
        spectra_dict[fp_name] = load_spectra_func(fp_name)
    
    # Get colors for fluorophores from config
    colors = [cfg.fluorophore_colors.get(fp_name, "#808080") for fp_name in fluorophore_names]
    
    # Filter to wavelength range and prepare spectra data (don't plot yet - filters go first)
    legend_patches = []
    spectra_data = []
    for i, (fp_name, df) in enumerate(spectra_dict.items()):
        # Filter to wavelength range
        mask = (df["Wavelength"] >= wavelength_range[0]) & (df["Wavelength"] <= wavelength_range[1])
        df_filtered = df[mask].copy()
        
        # Get wavelength spacing for smoothing
        wavelength_spacing = np.mean(np.diff(df_filtered["Wavelength"]))
        
        # Apply Gaussian smoothing to emission
        if smoothing_std > 0:
            # Convert std from nm to number of points
            sigma_points = smoothing_std / wavelength_spacing
            smoothed_emission = ndimage.gaussian_filter1d(
                df_filtered["Emission"].values, 
                sigma=sigma_points
            )
        else:
            smoothed_emission = df_filtered["Emission"].values
        
        # Normalize emission to max = 1 for visibility
        max_emission = smoothed_emission.max()
        if max_emission > 0:
            normalized_emission = smoothed_emission / max_emission
        else:
            normalized_emission = smoothed_emission
        
        # Store data for later plotting (filters drawn first, then spectra)
        spectra_data.append({
            'wavelength': df_filtered["Wavelength"].values,
            'emission': normalized_emission,
            'color': colors[i],
            'name': fp_name
        })
        
        # Create patch for legend matching the shaded fill (alpha=0.3)
        legend_patches.append(
            Patch(facecolor=colors[i], label=f"{fp_name} Emission", alpha=0.3)
        )
    
    # Plot emission filters as dashed vertical edges + curved top cap + very light interior shading.
    # emission_filters can be either [[min,max], ...] or keys into cfg.emission_filter_sets.
    # Now simplified: shading goes from y=0 to y_top (filters drawn first, then white spectra mask them)
    def _draw_filter_band(ax_in, x0, x1, y_top, facecolor, alpha, edgecolor="0.35", lw=1.5, label=None):
        """
        Draw a filter overlay spanning x0..x1 with height from y=0 to y_top (constant),
        with dashed vertical edges (extending to the x-axis) and a flat top with rounded corners.
        Shading goes full height; white spectra will mask it where needed.
        """
        trans = ax_in.get_xaxis_transform()  # x in data, y in axes fraction
        
        # Interior shading: full height rectangle (white spectra will mask it)
        from matplotlib.patches import FancyBboxPatch
        rounded = FancyBboxPatch(
            (x0, 0.0),
            x1 - x0,
            y_top,
            boxstyle="round,pad=0.0",
            transform=trans,
            facecolor=facecolor,
            edgecolor="none",
            alpha=alpha,
            linewidth=0,
            zorder=2,  # Behind white spectra mask
            clip_on=False,
            mutation_aspect=1,
        )
        ax_in.add_patch(rounded)

        # Vertical dashed edges (extend to x-axis). Stop slightly below the top so the rounded cap is visible.
        corner_gap = 0.02
        y_edge_top = max(0.0, y_top - corner_gap)
        ax_in.plot([x0, x0], [0.0, y_edge_top], transform=trans, color=edgecolor, linestyle="--", linewidth=lw, alpha=0.8, zorder=4, clip_on=False)
        ax_in.plot([x1, x1], [0.0, y_edge_top], transform=trans, color=edgecolor, linestyle="--", linewidth=lw, alpha=0.8, zorder=4, clip_on=False)

        # Flat top cap with rounded corners (round capstyle gives rounded ends)
        ax_in.plot(
            [x0, x1],
            [y_top, y_top],
            transform=trans,
            color=edgecolor,
            linewidth=max(lw, 2.0),
            alpha=0.9,
            zorder=5,
            clip_on=False,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

        if label:
            ax_in.text((x0 + x1) / 2, y_top + 0.01, label, transform=trans, ha="center", va="bottom", fontsize=10, color="0.25", zorder=6, clip_on=False)

    if emission_filters is not None:
        # Requested vertical layout:
        # - broad taller so it goes over the other two
        # - red and far red at the same height
        # Filters drawn first (zorder=2), then white opaque spectra mask them (zorder=3),
        # then translucent spectra on top (zorder=4)
        
        red_y1 = 1.05 * 0.9 * 0.9 * 1.05 * 1.05 * 0.98  # ~0.918 (a smidge lower)
        broad_y1 = 1.03  # Slightly higher for better visualization
        
        # Ensure y-limit has room for overlays and labels
        ax.set_ylim(0, 1.12)

        # Separate filters by type to control drawing order: broad first (behind), then others
        broad_filters = []
        other_filters = []
        for idx, filter_spec in enumerate(emission_filters):
            if isinstance(filter_spec, str) and filter_spec == "BR2":
                broad_filters.append((idx, filter_spec))
            else:
                other_filters.append((idx, filter_spec))
        
        # Draw broad filters first (so they're behind), then other filters
        ordered_filters = broad_filters + other_filters
        
        for idx, filter_spec in ordered_filters:
            if isinstance(filter_spec, str):
                if not hasattr(cfg, "emission_filter_sets") or filter_spec not in cfg.emission_filter_sets:
                    raise KeyError(f"Unknown emission filter set '{filter_spec}'. Add it to cfg.emission_filter_sets.")
                filter_range = cfg.emission_filter_sets[filter_spec]
                # Handle both list [min, max] and CSV path string
                if isinstance(filter_range, list) and len(filter_range) == 2:
                    filter_min, filter_max = filter_range[0], filter_range[1]
                elif isinstance(filter_range, str):
                    # CSV path - load it to get the range
                    # First check for display-only ranges (for visualization)
                    use_display = False
                    if use_display_ranges and hasattr(cfg, "emission_filter_display_ranges") and filter_spec in cfg.emission_filter_display_ranges:
                        display_range = cfg.emission_filter_display_ranges[filter_spec]
                        if isinstance(display_range, list) and len(display_range) == 2:
                            filter_min, filter_max = display_range[0], display_range[1]
                            use_display = True
                    
                    # If not using display range, use transmission-based range (>=90% transmission)
                    if not use_display:
                        filter_transmission_df_temp = load_filter_transmission(filter_spec)
                        if filter_transmission_df_temp is not None and len(filter_transmission_df_temp) > 0:
                            # Only highlight range where transmission >= 90%
                            high_transmission = filter_transmission_df_temp[filter_transmission_df_temp["Transmission"] >= 90]
                            if len(high_transmission) > 0:
                                filter_min = high_transmission["Wavelength"].min()
                                filter_max = high_transmission["Wavelength"].max()
                            else:
                                # Fallback: if no transmission >= 90%, use full range
                                filter_min = filter_transmission_df_temp["Wavelength"].min()
                                filter_max = filter_transmission_df_temp["Wavelength"].max()
                        else:
                            raise ValueError(f"Could not load transmission data for filter '{filter_spec}' to determine wavelength range")
                else:
                    raise ValueError(f"Invalid filter range format for '{filter_spec}': {filter_range}")
                label = filter_spec if channel_labels is None else channel_labels[idx] if idx < len(channel_labels) else filter_spec
            else:
                if len(filter_spec) != 2:
                    continue
                filter_min, filter_max = filter_spec[0], filter_spec[1]
                label = channel_labels[idx] if channel_labels is not None and idx < len(channel_labels) else f"Filter {idx + 1}"

            # Add "filter" suffix for on-plot labels (but keep dict keys clean)
            plot_label = f"{label} filter" if not str(label).endswith("filter") else str(label)

            # Very light shading colors requested:
            # - red: light red
            # - far red: light purple (mNeptune color)
            # - broad: VERY light grey (neutral, less blue)
            if isinstance(filter_spec, str) and filter_spec == "BR2":
                face = "#E0E0E0"  # neutral light grey (less blue than #D9D9D9)
                a = 0.36  # Higher alpha for more visible shading
                y1 = broad_y1
            elif isinstance(filter_spec, str) and filter_spec == "Red":
                face = "#FCA5A5"  # light red
                a = 0.25  # Even darker shading
                y1 = red_y1
            elif isinstance(filter_spec, str) and filter_spec == "FarRed":
                face = cfg.fluorophore_colors["mNeptune"]  # Use mNeptune color (deep violet)
                a = 0.12  # Lower alpha (lighter than red)
                y1 = red_y1
            else:
                face = "#E5E7EB"
                a = 0.08
                y1 = red_y1

            _draw_filter_band(ax, filter_min, filter_max, y_top=y1, facecolor=face, alpha=a, edgecolor="0.35", lw=1.5, label=plot_label)
    
    # Now draw spectra: first opaque white to mask filters, then translucent on top
    for spec_data in spectra_data:
        # Opaque white spectra (masks filters where spectra are)
        ax.fill_between(
            spec_data['wavelength'],
            spec_data['emission'],
            color='white',
            alpha=1.0,
            zorder=3,  # Above filters
            edgecolor='none'
        )
        # Translucent colored spectra on top
        ax.plot(
            spec_data['wavelength'],
            spec_data['emission'],
            color=spec_data['color'],
            linewidth=2,
            zorder=4
        )
        ax.fill_between(
            spec_data['wavelength'],
            spec_data['emission'],
            alpha=0.3,
            color=spec_data['color'],
            zorder=4
        )
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Emission")
    ax.set_xlim(wavelength_range)
    
    # Legend: fluorophore patches only (filters are labeled directly on the plot)
    # Legend inside axes, bottom-center, stacked, on white background
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="0.85",
    )
    # Clean panel style
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return fig, ax


def subpanel_2(row_dict, ax=None):
    """
    Generate subpanel 2: 1P emission spectra overlaid with emission filters.
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fluorophores = row_dict["Fluorophores"]
    
    # Show all filter sets (so subpanel 2 is identical across all rows)
    emission_filters = ["BR2", "Red", "FarRed"]
    channel_labels = [cfg.filter_display_names.get("BR2", "BR2"), 
                      cfg.filter_display_names.get("Red", "Red"),
                      cfg.filter_display_names.get("FarRed", "FarRed")]
    
    fig, ax = plot_1p_emission_spectra(
        fluorophore_names=fluorophores,
        emission_filters=emission_filters,
        channel_labels=channel_labels,
        wavelength_range=(550, 710),
        smoothing_std=10,
        ax=ax,
        use_display_ranges=True  # Use display-only ranges for subpanel 2 visualization
    )
    
    return fig, ax


def subpanel_3(row_dict, ax=None):
    """
    Subpanel 3: table-style visualization of excitation vs emission unmixing.

    - X axis: excitation wavelengths (always shows both 1080 and 1240, even if row only uses one)
    - Y axis: emission filter sets (keys in cfg.emission_filter_sets)
    - Cells corresponding to configured channels are filled with blue shades based on row type
      (matching subpanels 9.0, 9.1, 9.2 for visual consistency).
    """
    if ax is None:
        # Smaller panel; use larger text relative to figure size
        fig, ax = plt.subplots(figsize=(5.2, 2.6))
    else:
        fig = ax.figure

    if not hasattr(cfg, "emission_filter_sets"):
        raise AttributeError("cfg.emission_filter_sets is missing; define it in src/config.py")

    # Use blue shades from config based on row type (matching subpanels 9.0, 9.1, 9.2)
    row_name = row_dict.get("name", "").lower()
    row_color = cfg.row_colors.get(row_name, "#808080")  # Default gray if name not found

    # Collect channels
    channels = []
    for channel_key in ["Channel 1", "Channel 2"]:
        if channel_key in row_dict:
            channels.append((channel_key, row_dict[channel_key]))

    # Always show both excitation wavelengths (1080 and 1240), even if row only uses one
    excitation_wavelengths = sorted([1080, 1240])
    filter_keys = ["BR2", "Red", "FarRed"]
    # Keep only those present in config
    filter_keys = [k for k in filter_keys if k in cfg.emission_filter_sets]

    # Map to grid indices
    x_index = {wl: i for i, wl in enumerate(excitation_wavelengths)}
    y_index = {fk: i for i, fk in enumerate(filter_keys)}

    # Draw grid
    ax.set_xlim(-0.5, len(excitation_wavelengths) - 0.5)
    ax.set_ylim(-0.5, len(filter_keys) - 0.5)
    ax.set_xticks(list(range(len(excitation_wavelengths))))
    ax.set_xticklabels([str(wl) for wl in excitation_wavelengths])
    ax.set_yticks(list(range(len(filter_keys))))
    # Use display names for y-axis labels
    display_labels = [cfg.filter_display_names.get(key, key) for key in filter_keys]
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Excitation wavelength (nm)", fontsize=11)
    ax.set_ylabel("Emission filter set", fontsize=11)
    # Title intentionally omitted for figure panel styling

    # Light grid lines
    for xi in range(len(excitation_wavelengths) + 1):
        ax.axvline(xi - 0.5, color="0.85", linewidth=1)
    for yi in range(len(filter_keys) + 1):
        ax.axhline(yi - 0.5, color="0.85", linewidth=1)

    # Fill cells for each channel with blue shade based on row type
    for channel_key, ch in channels:
        wl = ch["Excitation wavelength"]
        fk = ch["emission filter"]
        if wl not in x_index or fk not in y_index:
            continue
        xi = x_index[wl]
        yi = y_index[fk]

        # Use row color (blue shade) for all cells in this row
        cell_color = row_color

        ax.add_patch(
            plt.Rectangle(
                (xi - 0.5, yi - 0.5),
                1.0,
                1.0,
                facecolor=cell_color,
                alpha=0.35,
                edgecolor="none",
            )
        )

        # Channel label inside the cell (short label, larger font)
        ch_label = "Ch 1" if channel_key.endswith("1") else "Ch 2"
        ax.text(xi, yi, ch_label, ha="center", va="center", fontsize=12, color="0.2", fontweight="bold")

    # Legend intentionally hidden for now (requested). If needed later, restore legend here.
    ax.invert_yaxis()  # top-to-bottom ordering like a table
    # Square-ish cells
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig, ax


def load_filter_transmission(filter_name, spectra_dir=None):
    """
    Load filter transmission values from CSV file.
    
    First checks cfg.emission_filter_sets for a CSV path. If not found or if it's a range,
    falls back to the old hardcoded mapping.
    
    Parameters
    ----------
    filter_name : str
        Filter name ("red", "farRed", "broad")
    spectra_dir : str, optional
        Directory containing filter CSV files. Defaults to dev_scripts/demo_data
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: Wavelength, Transmission
        Returns None if no transmission file exists for this filter
    """
    if spectra_dir is None:
        spectra_dir = os.path.join("dev_scripts", "demo_data")
    
    # First, check if config has a CSV path for this filter
    if hasattr(cfg, "emission_filter_sets") and filter_name in cfg.emission_filter_sets:
        filter_spec = cfg.emission_filter_sets[filter_name]
        # If it's a string, treat it as a CSV path
        if isinstance(filter_spec, str):
            csv_path = os.path.join(spectra_dir, filter_spec)
            if os.path.exists(csv_path):
                # Read CSV file
                df = pd.read_csv(csv_path)
                
                # Standardize column names
                df.columns = df.columns.str.strip()
                
                # Verify columns exist
                if "Wavelength" not in df.columns or "T" not in df.columns:
                    raise ValueError(f"Filter transmission file {csv_path} must have 'Wavelength' and 'T' columns")
                
                # Create standardized DataFrame
                filter_df = pd.DataFrame({
                    "Wavelength": df["Wavelength"],
                    "Transmission": df["T"]
                })
                
                return filter_df
    
    # Fallback to old hardcoded mapping (for backward compatibility)
    filter_file_map = {
        "red": "Chroma-at60530m--t--0.csv",  # Red filter ~605nm
        "FarRed": "Chroma-et67050m--t--0.csv",  # Far red filter ~670nm
        "broad": None  # No transmission file for broad filter (assume 95% transmission)
    }
    
    if filter_name not in filter_file_map:
        return None
    
    filename = filter_file_map[filter_name]
    if filename is None:
        return None  # No transmission file for this filter
    
    csv_path = os.path.join(spectra_dir, filename)
    
    if not os.path.exists(csv_path):
        return None
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Verify columns exist
    if "Wavelength" not in df.columns or "T" not in df.columns:
        raise ValueError(f"Filter transmission file {csv_path} must have 'Wavelength' and 'T' columns")
    
    # Create standardized DataFrame
    filter_df = pd.DataFrame({
        "Wavelength": df["Wavelength"],
        "Transmission": df["T"]
    })
    
    return filter_df


def load_dichroic_transmission(dichroic_name, orientation, spectra_dir=None):
    """
    Load dichroic transmission values from CSV file.
    
    Parameters
    ----------
    dichroic_name : str
        Dichroic name (e.g., "514")
    orientation : str
        Either "Transmitted" or "Reflected"
    spectra_dir : str, optional
        Directory containing dichroic CSV files. Defaults to dev_scripts/demo_data
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns: Wavelength, Transmission
        For "Reflected" orientation, transmission is inverted (1-T)
        Returns None if no transmission file exists for this dichroic
    """
    if spectra_dir is None:
        spectra_dir = os.path.join("dev_scripts", "demo_data")
    
    # Map dichroic name to CSV filename
    # Currently supports "514" -> "Chroma-zt514rdc--t--0.csv"
    dichroic_file_map = {
        "514": "Chroma-zt514rdc--t--0.csv"
    }
    
    if dichroic_name not in dichroic_file_map:
        return None
    
    filename = dichroic_file_map[dichroic_name]
    csv_path = os.path.join(spectra_dir, filename)
    
    if not os.path.exists(csv_path):
        return None
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Verify columns exist
    if "Wavelength" not in df.columns or "T" not in df.columns:
        raise ValueError(f"Dichroic transmission file {csv_path} must have 'Wavelength' and 'T' columns")
    
    # Get transmission values from CSV
    # CSV values are percentages (0-100), but we need to work with decimals (0-1) for inversion
    transmission_pct = df["T"].values
    
    # For "Reflected" orientation, invert: (1 - T) where T is decimal (0-1)
    # For "Transmitted", use as-is
    if orientation == "Reflected":
        # Convert percentage to decimal, invert, then convert back to percentage
        transmission_decimal = transmission_pct / 100.0  # Convert to decimal (0-1)
        transmission_inverted = 1.0 - transmission_decimal  # Invert (1-T)
        transmission_pct = transmission_inverted * 100.0  # Convert back to percentage
    elif orientation != "Transmitted":
        raise ValueError(f"Orientation must be 'Transmitted' or 'Reflected', got '{orientation}'")
    
    # Create standardized DataFrame
    dichroic_df = pd.DataFrame({
        "Wavelength": df["Wavelength"],
        "Transmission": transmission_pct
    })
    
    return dichroic_df


def load_pockels_power_mapping():
    """
    Load Pockels to power mapping from CSV file.
    
    First tries to load from simplified CSV (pockels_power_mapping.csv).
    Falls back to original CSV format if simplified version doesn't exist.
    
    The CSV contains power readings at different Pockels values for each wavelength.
    For 2P excitation, power scales as P^2 (if power doubles, photons increase 4x).
    
    Returns
    -------
    dict
        Dictionary mapping (wavelength, pockels) -> power_mW
        e.g., {(1080, 230): 20.0, (1240, 377): 20.0}
    """
    # Find CSV file in dev_scripts/demo_data
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try simplified CSV first
    simplified_path = os.path.join(script_dir, "dev_scripts", "demo_data", "pockels_power_mapping.csv")
    if os.path.exists(simplified_path):
        return _load_pockels_power_mapping_simplified(simplified_path)
    
    # Fall back to original CSV format
    csv_path = os.path.join(script_dir, "dev_scripts", "demo_data", 
                           "Power readings after objective - 2P3_20250611.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: Pockels power mapping file not found: {csv_path}")
        return {}
    
    # Read CSV file
    df = pd.read_csv(csv_path, header=None)
    
    # Find wavelength row (row 1, 0-indexed)
    wavelength_row = df.iloc[1].values
    wavelengths = []
    for val in wavelength_row[1:]:  # Skip first column
        if pd.isna(val):
            break
        # Extract wavelength number (e.g., "1080nm" -> 1080)
        if isinstance(val, str) and 'nm' in val:
            wl_str = val.replace('nm', '').strip()
            try:
                wavelengths.append(int(wl_str))
            except:
                pass
    
    # Find power rows (rows 6-9, 0-indexed: 10mW, 20mW, 30mW, 40mW)
    power_mapping = {}
    power_levels = [10, 20, 30, 40]  # mW
    power_rows = [6, 7, 8, 9]  # 0-indexed rows
    
    for power_idx, row_idx in enumerate(power_rows):
        if row_idx >= len(df):
            continue
        power_mw = power_levels[power_idx]
        row_data = df.iloc[row_idx].values
        
        # First column is power label, rest are Pockels values
        for col_idx in range(1, min(len(row_data), len(wavelengths) + 1)):
            pockels_val = row_data[col_idx]
            if pd.notna(pockels_val):
                try:
                    pockels = int(float(pockels_val))
                    wl = wavelengths[col_idx - 1]
                    power_mapping[(wl, pockels)] = power_mw
                except:
                    pass
    
    return power_mapping


def _load_pockels_power_mapping_simplified(csv_path):
    """
    Load Pockels power mapping from simplified CSV format.
    
    CSV format:
    - First row: Headers (Wavelength, 10mW, 20mW, 30mW, 40mW)
    - Subsequent rows: Wavelength, Pockels value for 10mW, 20mW, 30mW, 40mW
    
    Parameters
    ----------
    csv_path : str
        Path to simplified CSV file
        
    Returns
    -------
    dict
        Dictionary mapping (wavelength, pockels) -> power_mW
    """
    df = pd.read_csv(csv_path)
    
    power_mapping = {}
    power_levels = [10, 20, 30, 40]  # mW
    power_columns = ['10mW', '20mW', '30mW', '40mW']
    
    for _, row in df.iterrows():
        wavelength = int(row['Wavelength'])
        
        for power_idx, col_name in enumerate(power_columns):
            if col_name not in df.columns:
                continue
            pockels_val = row[col_name]
            if pd.notna(pockels_val):
                try:
                    pockels = int(float(pockels_val))
                    power_mw = power_levels[power_idx]
                    power_mapping[(wavelength, pockels)] = power_mw
                except:
                    pass
    
    return power_mapping


def extract_pockels_from_filename(filename):
    """
    Extract Pockels value from filename.
    
    Parameters
    ----------
    filename : str
        Filename (e.g., "BR2EmFilt_1080nm_230poc.tif")
        
    Returns
    -------
    int or None
        Pockels value (e.g., 230) or None if not found
    """
    match = re.search(r'(\d+)poc', filename)
    if match:
        return int(match.group(1))
    return None


def get_power_from_pockels(wavelength, pockels, power_mapping=None):
    """
    Get power (mW) from wavelength and Pockels value.
    
    If exact match not found, interpolates between nearest Pockels values.
    
    Parameters
    ----------
    wavelength : int
        Excitation wavelength in nm
    pockels : int
        Pockels value
    power_mapping : dict, optional
        Dictionary mapping (wavelength, pockels) -> power_mW.
        If None, loads from CSV.
        
    Returns
    -------
    float
        Power in mW, or 20.0 (default) if not found
    """
    if power_mapping is None:
        power_mapping = load_pockels_power_mapping()
    
    # Try exact match
    if (wavelength, pockels) in power_mapping:
        return power_mapping[(wavelength, pockels)]
    
    # Find nearest Pockels values for this wavelength
    wl_pockels_powers = [(p, pw) for (wl, p), pw in power_mapping.items() if wl == wavelength]
    
    if len(wl_pockels_powers) == 0:
        # No data for this wavelength, return default
        return 20.0
    
    # Sort by Pockels value
    wl_pockels_powers.sort(key=lambda x: x[0])
    pockels_vals = [p for p, _ in wl_pockels_powers]
    power_vals = [pw for _, pw in wl_pockels_powers]
    
    # Interpolate
    power = np.interp(pockels, pockels_vals, power_vals)
    return float(power)


def _format_acquisition_name(wavelength, filter_name, pockels_value):
    """
    Format acquisition name with wavelength, filter, Pockels, and power.
    
    Parameters
    ----------
    wavelength : int
        Excitation wavelength in nm
    filter_name : str
        Filter name (e.g., 'broad', 'red', 'farRed')
    pockels_value : int or None
        Pockels value
        
    Returns
    -------
    str
        Formatted acquisition name (e.g., "1080nm_broad_230poc_20mW")
    """
    # Filter names are already in correct format (BR2, Red, FarRed)
    # Use as-is for filename formatting
    if pockels_value is not None:
        power_mapping = load_pockels_power_mapping()
        power = get_power_from_pockels(wavelength, pockels_value, power_mapping)
        power_suffix = f"_{int(power)}mW" if power is not None else ""
        return f"{wavelength}nm_{filter_name}_{pockels_value}poc{power_suffix}"
    else:
        return f"{wavelength}nm_{filter_name}"


def compute_predicted_channel_signals(row_dict, ch1_pockels=None, ch2_pockels=None, debug=False, load_spectra_func=None):
    """
    Compute predicted signal for each fluorophore in each channel.
    
    Uses actual filter transmission values when available (for red and far red filters).
    For broad filter, assumes 95% transmission within the filter range.
    Accounts for Pockels power: for 2P excitation, signal scales as (power/20)^2.
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
    ch1_pockels : int, optional
        Pockels value for channel 1. If None, assumes 20mW power.
    ch2_pockels : int, optional
        Pockels value for channel 2. If None, assumes 20mW power.
    debug : bool, optional
        If True, print debug information
    load_spectra_func : callable, optional
        Function to load 2P spectra. If None, uses load_2p_spectra from this module.
        Should have signature: load_spectra_func(fluorophore_name) -> pd.DataFrame
        
    Returns
    -------
    dict
        Dictionary with structure: {fp_name: {channel_name: signal_value}}
    """
    fluorophores = row_dict["Fluorophores"]
    signals = {fp: {} for fp in fluorophores}
    
    # Load Pockels power mapping
    power_mapping = load_pockels_power_mapping()
    
    # Use provided loader or default to load_2p_spectra
    if load_spectra_func is None:
        load_spectra_func = load_2p_spectra
    
    # Load spectra for all fluorophores and apply smoothing
    spectra_dict = {}
    for fp_name in fluorophores:
        print(f"DEBUG compute_predicted_channel_signals: Loading spectra for '{fp_name}'")
        try:
            df_raw = load_spectra_func(fp_name)
            print(f"DEBUG: Successfully loaded spectra for '{fp_name}', shape: {df_raw.shape}")
            # Apply 5nm Gaussian smoothing to excitation spectra
            df_smoothed = apply_smoothing_to_spectrum(df_raw, smoothing_std=5)
            spectra_dict[fp_name] = df_smoothed
            print(f"DEBUG: Applied smoothing to '{fp_name}'")
        except Exception as e:
            print(f"ERROR: Failed to load spectra for '{fp_name}': {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Process each channel
    for channel_key in ["Channel 1", "Channel 2"]:
        if channel_key not in row_dict:
            continue
            
        channel_config = row_dict[channel_key]
        exc_wl = channel_config["Excitation wavelength"]
        filter_name = channel_config["emission filter"]
        
        # Filter names match filename prefixes exactly (no normalization needed)
        # Valid names: BR2, Red, FarRed, Orange
        if filter_name not in ["BR2", "Red", "FarRed", "Orange"]:
            # Try to map common variations for backward compatibility
            filter_name_map = {
                "broad": "BR2",
                "red": "Red",
                "farRed": "FarRed",
                "far_red": "FarRed",
                "far red": "FarRed",
                "orange": "Orange"
            }
            if filter_name in filter_name_map:
                filter_name = filter_name_map[filter_name]
            else:
                raise ValueError(f"Unknown filter name: {filter_name}. Available: BR2, Red, FarRed, Orange")
        
        # Get Pockels value for this channel
        channel_pockels = ch1_pockels if channel_key == "Channel 1" else ch2_pockels
        
        # Get power from Pockels (default 20mW if not provided)
        if channel_pockels is not None:
            power_mw = get_power_from_pockels(exc_wl, channel_pockels, power_mapping)
        else:
            power_mw = 20.0  # Default power
        
        # Power correction factor: (power/20)^2 for 2P excitation
        power_factor = (power_mw / 20.0) ** 2
        
        # Load filter transmission if available
        filter_transmission_df = load_filter_transmission(filter_name)
        
        # Handle filter range (can be list, CSV path string, or key)
        if isinstance(filter_name, str):
            if hasattr(cfg, "emission_filter_sets") and filter_name in cfg.emission_filter_sets:
                filter_spec = cfg.emission_filter_sets[filter_name]
                # If it's a list [min, max], use it as wavelength range
                if isinstance(filter_spec, list) and len(filter_spec) == 2:
                    filter_min, filter_max = filter_spec[0], filter_spec[1]
                # If it's a string (CSV path), we need to get the range from the CSV
                elif isinstance(filter_spec, str):
                    # Load the CSV to get the wavelength range
                    filter_transmission_df_temp = load_filter_transmission(filter_name)
                    if filter_transmission_df_temp is not None and len(filter_transmission_df_temp) > 0:
                        filter_min = filter_transmission_df_temp["Wavelength"].min()
                        filter_max = filter_transmission_df_temp["Wavelength"].max()
                    else:
                        # Fallback: can't determine range, skip this filter
                        continue
                else:
                    # Unknown format, skip
                    continue
            else:
                # Debug: print what filter names are available
                if filter_name in ['Red', 'FarRed']:
                    print(f"WARNING: Filter name '{filter_name}' not found in cfg.emission_filter_sets")
                    print(f"  Available keys: {list(cfg.emission_filter_sets.keys())}")
                    print(f"  Filter name repr: {repr(filter_name)}")
                continue
        else:
            filter_min, filter_max = filter_name[0], filter_name[1]
        
        # Compute signal for each fluorophore
        for fp_name in fluorophores:
            df = spectra_dict[fp_name]
            
            # Get 2P excitation at excitation wavelength (now using smoothed values)
            exc_mask = np.abs(df["Wavelength"] - exc_wl).idxmin()
            exc_value = df.loc[exc_mask, "Excitation"]
            
            # Get 1P emission filtered by emission filter with transmission values
            em_mask = (df["Wavelength"] >= filter_min) & (df["Wavelength"] <= filter_max)
            emission_in_range = df.loc[em_mask].copy()
            
            # Calculate wavelength spacing for proper integration
            # Use trapezoidal integration: sum of (value * wavelength_spacing)
            wavelengths = emission_in_range["Wavelength"].values
            if len(wavelengths) > 1:
                # Calculate spacing between adjacent points (use mean for trapezoidal rule)
                wavelength_spacings = np.diff(wavelengths)
                # For trapezoidal integration, use spacing to next point for each value
                # First point uses spacing to second, last point uses spacing from second-to-last
                spacings = np.concatenate([[wavelength_spacings[0]], 
                                          (wavelength_spacings[:-1] + wavelength_spacings[1:]) / 2,
                                          [wavelength_spacings[-1]]])
            else:
                # Single point - use a default spacing (shouldn't happen, but handle it)
                spacings = np.array([1.0])
            
            # Apply dichroics first (if specified)
            dichroic_transmission_interp = None
            if "dichroics" in channel_config:
                # Process all dichroics sequentially
                # Start with 100% transmission (no loss)
                combined_dichroic_transmission = np.ones_like(wavelengths)
                
                for dichroic_spec in channel_config["dichroics"]:
                    dichroic_name = dichroic_spec["name"]
                    orientation = dichroic_spec["orientation"]
                    
                    # Load dichroic transmission
                    dichroic_df = load_dichroic_transmission(dichroic_name, orientation)
                    if dichroic_df is not None:
                        # Interpolate dichroic transmission to match emission wavelengths
                        # Use linear interpolation, extrapolate with closest value
                        dichroic_transmission = np.interp(
                            wavelengths,
                            dichroic_df["Wavelength"].values,
                            dichroic_df["Transmission"].values,
                            left=dichroic_df["Transmission"].iloc[0],  # Use first value if below range
                            right=dichroic_df["Transmission"].iloc[-1]  # Use last value if above range
                        ) / 100.0  # Convert from percentage to fraction
                        
                        # Multiply with existing transmission (sequential application)
                        combined_dichroic_transmission *= dichroic_transmission
                
                dichroic_transmission_interp = combined_dichroic_transmission
            
            # Apply filter transmission
            if filter_transmission_df is not None:
                # Apply filter transmission: interpolate transmission to match emission wavelengths
                # Use numpy interp to interpolate transmission values
                # Note: transmission values in CSV are percentages (0-100), convert to fractions (0-1)
                filter_transmission_interp = np.interp(
                    wavelengths,
                    filter_transmission_df["Wavelength"].values,
                    filter_transmission_df["Transmission"].values,
                    left=0,  # Outside range: 0 transmission
                    right=0  # Outside range: 0 transmission
                ) / 100.0  # Convert from percentage to fraction
                
                # Combine dichroic and filter transmission
                if dichroic_transmission_interp is not None:
                    total_transmission = filter_transmission_interp * dichroic_transmission_interp
                else:
                    total_transmission = filter_transmission_interp
                
                # Integrate: multiply emission by total transmission and wavelength spacing, then sum
                filtered_emission = (emission_in_range["Emission"].values * total_transmission * spacings).sum()
            else:
                # No transmission file (e.g., broad filter) - assume 95% transmission
                if dichroic_transmission_interp is not None:
                    # Apply dichroic transmission, then 95% for filter
                    filtered_emission = (emission_in_range["Emission"].values * dichroic_transmission_interp * spacings).sum() * 0.95
                else:
                    # No dichroic, no filter file - assume 95% transmission
                    filtered_emission = (emission_in_range["Emission"].values * spacings).sum() * 0.95
            
            # Predicted signal = excitation * filtered emission * power_factor
            signal = exc_value * filtered_emission * power_factor
            signals[fp_name][channel_key] = signal
            
            # Debug output for specific cases
            if debug and fp_name == 'mNeptune' and (exc_wl == 1080 or exc_wl == 1240):
                print(f"\nDEBUG compute_predicted_channel_signals: {fp_name}, {channel_key}")
                print(f"  Excitation wavelength: {exc_wl}nm")
                print(f"  Filter name from config: '{filter_name}'")
                print(f"  Filter range lookup: {cfg.emission_filter_sets.get(filter_name, 'NOT FOUND')}")
                print(f"  Filter: {filter_name} ({filter_min}-{filter_max}nm)")
                print(f"  Pockels: {channel_pockels}, Power: {power_mw}mW, Factor: {power_factor:.4f}")
                print(f"  2P Excitation value: {exc_value:.6f}")
                print(f"  Emission in range (raw sum, not integrated): {emission_in_range['Emission'].sum():.6f}")
                print(f"  Wavelength range: {filter_min}-{filter_max}nm ({filter_max-filter_min:.1f}nm)")
                print(f"  Number of data points: {len(emission_in_range)}")
                if len(wavelengths) > 1:
                    print(f"  Mean wavelength spacing: {np.mean(spacings):.3f}nm")
                if filter_transmission_df is not None:
                    print(f"  Filter transmission applied (integrated): {filtered_emission:.6f}")
                    if (
                        "filter_transmission_interp" in locals()
                        and filter_transmission_interp is not None
                        and len(filter_transmission_interp) > 0
                    ):
                        print(
                            f"  Transmission range: {filter_transmission_interp.min():.4f} - {filter_transmission_interp.max():.4f}"
                        )
                else:
                    print(f"  No filter transmission (broad, integrated): {filtered_emission:.6f}")
                print(f"  Final signal: {signal:.6f} = {exc_value:.6f} * {filtered_emission:.6f} * {power_factor:.4f}")
    
    return signals


def subpanel_4(row_dict, ax=None):
    """
    Generate subpanel 4: Visualization of predicted unmixing ratios.
    
    Creates 3 vertically stacked bar graphs:
    - Top: both fluorophores overlaid (alpha transparency)
    - Middle: first FP only
    - Bottom: second FP only
    
    Bars normalized so brightest channel for each FP is 1.
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
    ax : matplotlib.axes.Axes or array, optional
        Axes to plot on (will create subplots if None)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    fluorophores = row_dict["Fluorophores"]
    
    # Compute predicted signals
    signals = compute_predicted_channel_signals(row_dict)
    
    # Get channel names
    channel_names = []
    for channel_key in ["Channel 1", "Channel 2"]:
        if channel_key in row_dict:
            channel_names.append(channel_key)
    
    # Extract signal values and normalize (vector length = 1, L2 norm)
    fp_data = {}
    for fp_name in fluorophores:
        fp_signals = np.array([signals[fp_name].get(ch, 0) for ch in channel_names])
        vector_length = np.linalg.norm(fp_signals)
        fp_data[fp_name] = (fp_signals / vector_length if vector_length > 0 else fp_signals).tolist()
    
    # Create subplots - square and smaller relative to text (like subpanel 3)
    if ax is None:
        fig, axes = plt.subplots(3, 1, figsize=(4, 4.5), sharex=True)
    else:
        if isinstance(ax, np.ndarray):
            axes = ax
            fig = axes[0].figure
        else:
            fig, axes = plt.subplots(3, 1, figsize=(4, 4.5), sharex=True)
    
    # Get colors
    colors = [cfg.fluorophore_colors.get(fp_name, "#808080") for fp_name in fluorophores]
    
    x_pos = np.arange(len(channel_names))
    width = 1.0  # Full width for no gaps (histogram style)
    
    # Create short channel labels
    channel_labels_short = ["Ch 1" if ch.endswith("1") else "Ch 2" for ch in channel_names]
    
    # Determine max y value for common y-axis
    max_y = max(max(fp_data[fp]) for fp in fluorophores) * 1.1
    
    # Top subplot: first FP only (mCherry)
    if len(fluorophores) > 0:
        ax_top = axes[0]
        values = fp_data[fluorophores[0]]
        ax_top.bar(x_pos, values, width,
                   color=colors[0], alpha=0.3, edgecolor='black', linewidth=1)
        ax_top.set_title(fluorophores[0], fontsize=11, fontweight='bold')
        ax_top.set_ylim(0, max_y)
        ax_top.set_yticks([0, 1])
        ax_top.tick_params(axis='y', labelsize=10)
        ax_top.grid(True, alpha=0.3, axis='y')
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
    
    # Middle subplot: second FP only (mNeptune)
    if len(fluorophores) > 1:
        ax_mid = axes[1]
        values = fp_data[fluorophores[1]]
        ax_mid.bar(x_pos, values, width,
                   color=colors[1], alpha=0.3, edgecolor='black', linewidth=1)
        ax_mid.set_title(fluorophores[1], fontsize=11, fontweight='bold')
        ax_mid.set_ylim(0, max_y)
        ax_mid.set_yticks([0, 1])
        ax_mid.set_ylabel("Relative Signal", fontsize=16, labelpad=20)
        ax_mid.tick_params(axis='y', labelsize=10)
        ax_mid.grid(True, alpha=0.3, axis='y')
        ax_mid.spines["top"].set_visible(False)
        ax_mid.spines["right"].set_visible(False)
    
    # Bottom subplot: both fluorophores overlaid (alpha=0.3 like spectra)
    ax_bot = axes[2]
    for i, fp_name in enumerate(fluorophores):
        values = fp_data[fp_name]
        ax_bot.bar(x_pos, values, width, 
                   color=colors[i], alpha=0.3, edgecolor='black', linewidth=1)
    ax_bot.set_title("Overlay", fontsize=11, fontweight='bold')
    ax_bot.set_ylim(0, max_y)
    ax_bot.set_yticks([0, 1])
    ax_bot.set_xticks(x_pos)
    ax_bot.set_xticklabels(channel_labels_short, fontsize=10)
    ax_bot.tick_params(axis='x', labelsize=10)
    ax_bot.tick_params(axis='y', labelsize=10)
    ax_bot.grid(True, alpha=0.3, axis='y')
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    
    # Make subplots square
    for ax_sub in axes:
        ax_sub.set_aspect('auto')
        # Adjust aspect ratio to be approximately square
        xlim = ax_sub.get_xlim()
        ylim = ax_sub.get_ylim()
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        if xrange > 0 and yrange > 0:
            # Try to make it square-ish by adjusting
            ax_sub.set_box_aspect(1.0)
    
    # Remove spacing between subplots
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    
    return fig, axes


def find_image_folder(data_dir, fluorophore_name, excitation_wl, filter_name):
    """
    Find the folder containing aligned images for a specific fluorophore.
    
    Aligned images are now stored directly in the FP directory (e.g., mCherry/ or mCherry_mouse/),
    not in subdirectories. This function returns the FP directory path.
    
    Parameters
    ----------
    data_dir : str
        Base data directory (e.g., "data/fig1_fig2_1color_3mice_singleplane_june20250619")
    fluorophore_name : str
        Name of fluorophore (e.g., "mCherry", "mNeptune")
    excitation_wl : int
        Excitation wavelength in nm (e.g., 1080, 1240) - not used for folder lookup anymore
    filter_name : str
        Filter name ("broad", "red", "far red") - not used for folder lookup anymore
        
    Returns
    -------
    str or None
        Path to the FP directory containing aligned images, or None if not found
    """
    # Convert to absolute path if relative
    if not os.path.isabs(data_dir):
        # Try to resolve relative to current working directory
        data_dir = os.path.abspath(data_dir)
    
    # Check if base data directory exists
    if not os.path.isdir(data_dir):
        return None
    
    # Try multiple possible directory name formats
    possible_folder_names = [
        fluorophore_name,  # e.g., "mCherry"
        f"{fluorophore_name}_mouse",  # e.g., "mCherry_mouse"
    ]
    
    # Check each possible directory name
    for fp_folder in possible_folder_names:
        fp_dir = os.path.join(data_dir, fp_folder)
        if os.path.isdir(fp_dir):
            return fp_dir
    
    # If none found, return None
    return None


def load_channel_data(data_dir, fluorophore_name, excitation_wl, filter_name, 
                      channel_num=1, subsample_factor=None, allow_subdirectories=False):
    """
    Load image data for a specific channel configuration.
    
    Parameters
    ----------
    data_dir : str
        Base data directory
    fluorophore_name : str
        Name of fluorophore
    excitation_wl : int
        Excitation wavelength in nm
    filter_name : str
        Filter name
    channel_num : int
        Channel number (1 or 2)
    subsample_factor : int, optional
        Factor to subsample pixels (e.g., 2 means take every 2nd pixel)
    allow_subdirectories : bool, optional
        If True, search in subdirectories if files not found directly in FP directory.
        Default False - only searches directly in FP directory for aligned files.
        
    Returns
    -------
    np.ndarray or tuple
        If return_pockels is False: 2D array of pixel intensities, flattened if subsampled
        If return_pockels is True: tuple of (data_array, pockels_value)
    """
    folder_path = find_image_folder(data_dir, fluorophore_name, excitation_wl, filter_name)

    # Fallback: multiplexed layout (no per-fluorophore subfolders).
    # In this case, `data_dir` contains acquisition subdirectories like:
    #   RedEmFilt_1080nm_<pockels>poc_<pmt>...
    # and the aligned image(s) are stored inside those subdirectories.
    if folder_path is None:
        abs_data_dir = os.path.abspath(data_dir) if not os.path.isabs(data_dir) else data_dir
        multiplex_prefix = f"{filter_name}EmFilt_{excitation_wl}nm"
        candidate_dirs = []
        if os.path.isdir(abs_data_dir):
            for d in os.listdir(abs_data_dir):
                p = os.path.join(abs_data_dir, d)
                if os.path.isdir(p) and d.startswith(multiplex_prefix):
                    candidate_dirs.append(d)

        if len(candidate_dirs) > 0:
            candidate_dirs.sort()
            folder_path = os.path.join(abs_data_dir, candidate_dirs[0])
        else:
            # List available directories for debugging
            if os.path.isdir(abs_data_dir):
                available_dirs = [d for d in os.listdir(abs_data_dir)
                                   if os.path.isdir(os.path.join(abs_data_dir, d))]
                available_dirs_str = ", ".join(available_dirs[:10])  # Limit to first 10
                if len(available_dirs) > 10:
                    available_dirs_str += f", ... ({len(available_dirs)} total)"
            else:
                available_dirs_str = f"data_dir does not exist: {abs_data_dir}"
            raise ValueError(
                f"Could not find folder for {fluorophore_name} in {abs_data_dir}. "
                f"Tried multiplexed acquisition prefix '{multiplex_prefix}'. "
                f"Available directories: {available_dirs_str}"
            )
    
    # Filter names match filename prefixes exactly (no normalization needed)
    # Valid names: BR2, Red, FarRed, Orange
    # Just add "EmFilt" suffix to get the full prefix
    if filter_name not in ["BR2", "Red", "FarRed", "Orange"]:
        raise ValueError(f"Unknown filter name: {filter_name}. Available: BR2, Red, FarRed, Orange")
    
    filter_prefix = f"{filter_name}EmFilt"
    
    # Find aligned files matching excitation wavelength and filter
    # Aligned files don't have Ch1/Ch2 in filename - search for .tif and .ome.tif files
    # Pattern must match: filter_prefix, then excitation wavelength, in that order
    # e.g., BR2EmFilt_1080nm_*.tif or RedEmFilt_1080nm_*.tif
    # Use underscore to ensure filter comes before wavelength
    pattern1 = os.path.join(folder_path, f"{filter_prefix}_{excitation_wl}nm*.tif")
    pattern2 = os.path.join(folder_path, f"{filter_prefix}_{excitation_wl}nm*.ome.tif")
    
    matching_files = glob.glob(pattern1) + glob.glob(pattern2)
    matching_files = list(set(matching_files))  # Remove duplicates
    
    # Validate that matched files actually start with the filter prefix
    # (glob might match incorrectly in some edge cases)
    validated_files = []
    for f in matching_files:
        basename = os.path.basename(f)
        if basename.startswith(filter_prefix):
            validated_files.append(f)
        else:
            print(f"DEBUG: Rejected false match: {basename} (doesn't start with {filter_prefix})")
    matching_files = validated_files
    
    # If not found, try more flexible pattern (filter and wavelength anywhere)
    # But be careful: "RedEmFilt" should NOT match "FarRedEmFilt"
    # So we need to ensure the filter prefix is a complete token (at start or after underscore)
    if len(matching_files) == 0:
        all_tif_files = glob.glob(os.path.join(folder_path, "*.tif")) + glob.glob(os.path.join(folder_path, "*.ome.tif"))
        matching_files = []
        for f in all_tif_files:
            basename = os.path.basename(f)
            # Check if filter_prefix appears as a complete token (not embedded in another word):
            # 1. At the start of filename: "RedEmFilt_..." ✓
            # 2. After underscore: "_RedEmFilt_..." ✓ (but NOT "FarRedEmFilt_..." ✗)
            # This ensures "RedEmFilt" doesn't match "FarRedEmFilt"
            is_match = False
            if basename.startswith(filter_prefix):
                # Starts with filter prefix - good match
                is_match = True
            elif f"_{filter_prefix}_" in basename:
                # Has underscore before and after - good match (complete token)
                # Check that the character before the underscore is not a letter (to avoid "FarRedEmFilt")
                idx = basename.find(f"_{filter_prefix}_")
                if idx > 0:
                    char_before = basename[idx - 1]
                    # If it's an underscore or non-letter, it's a complete token
                    if not char_before.isalpha():
                        is_match = True
                else:
                    is_match = True
            elif f"_{filter_prefix}" in basename:
                # Check if it's at the end of the base name (before extension)
                base_no_ext = basename.split(".")[0]
                if base_no_ext.endswith(f"_{filter_prefix}"):
                    idx = basename.find(f"_{filter_prefix}")
                    if idx > 0:
                        char_before = basename[idx - 1]
                        if not char_before.isalpha():
                            is_match = True
                    else:
                        is_match = True
            
            if is_match and f"{excitation_wl}nm" in basename:
                matching_files.append(f)
        matching_files = list(set(matching_files))
    
    # If not found and allow_subdirectories is True, search in subdirectories (with Ch1/Ch2)
    # But be careful: "RedEmFilt" should NOT match "FarRedEmFilt"
    if len(matching_files) == 0 and allow_subdirectories:
        # Search all files first, then filter for exact prefix match
        all_subdir_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.tif', '.ome.tif')) and f"{excitation_wl}nm" in file:
                    full_path = os.path.join(root, file)
                    basename = os.path.basename(full_path)
                    # Check if filter_prefix appears as complete token (at start or after underscore)
                    if basename.startswith(filter_prefix) or f"_{filter_prefix}" in basename:
                        if f"Ch{channel_num}" in basename:
                            all_subdir_files.append(full_path)
        matching_files = all_subdir_files
        matching_files = list(set(matching_files))
    
    if len(matching_files) == 0:
        # List some example files for debugging
        all_tif_files = glob.glob(os.path.join(folder_path, "*.tif")) + glob.glob(os.path.join(folder_path, "*.ome.tif"))
        example_files = [os.path.basename(f) for f in all_tif_files[:10]]
        example_str = ", ".join(example_files) if example_files else "none found"
        error_msg = (f"Could not find file for {fluorophore_name}, "
                    f"{excitation_wl}nm, {filter_name} filter (prefix: {filter_prefix}) in {folder_path}. "
                    f"Example files found: {example_str}")
        if not allow_subdirectories:
            error_msg += " (Subdirectory search disabled - only searching for aligned files directly in FP directory)"
        raise ValueError(error_msg)
    
    # If multiple matches, prefer the most specific match (exact filter_wavelength pattern)
    # Also filter out false matches (e.g., "RedEmFilt" should not match "FarRedEmFilt")
    if len(matching_files) > 1:
        # Filter out false matches: ensure filter_prefix is a complete token
        # "RedEmFilt" should NOT match "FarRedEmFilt"
        valid_matches = []
        for f in matching_files:
            basename = os.path.basename(f)
            # Check if filter_prefix appears as a complete token (at start or after underscore/non-letter)
            if basename.startswith(filter_prefix) or f"_{filter_prefix}" in basename:
                valid_matches.append(f)
        
        if len(valid_matches) > 0:
            matching_files = valid_matches
        
        # Prefer files where filter_prefix comes directly before wavelength with underscore
        if len(matching_files) > 1:
            preferred = [f for f in matching_files if f"{filter_prefix}_{excitation_wl}nm" in os.path.basename(f)]
            if len(preferred) > 0:
                matching_files = preferred
        # If still multiple, prefer files without subdirectory paths (direct in FP directory)
        if len(matching_files) > 1:
            direct_files = [f for f in matching_files if os.path.dirname(f) == folder_path]
            if len(direct_files) > 0:
                matching_files = direct_files
        # If still multiple, take the first one alphabetically
        matching_files = sorted(matching_files)[:1]
    
    # Extract Pockels value from filename
    selected_file = matching_files[0] if len(matching_files) > 0 else None
    
    # Debug: print which file was selected
    if selected_file:
        print(f"DEBUG load_channel_data: {fluorophore_name}, {excitation_wl}nm, {filter_name} -> {os.path.basename(selected_file)}")
    pockels_value = None
    if selected_file:
        filename = os.path.basename(selected_file)
        pockels_value = extract_pockels_from_filename(filename)
    
    # Debug: print which file was selected (can be removed later)
    if selected_file and (len(matching_files) > 1 or len(glob.glob(pattern1) + glob.glob(pattern2)) > 1):
        print(f"DEBUG: Selected file for {fluorophore_name}, {excitation_wl}nm, {filter_name}: {os.path.basename(selected_file)}")
        if len(matching_files) > 1:
            print(f"DEBUG: Had {len(matching_files)} matches, selected: {os.path.basename(selected_file)}")
    
    # Load the image directly with tifffile
    image = tf.imread(selected_file)
    
    # Handle different image shapes
    # Could be: (height, width), (z, height, width), or (height, width, channels)
    if len(image.shape) == 3:
        if image.shape[0] < image.shape[2]:  # Likely (z, height, width)
            image = image[0, :, :]  # Take first z-slice
        elif image.shape[2] < image.shape[0]:  # Likely (height, width, channels)
            # Multi-channel image - extract the requested channel (0-indexed, so channel_num-1)
            if image.shape[2] >= channel_num:
                image = image[:, :, channel_num - 1]
            else:
                # If fewer channels than requested, take first channel
                image = image[:, :, 0]
        else:
            # Default: take first slice
            image = image[0, :, :]
    elif len(image.shape) == 2:
        pass  # Already 2D
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    # Subsample if requested
    if subsample_factor is not None and subsample_factor > 1:
        image = image[::subsample_factor, ::subsample_factor]
    
    return image.flatten(), pockels_value


# compute_data_vector is now imported from figure_scatterplot_helpers


def subpanel_5(
    row_dict,
    ax=None,
    data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619",
    predicted_signals=None,
    preselected_points=None,
):
    """
    Generate subpanel 5: Scatterplot with vectors and classification cones.
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
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
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    
    fluorophores = row_dict["Fluorophores"]
    ch1_config = row_dict["Channel 1"]
    ch2_config = row_dict["Channel 2"]

    # Fast path: optionally plot *exactly* the externally preselected points.
    # This is used to ensure panels (G) share the same balanced subselection
    # and the same color labels as panels (H/I), computed once in new_figure_1.py.
    if isinstance(preselected_points, dict):
        required = ["ch1_plot", "ch2_plot", "fp_labels_plot"]
        if not all(k in preselected_points for k in required):
            raise ValueError(
                "subpanel_5 preselected_points requires keys: "
                f"{required}. Got keys={list(preselected_points.keys())}"
            )

        ch1_plot = np.asarray(preselected_points["ch1_plot"], dtype=float).ravel()
        ch2_plot = np.asarray(preselected_points["ch2_plot"], dtype=float).ravel()
        fp_labels_plot = np.asarray(preselected_points["fp_labels_plot"], dtype=object).ravel()
        max_value = float(preselected_points.get("max_value", 3000.0))

        if ch1_plot.shape != ch2_plot.shape or ch1_plot.shape[0] != fp_labels_plot.shape[0]:
            raise ValueError(
                "subpanel_5 preselected_points arrays must have matching lengths: "
                f"ch1_plot={ch1_plot.shape}, ch2_plot={ch2_plot.shape}, fp_labels_plot={fp_labels_plot.shape}"
            )

        # Keep only points in the displayed square region (defensive; balanced
        # selection should already enforce this).
        keep_mask = (
            np.isfinite(ch1_plot)
            & np.isfinite(ch2_plot)
            & (ch1_plot >= 0) & (ch1_plot <= max_value)
            & (ch2_plot >= 0) & (ch2_plot <= max_value)
        )
        ch1_plot = ch1_plot[keep_mask]
        ch2_plot = ch2_plot[keep_mask]
        fp_labels_plot = fp_labels_plot[keep_mask]

        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)

        # Get predicted vectors (either provided by wrapper or computed here).
        if predicted_signals is None:
            predicted_signals = compute_predicted_channel_signals(row_dict)

        # Compute unit vectors from the provided pixel subset.
        predicted_vectors = {}
        data_vectors = {}
        for fp_name in fluorophores:
            # Predicted vector (from predicted_signals)
            ch1_signal = predicted_signals[fp_name]["Channel 1"]
            ch2_signal = predicted_signals[fp_name]["Channel 2"]
            pred_vec = np.array([ch1_signal, ch2_signal], dtype=float)
            pred_norm = np.linalg.norm(pred_vec)
            if pred_norm > 0:
                pred_vec = pred_vec / pred_norm
            predicted_vectors[fp_name] = pred_vec

            # Data vector from the same externally selected pixels
            fp_mask = (fp_labels_plot == fp_name)
            if np.any(fp_mask):
                data_vectors[fp_name] = compute_data_vector(ch1_plot[fp_mask], ch2_plot[fp_mask])
            else:
                # Fallback: keep vector well-defined even if a label has no pixels.
                data_vectors[fp_name] = np.array([1.0, 0.0])

        # Scale vectors to reach the configured intensity percentile; arrows are 2x longer.
        ch1_70th = np.percentile(ch1_plot, cfg.vector_scaling_percentile) if ch1_plot.size else 0.0
        ch2_70th = np.percentile(ch2_plot, cfg.vector_scaling_percentile) if ch2_plot.size else 0.0
        max_scale = max(ch1_70th, ch2_70th) * 2.0

        # Compute classification zones (symmetric angle range).
        classification_zones = {}
        zone_angles = {}
        zone_min_distance = getattr(cfg, "classification_zone_min_distance", 500)
        for fp_name in fluorophores:
            data_vec = data_vectors[fp_name]
            half_angle = compute_classification_zone(
                ch1_plot, ch2_plot, fp_labels_plot,
                fp_name, data_vec,
                percentile=cfg.classification_zone_percentile,
                min_distance=zone_min_distance,
            )
            classification_zones[fp_name] = half_angle
            zone_angles[fp_name] = vector_angle(data_vec)

        # Optional overlap adjustment for the 2-FP case.
        if len(fluorophores) == 2 and all(
            fp in classification_zones and classification_zones[fp] is not None for fp in fluorophores
        ):
            fp1, fp2 = fluorophores[0], fluorophores[1]
            vec1_angle = zone_angles[fp1]
            vec2_angle = zone_angles[fp2]
            half1 = classification_zones[fp1]
            half2 = classification_zones[fp2]

            angle_diff = abs(vec1_angle - vec2_angle)
            angle_diff = min(angle_diff, 360 - angle_diff)
            if angle_diff < (half1 + half2):
                overlap_center = (vec1_angle + vec2_angle) / 2.0
                if abs(vec1_angle - vec2_angle) > 180:
                    overlap_center = (overlap_center + 180) % 360

                dist1_to_overlap = abs(vec1_angle - overlap_center)
                dist1_to_overlap = min(dist1_to_overlap, 360 - dist1_to_overlap)
                classification_zones[fp1] = dist1_to_overlap

                dist2_to_overlap = abs(vec2_angle - overlap_center)
                dist2_to_overlap = min(dist2_to_overlap, 360 - dist2_to_overlap)
                classification_zones[fp2] = dist2_to_overlap

        # Plot zones behind points.
        from matplotlib.patches import Wedge, FancyArrowPatch, Arc
        for fp_name in fluorophores:
            color = cfg.fluorophore_colors.get(fp_name, "#000000")
            half_angle_deg = classification_zones.get(fp_name)
            if half_angle_deg is None:
                continue

            data_vec = data_vectors[fp_name]
            vec_angle_deg = vector_angle(data_vec)

            wedge_radius = max_value * 1.5
            wedge = Wedge(
                (0, 0),
                wedge_radius,
                vec_angle_deg - half_angle_deg,
                vec_angle_deg + half_angle_deg,
                color=color,
                alpha=0.3,
                edgecolor="none",
                zorder=1,
            )
            ax.add_patch(wedge)

        # Scatter points colored by fp_labels_plot.
        colors_list = [cfg.fluorophore_colors.get(fp, "#000000") for fp in fp_labels_plot]
        ax.scatter(ch1_plot, ch2_plot, c=colors_list, alpha=0.4, s=2, zorder=2)

        # Plot vectors (predicted dashed, data solid).
        arrow_mutation_scale = 18
        for fp_name in fluorophores:
            color = cfg.fluorophore_colors.get(fp_name, "#000000")

            pred_vec = predicted_vectors[fp_name]
            pred_end_x = float(pred_vec[0]) * max_scale
            pred_end_y = float(pred_vec[1]) * max_scale
            pred_arrow = FancyArrowPatch(
                (0, 0), (pred_end_x, pred_end_y),
                arrowstyle="->",
                mutation_scale=arrow_mutation_scale,
                linestyle="--",
                linewidth=2,
                color=color,
                alpha=0.7,
            )
            ax.add_patch(pred_arrow)

            data_vec = data_vectors[fp_name]
            data_end_x = float(data_vec[0]) * max_scale
            data_end_y = float(data_vec[1]) * max_scale
            data_arrow = FancyArrowPatch(
                (0, 0), (data_end_x, data_end_y),
                arrowstyle="->",
                mutation_scale=arrow_mutation_scale,
                linestyle="-",
                linewidth=2,
                color=color,
                alpha=0.7,
            )
            ax.add_patch(data_arrow)

        # Angle between two vectors, if exactly 2 FPs.
        if len(fluorophores) == 2:
            vec1 = data_vectors[fluorophores[0]]
            vec2 = data_vectors[fluorophores[1]]
            angle_rad = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            vec1_angle_deg = np.degrees(np.arctan2(vec1[1], vec1[0]))
            vec2_angle_deg = np.degrees(np.arctan2(vec2[1], vec2[0]))

            arc_radius = max_scale * 0.5
            arc = Arc(
                (0, 0), arc_radius * 2, arc_radius * 2,
                angle=0, theta1=vec1_angle_deg, theta2=vec2_angle_deg,
                color="black", linewidth=2.0,
            )
            ax.add_patch(arc)

            mid_angle_deg = (vec1_angle_deg + vec2_angle_deg) / 2.0
            label_radius = arc_radius * 1.15
            label_x = label_radius * np.cos(np.radians(mid_angle_deg))
            label_y = label_radius * np.sin(np.radians(mid_angle_deg))
            ax.text(
                label_x,
                label_y,
                f"{angle_deg:.1f}°",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Axis labels
        ax.set_xlabel("Channel 1 Signal", fontsize=12)
        ax.set_ylabel("Channel 2 Signal", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal", adjustable="box")
        return fig, ax

    # ---------- Legacy path (loads images + performs distance/bin subsampling) ----------
    
    # Load data for each fluorophore (no subsampling)
    all_ch1_data = []
    all_ch2_data = []
    fp_labels = []
    
    for fp_name in fluorophores:
        # Load channel 1 data: Use Ch1 from the Channel 1 configuration folder
        ch1_data, _ = load_channel_data(data_dir, fp_name, 
                                     ch1_config["Excitation wavelength"],
                                     ch1_config["emission filter"],
                                     channel_num=1, subsample_factor=None)
        
        # Load channel 2 data: Use Ch1 from the Channel 2 configuration folder
        # (Ch1 is the data channel for each imaging configuration)
        ch2_data, _ = load_channel_data(data_dir, fp_name,
                                     ch2_config["Excitation wavelength"],
                                     ch2_config["emission filter"],
                                     channel_num=1, subsample_factor=None)
        
        # Debug: check data range and type
        print(f"\n{fp_name} data check:")
        print(f"  ch1_data: dtype={ch1_data.dtype}, min={np.min(ch1_data)}, max={np.max(ch1_data)}, mean={np.mean(ch1_data):.2f}")
        print(f"  ch2_data: dtype={ch2_data.dtype}, min={np.min(ch2_data)}, max={np.max(ch2_data)}, mean={np.mean(ch2_data):.2f}")
        
        all_ch1_data.append(ch1_data)
        all_ch2_data.append(ch2_data)
        fp_labels.extend([fp_name] * len(ch1_data))
    
    # Combine all data
    ch1_combined = np.concatenate(all_ch1_data)
    ch2_combined = np.concatenate(all_ch2_data)
    
    # Debug: check combined data
    print(f"\nCombined data check:")
    print(f"  ch1_combined: shape={ch1_combined.shape}, dtype={ch1_combined.dtype}, min={np.min(ch1_combined)}, max={np.max(ch1_combined)}, mean={np.mean(ch1_combined):.2f}")
    print(f"  ch2_combined: shape={ch2_combined.shape}, dtype={ch2_combined.dtype}, min={np.min(ch2_combined)}, max={np.max(ch2_combined)}, mean={np.mean(ch2_combined):.2f}")
    
    # Ensure arrays are 1D and flattened
    if len(ch1_combined.shape) > 1:
        print(f"  WARNING: ch1_combined is not 1D! Flattening...")
        ch1_combined = ch1_combined.flatten()
    if len(ch2_combined.shape) > 1:
        print(f"  WARNING: ch2_combined is not 1D! Flattening...")
        ch2_combined = ch2_combined.flatten()
    
    # NOTE: filter by distance AFTER computing distances (per your request).
    # We want to keep any point that could appear inside the 3000x3000 square,
    # so we use r < 3000*sqrt(2) rather than ch1<3000 and ch2<3000 first.
    max_value = 3000
    max_distance = max_value * np.sqrt(2)

    # Ensure arrays are 1D
    if len(ch1_combined.shape) > 1:
        ch1_combined = ch1_combined.flatten()
    if len(ch2_combined.shape) > 1:
        ch2_combined = ch2_combined.flatten()

    # Compute distance from origin for each point (float to avoid uint16 overflow on squaring)
    ch1_float_all = ch1_combined.astype(np.float64)
    ch2_float_all = ch2_combined.astype(np.float64)
    distances_all = np.sqrt(ch1_float_all**2 + ch2_float_all**2)

    distance_mask = distances_all <= max_distance

    ch1_filtered = ch1_combined[distance_mask]
    ch2_filtered = ch2_combined[distance_mask]
    fp_labels_filtered = np.array(fp_labels, dtype=object)[distance_mask]

    # Compute distance from origin for each filtered point
    ch1_float = ch1_filtered.astype(np.float64)
    ch2_float = ch2_filtered.astype(np.float64)
    
    # Debug: check a few sample calculations
    print(f"\nSample distance calculations:")
    print(f"  Sample point 0: ch1={ch1_float[0]}, ch2={ch2_float[0]}, distance={np.sqrt(ch1_float[0]**2 + ch2_float[0]**2):.2f}")
    if len(ch1_float) > 1000:
        print(f"  Sample point 1000: ch1={ch1_float[1000]}, ch2={ch2_float[1000]}, distance={np.sqrt(ch1_float[1000]**2 + ch2_float[1000]**2):.2f}")
    max_idx = np.argmax(ch1_float**2 + ch2_float**2)
    print(f"  Max distance point: ch1={ch1_float[max_idx]}, ch2={ch2_float[max_idx]}, distance={np.sqrt(ch1_float[max_idx]**2 + ch2_float[max_idx]**2):.2f}")
    
    distances = np.sqrt(ch1_float**2 + ch2_float**2)
    
    # Debug: check distance statistics
    print(f"\nDistance stats:")
    print(f"  min={np.min(distances):.2f}, max={np.max(distances):.2f}, mean={np.mean(distances):.2f}")
    print(f"  Points with distance > 400: {np.sum(distances > 400)}")
    print(f"  Points with distance > 500: {np.sum(distances > 500)}")
    print(f"  Points with distance > 1000: {np.sum(distances > 1000)}")
    print(f"  Points with distance > 2000: {np.sum(distances > 2000)}")
    
    # Bin by distance: <100, <200, <300, etc. up to max_value
    bin_width = 100
    max_distance = max_value * np.sqrt(2)  # Maximum possible distance in rectangular region
    n_bins = int(np.ceil(max_distance / bin_width))
    
    # Create list of arrays, one per bin
    bin_arrays_ch1 = []
    bin_arrays_ch2 = []
    bin_arrays_labels = []
    bin_distances = []  # Track actual distances for debugging
    
    for bin_idx in range(n_bins):
        bin_max = (bin_idx + 1) * bin_width
        prev_bin_max = bin_idx * bin_width
        
        # Create mask for this bin
        if bin_idx == 0:
            bin_mask = distances < bin_max
        else:
            bin_mask = (distances >= prev_bin_max) & (distances < bin_max)
        
        if np.any(bin_mask):
            bin_arrays_ch1.append(ch1_filtered[bin_mask])
            bin_arrays_ch2.append(ch2_filtered[bin_mask])
            bin_arrays_labels.append(fp_labels_filtered[bin_mask])
            bin_distances.append(distances[bin_mask])  # Track distances for this bin
        else:
            # Empty bin - add empty arrays to maintain indexing
            bin_arrays_ch1.append(np.array([], dtype=ch1_filtered.dtype))
            bin_arrays_ch2.append(np.array([], dtype=ch2_filtered.dtype))
            bin_arrays_labels.append(np.array([], dtype=fp_labels_filtered.dtype))
            bin_distances.append(np.array([], dtype=distances.dtype))
    
    # Track lengths and means before subsampling for debugging (separate for each FP)
    bin_lengths_before_mCherry = []
    bin_mean_ch1_before_mCherry = []
    bin_mean_ch2_before_mCherry = []
    bin_mean_distance_before_mCherry = []
    
    bin_lengths_before_mNeptune = []
    bin_mean_ch1_before_mNeptune = []
    bin_mean_ch2_before_mNeptune = []
    bin_mean_distance_before_mNeptune = []
    
    for bin_idx in range(n_bins):
        ch1_bin = bin_arrays_ch1[bin_idx]
        ch2_bin = bin_arrays_ch2[bin_idx]
        dist_bin = bin_distances[bin_idx]
        labels_bin = bin_arrays_labels[bin_idx]
        
        # Separate by fluorophore
        mCherry_mask = labels_bin == "mCherry"
        mNeptune_mask = labels_bin == "mNeptune"
        
        # mCherry stats
        if np.any(mCherry_mask):
            bin_lengths_before_mCherry.append(np.sum(mCherry_mask))
            bin_mean_ch1_before_mCherry.append(np.mean(ch1_bin[mCherry_mask]))
            bin_mean_ch2_before_mCherry.append(np.mean(ch2_bin[mCherry_mask]))
            bin_mean_distance_before_mCherry.append(np.mean(dist_bin[mCherry_mask]))
        else:
            bin_lengths_before_mCherry.append(0)
            bin_mean_ch1_before_mCherry.append(np.nan)
            bin_mean_ch2_before_mCherry.append(np.nan)
            bin_mean_distance_before_mCherry.append(np.nan)
        
        # mNeptune stats
        if np.any(mNeptune_mask):
            bin_lengths_before_mNeptune.append(np.sum(mNeptune_mask))
            bin_mean_ch1_before_mNeptune.append(np.mean(ch1_bin[mNeptune_mask]))
            bin_mean_ch2_before_mNeptune.append(np.mean(ch2_bin[mNeptune_mask]))
            bin_mean_distance_before_mNeptune.append(np.mean(dist_bin[mNeptune_mask]))
        else:
            bin_lengths_before_mNeptune.append(0)
            bin_mean_ch1_before_mNeptune.append(np.nan)
            bin_mean_ch2_before_mNeptune.append(np.nan)
            bin_mean_distance_before_mNeptune.append(np.nan)
    
    # Sample the same number from each bin (per fluorophore)
    samples_per_bin = 300
    
    # Subsample each bin and collect results (sample per-bin per-fluorophore)
    ch1_plot_list = []
    ch2_plot_list = []
    fp_labels_plot_list = []

    # Track mCherry after-sampling (for the debug CSV)
    bin_lengths_after_mCherry = []
    bin_mean_ch1_after_mCherry = []
    bin_mean_ch2_after_mCherry = []

    for bin_idx in range(n_bins):
        ch1_bin = bin_arrays_ch1[bin_idx]
        ch2_bin = bin_arrays_ch2[bin_idx]
        labels_bin = bin_arrays_labels[bin_idx]

        if len(ch1_bin) == 0:
            bin_lengths_after_mCherry.append(0)
            bin_mean_ch1_after_mCherry.append(np.nan)
            bin_mean_ch2_after_mCherry.append(np.nan)
            continue

        # We'll build this bin's sampled points by fluorophore, then append once.
        ch1_bin_sampled_list = []
        ch2_bin_sampled_list = []
        labels_bin_sampled_list = []

        for fp_name in fluorophores:
            fp_mask = labels_bin == fp_name
            if not np.any(fp_mask):
                continue

            ch1_fp = ch1_bin[fp_mask]
            ch2_fp = ch2_bin[fp_mask]
            n_fp = len(ch1_fp)
            n_take = min(samples_per_bin, n_fp)

            if n_fp > n_take:
                fp_indices = np.random.choice(n_fp, n_take, replace=False)
                ch1_fp = ch1_fp[fp_indices]
                ch2_fp = ch2_fp[fp_indices]

            ch1_bin_sampled_list.append(ch1_fp)
            ch2_bin_sampled_list.append(ch2_fp)
            labels_bin_sampled_list.append(np.array([fp_name] * len(ch1_fp), dtype=object))

        if len(ch1_bin_sampled_list) == 0:
            bin_lengths_after_mCherry.append(0)
            bin_mean_ch1_after_mCherry.append(np.nan)
            bin_mean_ch2_after_mCherry.append(np.nan)
            continue

        ch1_sampled = np.concatenate(ch1_bin_sampled_list)
        ch2_sampled = np.concatenate(ch2_bin_sampled_list)
        labels_sampled = np.concatenate(labels_bin_sampled_list)

        ch1_plot_list.append(ch1_sampled)
        ch2_plot_list.append(ch2_sampled)
        fp_labels_plot_list.append(labels_sampled)

        # Track mCherry separately
        mCherry_mask_sampled = labels_sampled == "mCherry"
        if np.any(mCherry_mask_sampled):
            bin_lengths_after_mCherry.append(int(np.sum(mCherry_mask_sampled)))
            bin_mean_ch1_after_mCherry.append(float(np.mean(ch1_sampled[mCherry_mask_sampled])))
            bin_mean_ch2_after_mCherry.append(float(np.mean(ch2_sampled[mCherry_mask_sampled])))
        else:
            bin_lengths_after_mCherry.append(0)
            bin_mean_ch1_after_mCherry.append(np.nan)
            bin_mean_ch2_after_mCherry.append(np.nan)
    
    # Save debugging info to CSV (only for emission-based row, mCherry only)
    row_name = row_dict.get("name", "")
    if "emission" in row_name.lower():
        import pandas as pd
        debug_df = pd.DataFrame({
            'bin_index': range(n_bins),
            'bin_min_distance': [i * bin_width for i in range(n_bins)],
            'bin_max_distance': [(i + 1) * bin_width for i in range(n_bins)],
            'mCherry_length_before_subsampling': bin_lengths_before_mCherry,
            'mCherry_mean_distance_before_subsampling': bin_mean_distance_before_mCherry,
            'mCherry_mean_ch1_before_subsampling': bin_mean_ch1_before_mCherry,
            'mCherry_mean_ch2_before_subsampling': bin_mean_ch2_before_mCherry,
            'mCherry_length_after_subsampling': bin_lengths_after_mCherry,
            'mCherry_mean_ch1_after_subsampling': bin_mean_ch1_after_mCherry,
            'mCherry_mean_ch2_after_subsampling': bin_mean_ch2_after_mCherry
        })
        debug_file = 'subpanel5_subsampling_debug.csv'
        debug_df.to_csv(debug_file, index=False)
        print(f"Saved subsampling debug info to {debug_file} (emission-based row, mCherry only)")
    
    # Combine all subsampled arrays efficiently (single concatenation)
    if len(ch1_plot_list) > 0:
        ch1_plot = np.concatenate(ch1_plot_list)
        ch2_plot = np.concatenate(ch2_plot_list)
        fp_labels_plot = np.concatenate(fp_labels_plot_list)
    else:
        # Fallback if no bins had points
        ch1_plot = ch1_filtered
        ch2_plot = ch2_filtered
        fp_labels_plot = fp_labels_filtered
    
    # Set axis limits to show filtered region (square plot)
    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)
    
    # Get predicted vectors from subpanel 4 (or use externally provided predictions)
    if predicted_signals is None:
        predicted_signals = compute_predicted_channel_signals(row_dict)
    
    # Compute unit vectors
    predicted_vectors = {}
    data_vectors = {}
    
    for i, fp_name in enumerate(fluorophores):
        # Predicted vector (from subpanel 4)
        ch1_signal = predicted_signals[fp_name]["Channel 1"]
        ch2_signal = predicted_signals[fp_name]["Channel 2"]
        pred_vec = np.array([ch1_signal, ch2_signal])
        pred_vec = pred_vec / np.linalg.norm(pred_vec)  # Normalize
        predicted_vectors[fp_name] = pred_vec
        
        # Data vector (from actual data)
        data_vec = compute_data_vector(all_ch1_data[i], all_ch2_data[i])
        data_vectors[fp_name] = data_vec
    
    # Scale vectors to reach 70th percentile, then make arrows 2x as long
    # Use filtered data for scaling
    ch1_70th = np.percentile(ch1_plot, cfg.vector_scaling_percentile)
    ch2_70th = np.percentile(ch2_plot, cfg.vector_scaling_percentile)
    max_scale = max(ch1_70th, ch2_70th) * 2.0  # Make arrows 2x as long
    
    # Compute classification zones (symmetric angle range)
    # Use subsampled data and filter pixels dimmer than 500
    classification_zones = {}
    zone_angles = {}  # Store vector angles for overlap detection
    for i, fp_name in enumerate(fluorophores):
        data_vec = data_vectors[fp_name]
        half_angle = compute_classification_zone(ch1_plot, ch2_plot, fp_labels_plot, 
                                                 fp_name, data_vec, 
                                                 percentile=cfg.classification_zone_percentile, 
                                                 min_distance=500)
        classification_zones[fp_name] = half_angle
        vec_angle_deg = vector_angle(data_vec)
        zone_angles[fp_name] = vec_angle_deg
    
    # Check for overlapping zones and adjust boundaries to center of overlap
    if len(fluorophores) == 2 and all(fp in classification_zones and classification_zones[fp] is not None for fp in fluorophores):
        fp1, fp2 = fluorophores[0], fluorophores[1]
        vec1_angle = zone_angles[fp1]
        vec2_angle = zone_angles[fp2]
        half1 = classification_zones[fp1]
        half2 = classification_zones[fp2]
        
        # Compute zone boundaries (handle wrap-around at 0/360)
        zone1_lower = (vec1_angle - half1) % 360
        zone1_upper = (vec1_angle + half1) % 360
        zone2_lower = (vec2_angle - half2) % 360
        zone2_upper = (vec2_angle + half2) % 360
        
        # Check for overlap (simplified: if zones are close enough, they overlap)
        # Compute angular distance between zone centers
        angle_diff = abs(vec1_angle - vec2_angle)
        angle_diff = min(angle_diff, 360 - angle_diff)  # Handle wrap-around
        
        if angle_diff < (half1 + half2):
            # Zones overlap - set boundary at center of overlap
            overlap_center = (vec1_angle + vec2_angle) / 2.0
            # Adjust to handle wrap-around
            if abs(vec1_angle - vec2_angle) > 180:
                overlap_center = (overlap_center + 180) % 360
            
            # Adjust half-angles so boundaries meet at overlap center
            # For fp1: extend to overlap_center
            dist1_to_overlap = abs(vec1_angle - overlap_center)
            dist1_to_overlap = min(dist1_to_overlap, 360 - dist1_to_overlap)
            classification_zones[fp1] = dist1_to_overlap
            
            # For fp2: extend to overlap_center
            dist2_to_overlap = abs(vec2_angle - overlap_center)
            dist2_to_overlap = min(dist2_to_overlap, 360 - dist2_to_overlap)
            classification_zones[fp2] = dist2_to_overlap
            
            print(f"Subpanel 5 - Zones overlap detected. Setting boundaries at overlap center: {overlap_center:.2f}°")
    
    # Plot classification zones FIRST (so they appear behind scatter points)
    from matplotlib.patches import Wedge
    for fp_name in fluorophores:
        color = cfg.fluorophore_colors.get(fp_name, "#000000")
        if fp_name in classification_zones and classification_zones[fp_name] is not None:
            half_angle_deg = classification_zones[fp_name]
            data_vec = data_vectors[fp_name]
            vec_angle_deg = vector_angle(data_vec)
            
            print(f"Subpanel 5 - {fp_name}: half_angle={half_angle_deg:.2f}°, vec_angle={vec_angle_deg:.2f}°")
            
            # Create wedge from origin extending beyond max_scale to cover the plot
            # Use a large radius to ensure it covers the entire plot area
            wedge_radius = max_value * 1.5  # Extend beyond plot limits
            wedge = Wedge((0, 0), wedge_radius,
                         vec_angle_deg - half_angle_deg,
                         vec_angle_deg + half_angle_deg,
                         color=color, alpha=0.3, edgecolor='none', zorder=1)
            ax.add_patch(wedge)
        else:
            print(f"Subpanel 5 - {fp_name}: No classification zone computed (value is None)")
    
    # Create plot colored by FP with intermediate transparency (on top of zones)
    colors_list = [cfg.fluorophore_colors.get(fp, "#000000") for fp in fp_labels_plot]
    ax.scatter(ch1_plot, ch2_plot, c=colors_list, alpha=0.4, s=2, zorder=2)
    
    # Plot vectors - use FancyArrowPatch for both so arrowheads are identical
    from matplotlib.patches import FancyArrowPatch
    # Use a consistent mutation_scale for both arrows to ensure identical arrowheads
    # Original arrows used head_width=max_scale * 0.05, so mutation_scale should match that
    arrow_mutation_scale = 18  # Adjusted to match original arrowhead size
    
    for fp_name in fluorophores:
        color = cfg.fluorophore_colors.get(fp_name, "#000000")
        
        # Predicted vector (dashed)
        pred_vec = predicted_vectors[fp_name]
        pred_end_x = pred_vec[0] * max_scale
        pred_end_y = pred_vec[1] * max_scale
        pred_arrow = FancyArrowPatch((0, 0), (pred_end_x, pred_end_y),
                                     arrowstyle='->', mutation_scale=arrow_mutation_scale,
                                     linestyle='--', linewidth=2, color=color, alpha=0.7)
        ax.add_patch(pred_arrow)
        
        # Data vector (solid) - identical arrowhead size
        data_vec = data_vectors[fp_name]
        data_end_x = data_vec[0] * max_scale
        data_end_y = data_vec[1] * max_scale
        data_arrow = FancyArrowPatch((0, 0), (data_end_x, data_end_y),
                                     arrowstyle='->', mutation_scale=arrow_mutation_scale,
                                     linestyle='-', linewidth=2, color=color, alpha=0.7)
        ax.add_patch(data_arrow)
    
    # Compute and plot angle between vectors
    if len(fluorophores) == 2:
        vec1 = data_vectors[fluorophores[0]]
        vec2 = data_vectors[fluorophores[1]]
        angle_rad = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # Compute angles of both vectors (in degrees, measured from positive x-axis)
        vec1_angle_deg = np.degrees(np.arctan2(vec1[1], vec1[0]))
        vec2_angle_deg = np.degrees(np.arctan2(vec2[1], vec2[0]))
        
        # Draw arc between the two vectors (larger and further from origin)
        arc_radius = max_scale * 0.5  # Larger radius
        arc = Arc((0, 0), arc_radius * 2, arc_radius * 2, 
                 angle=0, theta1=vec1_angle_deg, theta2=vec2_angle_deg, 
                 color='black', linewidth=2.0)  # Thicker line
        ax.add_patch(arc)
        
        # Label angle (on outside of arc, further from origin)
        mid_angle_deg = (vec1_angle_deg + vec2_angle_deg) / 2
        label_radius = arc_radius * 1.15  # Outside the arc, further from origin
        label_x = label_radius * np.cos(np.radians(mid_angle_deg))
        label_y = label_radius * np.sin(np.radians(mid_angle_deg))
        ax.text(label_x, label_y, f'{angle_deg:.1f}°', 
               ha='center', va='center', fontsize=12,  # Larger font
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    
    # Set labels
    ax.set_xlabel("Channel 1 Signal", fontsize=12)
    ax.set_ylabel("Channel 2 Signal", fontsize=12)
    
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Ensure square aspect ratio (already set limits to be equal, but enforce it)
    ax.set_aspect('equal', adjustable='box')
    
    return fig, ax


def subpanel_6(row_dict, ax=None, data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619"):
    """
    Generate subpanel 6: Colored overlay image from first FP source.
    
    Shows one composite image (Channel 1 and Channel 2 overlaid), where each channel
    is tinted by its excitation color and added together in RGB space.
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (if None, creates a new axis)
    data_dir : str
        Path to data directory
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    return _subpanel_overlay_zoom(row_dict, fp_index=0, ax=ax, data_dir=data_dir)


def subpanel_7_overlay_image(row_dict, ax=None, data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619"):
    """
    Generate subpanel 7: Colored overlay image from second FP source (zoomed).
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    return _subpanel_overlay_zoom(row_dict, fp_index=1, ax=ax, data_dir=data_dir)


def subpanel_7(row_dict, axes=None, data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619"):
    """
    Generate subpanel 7: Histogram of per-vector pixel angular offsets (for classification debugging).

    This is designed to help troubleshoot classification thresholds.
    For each fluorophore's *data vector* (estimated from its own pixels), we compute the angle
    between every pixel vector (from both fluorophores) and that reference vector:
        offset_deg = angle(pixel_vector, fp_vector) in [0, 90]

    We then plot one subplot per reference vector (stacked vertically). Each subplot contains
    overlaid histograms for the true-label pixel groups (one color per fluorophore).

    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
    axes : array-like of matplotlib.axes.Axes, optional
        If provided, must have length == number of fluorophores (stacked vertically).
        If None, a new figure+axes are created.
    data_dir : str
        Path to data directory

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    fluorophores = row_dict["Fluorophores"]
    ch1_config = row_dict["Channel 1"]
    ch2_config = row_dict["Channel 2"]

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

    # Load data for each fluorophore (true labels)
    all_ch1_data = []
    all_ch2_data = []
    fp_labels = []
    for fp_name in fluorophores:
        ch1_data, _ = load_channel_data(
            data_dir, fp_name,
            ch1_config["Excitation wavelength"],
            ch1_config["emission filter"],
            channel_num=1, subsample_factor=None
        )
        ch2_data, _ = load_channel_data(
            data_dir, fp_name,
            ch2_config["Excitation wavelength"],
            ch2_config["emission filter"],
            channel_num=1, subsample_factor=None
        )
        all_ch1_data.append(ch1_data)
        all_ch2_data.append(ch2_data)
        fp_labels.extend([fp_name] * len(ch1_data))

    ch1_combined = np.concatenate(all_ch1_data).astype(np.float64)
    ch2_combined = np.concatenate(all_ch2_data).astype(np.float64)
    fp_labels_array = np.array(fp_labels, dtype=object)

    # Basic filtering: drop origin pixels (undefined angle)
    distances = np.sqrt(ch1_combined ** 2 + ch2_combined ** 2)
    valid_mask = distances > 0
    ch1_valid = ch1_combined[valid_mask]
    ch2_valid = ch2_combined[valid_mask]
    labels_valid = fp_labels_array[valid_mask]

    # Compute data vectors for each FP (unit vectors)
    data_vectors = {}
    for fp_idx, fp_name in enumerate(fluorophores):
        data_vectors[fp_name] = compute_data_vector(all_ch1_data[fp_idx], all_ch2_data[fp_idx])

    # Histogram settings
    bin_size = getattr(cfg, "angle_histogram_bin_size_degrees", 1)
    bins = np.arange(0, 90 + bin_size, bin_size)
    fp_colors = {fp: cfg.fluorophore_colors.get(fp, "#808080") for fp in fluorophores}

    # Optional: compute per-FP "keep" zone threshold (single-sided, since offset is >= 0)
    # This matches how we'd classify by "angle to vector < threshold" in 2D.
    zone_min_distance = getattr(cfg, "classification_zone_min_distance", 500)
    half_angles = {}
    for fp_name in fluorophores:
        try:
            half_angles[fp_name] = compute_classification_zone(
                ch1_valid, ch2_valid, labels_valid,
                fp_name, data_vectors[fp_name],
                percentile=cfg.classification_zone_percentile,
                min_distance=zone_min_distance
            )
        except Exception:
            half_angles[fp_name] = None

    # Plot: one axis per reference FP vector
    max_y = 0
    for ax_sub, ref_fp in zip(axes, fluorophores):
        ref_vec = data_vectors[ref_fp]
        offsets_deg_all = compute_angle_to_vector(ch1_valid, ch2_valid, ref_vec)

        # Shade "kept" region for this reference FP (0..threshold)
        thresh = half_angles.get(ref_fp)
        if thresh is not None:
            ax_sub.axvspan(0, thresh, color=fp_colors.get(ref_fp, "#808080"), alpha=0.15, zorder=0)
            ax_sub.axvline(thresh, color=fp_colors.get(ref_fp, "#808080"), linestyle=":", linewidth=2, alpha=0.9, zorder=3)

        # Overlaid histograms by true label
        for true_fp in fluorophores:
            mask = labels_valid == true_fp
            fp_offsets = offsets_deg_all[mask]
            fp_offsets = fp_offsets[~np.isnan(fp_offsets)]

            if len(fp_offsets) == 0:
                continue
            hist, _ = np.histogram(fp_offsets, bins=bins)
            max_y = max(max_y, int(np.max(hist)))
            ax_sub.bar(
                bins[:-1], hist, width=bin_size,
                color=fp_colors.get(true_fp, "#808080"),
                alpha=0.55, edgecolor="none",
                label=true_fp if ref_fp == fluorophores[0] else None,
                zorder=1
            )

        ax_sub.set_ylabel(f"to {ref_fp}\ncount", fontsize=10)
        ax_sub.grid(True, alpha=0.25, axis="y")
        ax_sub.spines["top"].set_visible(False)
        ax_sub.spines["right"].set_visible(False)

    # Shared axes formatting
    for ax_sub in axes:
        ax_sub.set_xlim(0, 90)
        if max_y > 0:
            ax_sub.set_ylim(0, max_y * 1.15)

    axes[-1].set_xlabel("Angle to reference vector (degrees)", fontsize=12)
    title = row_dict.get("name", "row")
    axes[0].set_title(f"{title}: pixel angular offsets to each vector", fontsize=12, fontweight="bold")

    # Legend once (top axis), matching fluorophore colors
    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) > 0:
        axes[0].legend(handles, labels, loc="upper right", fontsize=9, frameon=True)

    plt.tight_layout()
    return fig, axes


def compute_angle_to_vector(ch1_data, ch2_data, vector):
    """
    Compute angles between pixel vectors and a reference vector (0-90 degrees).
    
    Parameters
    ----------
    ch1_data : np.ndarray
        Channel 1 pixel intensities
    ch2_data : np.ndarray
        Channel 2 pixel intensities
    vector : np.ndarray
        Reference unit vector [ch1_component, ch2_component]
        
    Returns
    -------
    np.ndarray
        Angles in degrees between each pixel vector and the reference vector (0-90 degrees)
    """
    # Normalize pixel vectors
    magnitudes = np.sqrt(ch1_data**2 + ch2_data**2)
    valid_mask = magnitudes > 0
    
    # Initialize angles array
    angles = np.full(len(ch1_data), np.nan)
    
    if np.any(valid_mask):
        # Normalize valid pixel vectors
        ch1_normalized = ch1_data[valid_mask] / magnitudes[valid_mask]
        ch2_normalized = ch2_data[valid_mask] / magnitudes[valid_mask]
        
        # Compute dot product with reference vector
        dot_products = ch1_normalized * vector[0] + ch2_normalized * vector[1]
        # Clip to [-1, 1] to avoid numerical errors
        dot_products = np.clip(dot_products, -1.0, 1.0)
        
        # Compute angles in radians, then convert to degrees
        angles_rad = np.arccos(dot_products)
        angles_deg = np.degrees(angles_rad)
        # Ensure angles are in 0-90 range (take minimum of angle and 180-angle)
        angles_deg = np.minimum(angles_deg, 180 - angles_deg)
        angles[valid_mask] = angles_deg
    
    return angles


# vector_angle is now imported from figure_scatterplot_helpers


# compute_classification_zone is now imported from figure_scatterplot_helpers
# Note: Debug output was removed from the shared version


def subpanel_8(row_dict, ax=None, data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619"):
    """
    Generate subpanel 8: Histogram of pixel angles with vector lines.
    
    For a single row, creates a histogram showing the distribution of pixel angles
    (0-360 degrees), colored by fluorophore label. Overlays lines for predicted
    and actual vectors.
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (will create if None)
    data_dir : str
        Path to data directory
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    fluorophores = row_dict["Fluorophores"]
    ch1_config = row_dict["Channel 1"]
    ch2_config = row_dict["Channel 2"]
    
    # Load data for each fluorophore
    all_ch1_data = []
    all_ch2_data = []
    fp_labels = []
    
    for fp_name in fluorophores:
        # Load channel 1 data
        ch1_data, _ = load_channel_data(data_dir, fp_name, 
                                    ch1_config["Excitation wavelength"],
                                    ch1_config["emission filter"],
                                    channel_num=1, subsample_factor=None)
        
        # Load channel 2 data
        ch2_data, _ = load_channel_data(data_dir, fp_name,
                                    ch2_config["Excitation wavelength"],
                                    ch2_config["emission filter"],
                                    channel_num=1, subsample_factor=None)
        
        all_ch1_data.append(ch1_data)
        all_ch2_data.append(ch2_data)
        fp_labels.extend([fp_name] * len(ch1_data))
    
    # Combine all data
    ch1_combined = np.concatenate(all_ch1_data)
    ch2_combined = np.concatenate(all_ch2_data)
    fp_labels_array = np.array(fp_labels, dtype=object)
    
    # Filter by distance (same as subpanel 5)
    max_value = 3000
    max_distance = max_value * np.sqrt(2)
    ch1_float = ch1_combined.astype(np.float64)
    ch2_float = ch2_combined.astype(np.float64)
    distances = np.sqrt(ch1_float**2 + ch2_float**2)
    distance_mask = distances <= max_distance
    
    ch1_filtered = ch1_combined[distance_mask]
    ch2_filtered = ch2_combined[distance_mask]
    fp_labels_filtered = fp_labels_array[distance_mask]
    
    # Filter out pixels at/near origin to avoid 0-degree spike
    min_distance = 10  # Minimum distance from origin
    valid_mask = distances[distance_mask] >= min_distance
    
    ch1_valid = ch1_filtered[valid_mask]
    ch2_valid = ch2_filtered[valid_mask]
    fp_labels_valid = fp_labels_filtered[valid_mask]
    
    # Apply same subsampling as subpanel 5 to get subsampled data for zone computation
    # Compute distances for valid pixels
    ch1_float_valid = ch1_valid.astype(np.float64)
    ch2_float_valid = ch2_valid.astype(np.float64)
    distances_valid = np.sqrt(ch1_float_valid**2 + ch2_float_valid**2)
    
    # Bin by distance (same as subpanel 5)
    bin_width = 100
    max_distance_valid = max_value * np.sqrt(2)
    n_bins = int(np.ceil(max_distance_valid / bin_width))
    samples_per_bin = 300
    
    ch1_plot_list = []
    ch2_plot_list = []
    fp_labels_plot_list = []
    
    for bin_idx in range(n_bins):
        bin_max = (bin_idx + 1) * bin_width
        prev_bin_max = bin_idx * bin_width
        
        if bin_idx == 0:
            bin_mask = distances_valid < bin_max
        else:
            bin_mask = (distances_valid >= prev_bin_max) & (distances_valid < bin_max)
        
        if np.any(bin_mask):
            ch1_bin = ch1_valid[bin_mask]
            ch2_bin = ch2_valid[bin_mask]
            labels_bin = fp_labels_valid[bin_mask]
            
            # Sample per fluorophore per bin
            for fp_name in fluorophores:
                fp_mask = labels_bin == fp_name
                if not np.any(fp_mask):
                    continue
                
                ch1_fp = ch1_bin[fp_mask]
                ch2_fp = ch2_bin[fp_mask]
                n_fp = len(ch1_fp)
                n_take = min(samples_per_bin, n_fp)
                
                if n_fp > n_take:
                    fp_indices = np.random.choice(n_fp, n_take, replace=False)
                    ch1_plot_list.append(ch1_fp[fp_indices])
                    ch2_plot_list.append(ch2_fp[fp_indices])
                    fp_labels_plot_list.append(np.array([fp_name] * n_take, dtype=object))
                else:
                    ch1_plot_list.append(ch1_fp)
                    ch2_plot_list.append(ch2_fp)
                    fp_labels_plot_list.append(np.array([fp_name] * n_fp, dtype=object))
    
    # Combine subsampled data
    if len(ch1_plot_list) > 0:
        ch1_plot = np.concatenate(ch1_plot_list)
        ch2_plot = np.concatenate(ch2_plot_list)
        fp_labels_plot = np.concatenate(fp_labels_plot_list)
    else:
        ch1_plot = ch1_valid
        ch2_plot = ch2_valid
        fp_labels_plot = fp_labels_valid
    
    # Get predicted vectors (from subpanel 4)
    predicted_signals = compute_predicted_channel_signals(row_dict)
    predicted_vectors = {}
    data_vectors = {}
    
    for i, fp_name in enumerate(fluorophores):
        # Predicted vector
        ch1_signal = predicted_signals[fp_name]["Channel 1"]
        ch2_signal = predicted_signals[fp_name]["Channel 2"]
        pred_vec = np.array([ch1_signal, ch2_signal])
        pred_vec = pred_vec / np.linalg.norm(pred_vec)
        predicted_vectors[fp_name] = pred_vec
        
        # Data vector (from actual data)
        data_vec = compute_data_vector(all_ch1_data[i], all_ch2_data[i])
        data_vectors[fp_name] = data_vec
    
    # Compute classification zones (symmetric angle range)
    # Use subsampled data and filter pixels dimmer than 500
    classification_zones = {}
    zone_angles = {}  # Store vector angles for overlap detection
    for fp_name in fluorophores:
        data_vec = data_vectors[fp_name]
        half_angle = compute_classification_zone(ch1_plot, ch2_plot, fp_labels_plot, 
                                                 fp_name, data_vec, 
                                                 percentile=cfg.classification_zone_percentile, 
                                                 min_distance=500)
        classification_zones[fp_name] = half_angle
        data_angle_rad = np.arctan2(data_vec[1], data_vec[0])
        data_angle_deg = np.degrees(data_angle_rad)
        data_angle_deg = np.abs(data_angle_deg) % 180
        data_angle_deg = data_angle_deg if data_angle_deg <= 90 else 180 - data_angle_deg
        zone_angles[fp_name] = data_angle_deg
    
    # Check for overlapping zones and adjust boundaries to center of overlap (in 0-90 space)
    if len(fluorophores) == 2 and all(fp in classification_zones and classification_zones[fp] is not None for fp in fluorophores):
        fp1, fp2 = fluorophores[0], fluorophores[1]
        vec1_angle = zone_angles[fp1]
        vec2_angle = zone_angles[fp2]
        half1 = classification_zones[fp1]
        half2 = classification_zones[fp2]
        
        # Compute zone boundaries in 0-90 space
        zone1_lower = max(0, vec1_angle - half1)
        zone1_upper = min(90, vec1_angle + half1)
        zone2_lower = max(0, vec2_angle - half2)
        zone2_upper = min(90, vec2_angle + half2)
        
        # Check for overlap
        if not (zone1_upper < zone2_lower or zone2_upper < zone1_lower):
            # Zones overlap - set boundary at center of overlap
            overlap_lower = max(zone1_lower, zone2_lower)
            overlap_upper = min(zone1_upper, zone2_upper)
            overlap_center = (overlap_lower + overlap_upper) / 2.0
            
            # Adjust half-angles so boundaries meet at overlap center
            classification_zones[fp1] = abs(vec1_angle - overlap_center)
            classification_zones[fp2] = abs(vec2_angle - overlap_center)
            
            print(f"Subpanel 8 - Zones overlap detected. Setting boundaries at overlap center: {overlap_center:.2f}°")
    
    # Create histogram bins (0 to 90 degrees) - doubled number of bins
    bins = np.linspace(0, 90, 91)  # 1 degree bins (doubled from 46)
    
    # Get colors for fluorophores
    fp_colors = {fp: cfg.fluorophore_colors.get(fp, "#808080") for fp in fluorophores}
    
    # Compute pixel angles (from x-axis, constrained to 0-90 degrees)
    # For pixels in first quadrant, use angle as-is; for others, map to 0-90
    pixel_angles_rad = np.arctan2(ch2_valid, ch1_valid)
    pixel_angles_deg = np.degrees(pixel_angles_rad)
    # Map all angles to 0-90 range (take absolute and fold)
    pixel_angles_deg = np.abs(pixel_angles_deg) % 180
    pixel_angles_deg = np.where(pixel_angles_deg > 90, 180 - pixel_angles_deg, pixel_angles_deg)
    
    # Draw classification zones FIRST (before bars so they appear behind)
    for fp_name in fluorophores:
        data_vec = data_vectors[fp_name]
        data_angle_rad = np.arctan2(data_vec[1], data_vec[0])
        data_angle_deg = np.degrees(data_angle_rad)
        data_angle_deg = np.abs(data_angle_deg) % 180
        data_angle_deg = data_angle_deg if data_angle_deg <= 90 else 180 - data_angle_deg
        
        # Draw classification zone (90% symmetric angle range) with shading
        if fp_name in classification_zones and classification_zones[fp_name] is not None:
            half_angle = classification_zones[fp_name]
            # Zone is symmetric around the data vector angle
            zone_lower = max(0, data_angle_deg - half_angle)
            zone_upper = min(90, data_angle_deg + half_angle)
            
            print(f"Subpanel 8 - {fp_name}: half_angle={half_angle:.2f}°, data_angle={data_angle_deg:.2f}°, zone=[{zone_lower:.2f}°, {zone_upper:.2f}°]")
            
            # Shade between the lines - draw FIRST so it's behind bars
            ax.axvspan(zone_lower, zone_upper, color=fp_colors[fp_name], alpha=0.25, 
                      zorder=0, label=f'{fp_name} {cfg.classification_zone_percentile}% zone')
    
    # Create overlapping histograms for each fluorophore (as bars)
    max_hist_value = 0
    for fp_name in fluorophores:
        fp_mask = fp_labels_valid == fp_name
        fp_angles = pixel_angles_deg[fp_mask]
        
        if len(fp_angles) > 0:
            hist, _ = np.histogram(fp_angles, bins=bins)
            max_hist_value = max(max_hist_value, np.max(hist))
            bin_width = bins[1] - bins[0]
            ax.bar(bins[:-1], hist, width=bin_width,
                   label=f'{fp_name} pixels',
                   color=fp_colors[fp_name], alpha=0.6, edgecolor='none', zorder=1)
    
    # Set y-axis limits with space for labels
    ax.set_ylim(0, max_hist_value * 1.15)
    
    # Overlay vector lines (show where each vector's angle is, mapped to 0-90)
    # Order: mean (data vector) first, then predicted
    for fp_name in fluorophores:
        # Data vector angle (mean)
        data_vec = data_vectors[fp_name]
        data_angle_rad = np.arctan2(data_vec[1], data_vec[0])
        data_angle_deg = np.degrees(data_angle_rad)
        data_angle_deg = np.abs(data_angle_deg) % 180
        data_angle_deg = data_angle_deg if data_angle_deg <= 90 else 180 - data_angle_deg
        ax.axvline(data_angle_deg, color=fp_colors[fp_name], linestyle='-', 
                  linewidth=2, label=f'{fp_name} mean', alpha=0.8)
        
        # Predicted vector angle
        pred_vec = predicted_vectors[fp_name]
        pred_angle_rad = np.arctan2(pred_vec[1], pred_vec[0])
        pred_angle_deg = np.degrees(pred_angle_rad)
        pred_angle_deg = np.abs(pred_angle_deg) % 180
        pred_angle_deg = pred_angle_deg if pred_angle_deg <= 90 else 180 - pred_angle_deg
        ax.axvline(pred_angle_deg, color=fp_colors[fp_name], linestyle='--', 
                  linewidth=2, label=f'{fp_name} predicted', alpha=0.8)
        
        # Draw classification zone boundary lines (shading already drawn above)
        if fp_name in classification_zones and classification_zones[fp_name] is not None:
            half_angle = classification_zones[fp_name]
            # Zone is symmetric around the data vector angle
            zone_lower = max(0, data_angle_deg - half_angle)
            zone_upper = min(90, data_angle_deg + half_angle)
            
            # Draw two vertical lines
            ax.axvline(zone_lower, color=fp_colors[fp_name], linestyle=':', 
                      linewidth=1.5, alpha=0.8, zorder=3)
            ax.axvline(zone_upper, color=fp_colors[fp_name], linestyle=':', 
                      linewidth=1.5, alpha=0.8, zorder=3)
            
            # Add label above the zone (like in emission spectra)
            # Use x-axis transform: x in data coordinates, y in axes coordinates (0-1)
            label_x = (zone_lower + zone_upper) / 2.0
            ax.text(label_x, 1.02, f'{fp_name} {cfg.classification_zone_percentile}%', 
                   ha='center', va='bottom', fontsize=9, color=fp_colors[fp_name], 
                   transform=ax.get_xaxis_transform(), zorder=4, clip_on=False)
        else:
            print(f"Subpanel 8 - {fp_name}: No classification zone computed (value is None)")
    
    # Set labels and title
    ax.set_xlabel("Angle to Vector (degrees)", fontsize=12)
    ax.set_ylabel("Pixel Count", fontsize=12)
    ax.set_title(f"{row_dict['name']}", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 90)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Get legend handles and labels, then reorder: pixels, mean, predicted
    handles, labels = ax.get_legend_handles_labels()
    
    # Separate by type
    pixel_handles = []
    pixel_labels = []
    mean_handles = []
    mean_labels = []
    pred_handles = []
    pred_labels = []
    
    for handle, label in zip(handles, labels):
        if 'pixels' in label:
            pixel_handles.append(handle)
            pixel_labels.append(label)
        elif 'mean' in label:
            mean_handles.append(handle)
            mean_labels.append(label)
        elif 'predicted' in label:
            pred_handles.append(handle)
            pred_labels.append(label)
    
    # Reorder: pixels, mean, predicted
    ordered_handles = pixel_handles + mean_handles + pred_handles
    ordered_labels = pixel_labels + mean_labels + pred_labels
    
    ax.legend(ordered_handles, ordered_labels, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    return fig, ax


# classify_pixel_by_angle is now imported from figure_scatterplot_helpers


def subpanel_9(row_list, axes=None, data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619"):
    """
    Generate subpanel 9: Pixel-based accuracy across all rows.
    
    Creates 5 subplots:
    9.0: Percent correct vs pixel intensity
    9.1: Percent correct vs angle separation
    9.2: Percent correct vs separability score
    9.3: Scatterplot of actual angle vs predicted angle
    9.4: Scatterplot of actual variance vs predicted variance
    
    Parameters
    ----------
    row_list : list of dict
        List of row configuration dictionaries
    axes : array of matplotlib.axes.Axes, optional
        Axes to plot on (will create subplots if None)
    data_dir : str
        Path to data directory
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    # Create 5 subplots (2 rows, 3 columns)
    if axes is None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    else:
        if isinstance(axes, np.ndarray):
            fig = axes[0].figure
        else:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
    
    # Collect data from all rows
    all_row_data = []
    
    for row_idx, row_dict in enumerate(row_list):
        fluorophores = row_dict["Fluorophores"]
        ch1_config = row_dict["Channel 1"]
        ch2_config = row_dict["Channel 2"]
        
        # Load data for each fluorophore
        all_ch1_data = []
        all_ch2_data = []
        fp_labels = []
        
        for fp_name in fluorophores:
            ch1_data, _ = load_channel_data(data_dir, fp_name, 
                                        ch1_config["Excitation wavelength"],
                                        ch1_config["emission filter"],
                                        channel_num=1, subsample_factor=None)
            
            ch2_data, _ = load_channel_data(data_dir, fp_name,
                                        ch2_config["Excitation wavelength"],
                                        ch2_config["emission filter"],
                                        channel_num=1, subsample_factor=None)
            
            all_ch1_data.append(ch1_data)
            all_ch2_data.append(ch2_data)
            fp_labels.extend([fp_name] * len(ch1_data))
        
        # Combine all data
        ch1_combined = np.concatenate(all_ch1_data)
        ch2_combined = np.concatenate(all_ch2_data)
        fp_labels_array = np.array(fp_labels, dtype=object)
        
        # Get data vectors for classification
        data_vectors = {}
        for i, fp_name in enumerate(fluorophores):
            data_vec = compute_data_vector(all_ch1_data[i], all_ch2_data[i])
            data_vectors[fp_name] = data_vec
        
        # Get predicted vectors
        predicted_signals = compute_predicted_channel_signals(row_dict)
        predicted_vectors = {}
        for fp_name in fluorophores:
            ch1_signal = predicted_signals[fp_name]["Channel 1"]
            ch2_signal = predicted_signals[fp_name]["Channel 2"]
            pred_vec = np.array([ch1_signal, ch2_signal])
            pred_vec = pred_vec / np.linalg.norm(pred_vec)
            predicted_vectors[fp_name] = pred_vec
        
        # Store row data
        all_row_data.append({
            'row_dict': row_dict,
            'ch1': ch1_combined,
            'ch2': ch2_combined,
            'true_labels': fp_labels_array,
            'data_vectors': data_vectors,
            'predicted_vectors': predicted_vectors
        })
    
    # Subpanel 9.0: Percent correct vs pixel intensity
    ax0 = axes[0]
    _plot_9_0_percent_correct_vs_intensity(all_row_data, ax0)
    
    # Subpanel 9.1: Percent correct vs angle separation
    ax1 = axes[1]
    csv_data_9_1 = _plot_9_1_percent_correct_vs_angle_separation(all_row_data, ax1, data_dir)
    
    # Subpanel 9.2: Percent correct vs separability score (excludes tdTomato, avoids bidirectional)
    ax2 = axes[2]
    pairs_for_9_2, csv_data_9_2 = _plot_9_2_percent_correct_vs_separability(all_row_data, ax2, data_dir, csv_data_9_1)
    
    # Subpanel 9.3: Scatterplot of actual angle vs predicted angle (uses pairs from 9.2)
    ax3 = axes[3]
    _plot_9_3_actual_vs_predicted_angle(pairs_for_9_2, ax3, data_dir)
    
    # Subpanel 9.4: Scatterplot of actual variance vs predicted variance (uses pairs from 9.2)
    ax4 = axes[4]
    _plot_9_4_actual_vs_predicted_variance(pairs_for_9_2, ax4, data_dir)
    
    plt.tight_layout()
    
    return fig, axes


def _plot_9_0_percent_correct_vs_intensity(all_row_data, ax):
    """
    Plot subpanel 9.0: Percent correct vs pixel intensity.
    
    Parameters
    ----------
    all_row_data : list of dict
        List of row data dictionaries
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    # Intensity bins
    intensity_bins = np.logspace(1, 4, 50)  # 10 to 10000, log scale
    
    # Use colors and markers from config (matching 9.3 and 9.4)
    row_colors = cfg.row_colors
    row_markers = cfg.row_markers
    
    for row_idx, row_data in enumerate(all_row_data):
        ch1 = row_data['ch1']
        ch2 = row_data['ch2']
        true_labels = row_data['true_labels']
        data_vectors = row_data['data_vectors']
        
        # Compute pixel intensities (distance from origin)
        intensities = np.sqrt(ch1.astype(np.float64)**2 + ch2.astype(np.float64)**2)
        
        # Classify each pixel
        predicted_labels = []
        for i in range(len(ch1)):
            pred_label = classify_pixel_by_angle(ch1[i], ch2[i], data_vectors)
            predicted_labels.append(pred_label)
        predicted_labels = np.array(predicted_labels, dtype=object)
        
        # Filter out pixels at origin (intensity = 0)
        valid_mask = intensities > 0
        intensities_valid = intensities[valid_mask]
        true_labels_valid = true_labels[valid_mask]
        predicted_labels_valid = predicted_labels[valid_mask]
        
        # Compute percent correct for each intensity bin
        bin_centers = []
        percent_correct = []
        
        for i in range(len(intensity_bins) - 1):
            bin_min = intensity_bins[i]
            bin_max = intensity_bins[i + 1]
            bin_mask = (intensities_valid >= bin_min) & (intensities_valid < bin_max)
            
            if np.sum(bin_mask) > 0:
                bin_true = true_labels_valid[bin_mask]
                bin_pred = predicted_labels_valid[bin_mask]
                
                # Correct: predicted label matches true label (and not None)
                # None means pixel couldn't be classified (at origin)
                valid_pred_mask = bin_pred != None
                if np.sum(valid_pred_mask) > 0:
                    correct_mask = (bin_pred == bin_true) & valid_pred_mask
                    n_correct = np.sum(correct_mask)
                    n_total = np.sum(valid_pred_mask)  # Only count pixels that could be classified
                    
                    if n_total > 0:
                        pct_correct = 100.0 * n_correct / n_total
                        bin_centers.append((bin_min + bin_max) / 2.0)
                        percent_correct.append(pct_correct)
        
        # Plot line for this row with marker
        row_name = row_data['row_dict'].get('name', f'Row {row_idx + 1}')
        color = row_colors.get(row_name.lower(), '#1f77b4')  # Default blue if name not found
        marker = row_markers.get(row_name.lower(), 'o')  # Default circle if name not found
        ax.plot(bin_centers, percent_correct, label=row_name, color=color, linewidth=2, 
                marker=marker, markersize=6, markevery=max(1, len(bin_centers)//10))
    
    # Add 50% chance line (thicker than 9.1)
    ax.axhline(50, color='gray', linestyle='--', linewidth=3, alpha=0.5, label='Chance (50%)', zorder=1)
    
    ax.set_xlabel("Pixel Intensity", fontsize=12)
    ax.set_ylabel("Percent Correct", fontsize=12)
    ax.set_title("9.0: Percent Correct vs Pixel Intensity", fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(30, 7000)  # Start at 30 instead of 20
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def compute_predicted_signals_at_N(row_dict, N_fluorophores, ch1_pockels=None, ch2_pockels=None):
    """
    Compute predicted signal values at a given number of fluorophores in excitation volume.
    
    This scales the predicted signals from compute_predicted_channel_signals to represent
    the expected photon counts for N fluorophores.
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
    N_fluorophores : float
        Number of fluorophores in excitation volume (e.g., 10000)
    ch1_pockels : int, optional
        Pockels value for channel 1
    ch2_pockels : int, optional
        Pockels value for channel 2
        
    Returns
    -------
    dict
        Dictionary with structure: {fp_name: {channel_name: mu_i}}
        where mu_i is the expected detected photons in channel i
    """
    # Get base predicted signals (relative, not absolute)
    base_signals = compute_predicted_channel_signals(row_dict, ch1_pockels=ch1_pockels, ch2_pockels=ch2_pockels)
    
    # Scale to N fluorophores
    # The base signals are proportional to excitation * emission * power
    # We need to scale by N to get expected photon counts
    scaled_signals = {}
    for fp_name in base_signals:
        scaled_signals[fp_name] = {}
        for channel_name in base_signals[fp_name]:
            # Scale by N: each fluorophore contributes proportionally
            scaled_signals[fp_name][channel_name] = base_signals[fp_name][channel_name] * N_fluorophores
    
    return scaled_signals


def compute_perpendicular_variance_95_interval(vector, mu_values):
    """
    Compute 95% perpendicular noise interval for a given vector and expected photon counts.
    
    Based on the formula:
    - sigma_perp_sq = sum over i of (u_i)^2 * mu_i
    - delta_95_perp = 1.645 * sqrt(sigma_perp_sq)
    
    Parameters
    ----------
    vector : np.ndarray
        Unit vector [ch1_component, ch2_component] (normalized)
    mu_values : np.ndarray
        Expected photon counts [mu_1, mu_2] for each channel
        
    Returns
    -------
    float
        95% perpendicular noise interval (delta_95_perp)
    """
    # Ensure vector is normalized
    vector_norm = vector / np.linalg.norm(vector)
    
    # Compute sigma_perp_sq = sum_i (u_i)^2 * mu_i
    sigma_perp_sq = np.sum((vector_norm ** 2) * mu_values)
    
    # Compute 95% interval: 1.645 * sqrt(sigma_perp_sq)
    delta_95_perp = 1.645 * np.sqrt(sigma_perp_sq)
    
    return delta_95_perp


def compute_separability_score(row_dict, fp1, fp2, data_vectors, N_fluorophores=10000, 
                               ch1_pockels=None, ch2_pockels=None):
    """
    Compute separability score between two fluorophores.
    
    SS = 2 * distance_between_points / (sum of 95% perpendicular confidence intervals at both points)
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
    fp1 : str
        Name of first fluorophore
    fp2 : str
        Name of second fluorophore
    data_vectors : dict
        Dictionary mapping fp_name -> unit vector from data
    N_fluorophores : float
        Number of fluorophores in excitation volume (default 10000)
    ch1_pockels : int, optional
        Pockels value for channel 1
    ch2_pockels : int, optional
        Pockels value for channel 2
        
    Returns
    -------
    float
        Separability score, or None if computation fails
    """
    # Get predicted signals at N fluorophores
    predicted_signals = compute_predicted_signals_at_N(row_dict, N_fluorophores, 
                                                       ch1_pockels=ch1_pockels, ch2_pockels=ch2_pockels)
    
    if fp1 not in predicted_signals or fp2 not in predicted_signals:
        return None
    
    # Get predicted vectors (normalized)
    vec1_pred = np.array([predicted_signals[fp1]["Channel 1"], predicted_signals[fp1]["Channel 2"]])
    vec2_pred = np.array([predicted_signals[fp2]["Channel 1"], predicted_signals[fp2]["Channel 2"]])
    
    # Normalize to unit vectors
    vec1_pred = vec1_pred / np.linalg.norm(vec1_pred)
    vec2_pred = vec2_pred / np.linalg.norm(vec2_pred)
    
    # Compute distances from origin for each FP's predicted point
    # The point is at N fluorophores along the predicted vector
    # We need to scale the vector to represent N fluorophores
    # The magnitude should be such that the total photons = N * (excitation * emission * power)
    # For simplicity, we'll use the magnitude of the predicted signal vector
    point1 = vec1_pred * np.linalg.norm([predicted_signals[fp1]["Channel 1"], predicted_signals[fp1]["Channel 2"]])
    point2 = vec2_pred * np.linalg.norm([predicted_signals[fp2]["Channel 1"], predicted_signals[fp2]["Channel 2"]])
    
    # Find which FP is dimmer (closer to origin, smaller L2 norm)
    dist1 = np.linalg.norm(point1)
    dist2 = np.linalg.norm(point2)
    
    if dist1 <= dist2:
        # fp1 is dimmer - use its point
        dimmer_point = point1
        dimmer_vec = vec1_pred
        dimmer_mu = np.array([predicted_signals[fp1]["Channel 1"], predicted_signals[fp1]["Channel 2"]])
        other_vec = vec2_pred
    else:
        # fp2 is dimmer - use its point
        dimmer_point = point2
        dimmer_vec = vec2_pred
        dimmer_mu = np.array([predicted_signals[fp2]["Channel 1"], predicted_signals[fp2]["Channel 2"]])
        other_vec = vec1_pred
    
    # Find nearest point on other FP's vector
    # Project dimmer_point onto other_vec
    # The projection is: proj = dot(dimmer_point, other_vec) * other_vec
    proj_scalar = np.dot(dimmer_point, other_vec)
    nearest_point = proj_scalar * other_vec
    
    # Compute distance between dimmer_point and nearest_point
    distance = np.linalg.norm(dimmer_point - nearest_point)
    
    # Compute 95% perpendicular confidence intervals at both points
    # At dimmer_point
    delta_95_dimmer = compute_perpendicular_variance_95_interval(dimmer_vec, dimmer_mu)
    
    # At nearest_point (need mu values for other FP)
    if dist1 <= dist2:
        other_mu = np.array([predicted_signals[fp2]["Channel 1"], predicted_signals[fp2]["Channel 2"]])
    else:
        other_mu = np.array([predicted_signals[fp1]["Channel 1"], predicted_signals[fp1]["Channel 2"]])
    delta_95_nearest = compute_perpendicular_variance_95_interval(other_vec, other_mu)
    
    # Separability score
    SS = (2.0 * distance) / (delta_95_dimmer + delta_95_nearest)
    
    return SS


# compute_actual_variance_perpendicular is now imported from figure_scatterplot_helpers


def _get_all_acquisition_pairs(data_dir, exclude_fluorophores=None, avoid_bidirectional=True,
                               excitation_wls=None, filters=None):
    """
    Get all acquisition pairs from the data directory.
    
    Parameters
    ----------
    data_dir : str
        Path to data directory
    exclude_fluorophores : list of str, optional
        List of fluorophore names to exclude (e.g., ['tdTomato'])
    avoid_bidirectional : bool
        If True, skip bidirectional pairs (only process one direction)
    excitation_wls : list of int, optional
        List of excitation wavelengths to search for. Default [1080, 1240]
    filters : list of str, optional
        List of filter names to search for. Default ['BR2', 'Red', 'FarRed']
        
    Returns
    -------
    list of dict
        List of pair dictionaries with keys: fp1, fp2, ch1_wl, ch1_filter, ch2_wl, ch2_filter,
        fp1_ch1, fp1_ch2, fp2_ch1, fp2_ch2
    """
    import glob
    
    if exclude_fluorophores is None:
        exclude_fluorophores = []
    
    # Default excitation wavelengths and filters (for Figure 1 compatibility)
    if excitation_wls is None:
        excitation_wls = [1080, 1240]
    if filters is None:
        filters = ['BR2', 'Red', 'FarRed']
    
    # Find all fluorophore folders
    abs_data_dir = os.path.abspath(data_dir) if not os.path.isabs(data_dir) else data_dir
    if not os.path.isdir(abs_data_dir):
        print(f"Warning: Data directory not found: {abs_data_dir}")
        return []
    
    # Get all directories that might be fluorophore folders
    all_dirs = [d for d in os.listdir(abs_data_dir) 
                if os.path.isdir(os.path.join(abs_data_dir, d))]
    
    # Identify fluorophore names (remove _mouse suffix if present)
    fluorophore_names = set()
    for d in all_dirs:
        if d.endswith('_mouse'):
            fp_name = d[:-6]  # Remove '_mouse'
        else:
            fp_name = d
        # Check if excluded (case-insensitive partial match)
        is_excluded = False
        for excl in exclude_fluorophores:
            if excl.lower() in fp_name.lower():
                is_excluded = True
                break
        if not is_excluded:
            fluorophore_names.add(fp_name)
    
    fluorophore_names = sorted(list(fluorophore_names))
    
    # Collect all acquisition pairs
    all_pairs = []
    seen_channel_configs = set()  # Track channel configurations we've already added (to avoid bidirectional duplicates)
    
    # For each pair of fluorophores
    for fp1 in fluorophore_names:
        for fp2 in fluorophore_names:
            if fp1 >= fp2:  # Avoid duplicates (fp1, fp2) and (fp2, fp1)
                continue
            
            # For each combination of ch1 and ch2 configurations
            for ch1_wl in excitation_wls:
                for ch1_filter in filters:
                    for ch2_wl in excitation_wls:
                        for ch2_filter in filters:
                            # Skip identity pairs (same channel config for both)
                            if ch1_wl == ch2_wl and ch1_filter == ch2_filter:
                                continue  # Angle will be zero, all points on diagonal
                            
                            # Create channel configuration signature for bidirectional check
                            # Only consider channel configs, not fluorophores
                            channel_config = (ch1_wl, ch1_filter, ch2_wl, ch2_filter)
                            flipped_config = (ch2_wl, ch2_filter, ch1_wl, ch1_filter)
                            
                            # Avoid bidirectional: check if flipped channel config is already in the list
                            if avoid_bidirectional:
                                if flipped_config in seen_channel_configs:
                                    continue  # Skip if flipped channel config already exists
                            
                            # Try to load data for this pair
                            try:
                                fp1_ch1, ch1_poc1 = load_channel_data(data_dir, fp1, ch1_wl, ch1_filter, channel_num=1)
                                fp1_ch2, ch2_poc1 = load_channel_data(data_dir, fp1, ch2_wl, ch2_filter, channel_num=1)
                                fp2_ch1, ch1_poc2 = load_channel_data(data_dir, fp2, ch1_wl, ch1_filter, channel_num=1)
                                fp2_ch2, ch2_poc2 = load_channel_data(data_dir, fp2, ch2_wl, ch2_filter, channel_num=1)
                                
                                # Use Pockels from first FP for each channel (they should match for same channel config)
                                ch1_pockels = ch1_poc1 if ch1_poc1 is not None else ch1_poc2
                                ch2_pockels = ch2_poc1 if ch2_poc1 is not None else ch2_poc2
                                
                                # Check if all data exists and has same length
                                if (len(fp1_ch1) > 0 and len(fp1_ch2) > 0 and 
                                    len(fp2_ch1) > 0 and len(fp2_ch2) > 0 and
                                    len(fp1_ch1) == len(fp1_ch2) and 
                                    len(fp2_ch1) == len(fp2_ch2)):
                                    # Add pair and mark channel config as seen
                                    all_pairs.append({
                                        'fp1': fp1,
                                        'fp2': fp2,
                                        'ch1_wl': ch1_wl,
                                        'ch1_filter': ch1_filter,
                                        'ch2_wl': ch2_wl,
                                        'ch2_filter': ch2_filter,
                                        'ch1_pockels': ch1_pockels,
                                        'ch2_pockels': ch2_pockels,
                                        'fp1_ch1': fp1_ch1,
                                        'fp1_ch2': fp1_ch2,
                                        'fp2_ch1': fp2_ch1,
                                        'fp2_ch2': fp2_ch2
                                    })
                                    seen_channel_configs.add(channel_config)
                            except (ValueError, FileNotFoundError):
                                # Skip if data not found
                                continue
    
    return all_pairs


def _plot_9_2_percent_correct_vs_separability(all_row_data, ax, data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619", csv_data_9_1=None):
    """
    Plot subpanel 9.2: Percent correct vs separability score.
    
    Excludes tdTomato, avoids bidirectional pairs, includes FP names in CSV.
    Highlights known rows with same style as 9.1.
    
    Parameters
    ----------
    all_row_data : list of dict
        List of row data dictionaries (for the 3 known rows)
    ax : matplotlib.axes.Axes
        Axes to plot on
    data_dir : str
        Path to data directory
    csv_data_9_1 : list of dict, optional
        CSV data from 9.1 to merge with
        
    Returns
    -------
    list of dict
        List of processed pairs (for use in 9.3)
    list of dict
        CSV data for this subpanel
    """
    # Use colors and markers from config (matching 9.3 and 9.4)
    row_colors = cfg.row_colors
    row_markers = cfg.row_markers
    
    # Get the 3 known row configurations for highlighting
    known_row_configs = []
    for row_data in all_row_data:
        row_dict = row_data['row_dict']
        ch1_config = row_dict['Channel 1']
        ch2_config = row_dict['Channel 2']
        fluorophores = row_dict['Fluorophores']
        known_row_configs.append({
            'fp1': fluorophores[0],
            'fp2': fluorophores[1],
            'ch1_wl': ch1_config['Excitation wavelength'],
            'ch1_filter': ch1_config['emission filter'],
            'ch2_wl': ch2_config['Excitation wavelength'],
            'ch2_filter': ch2_config['emission filter'],
            'name': row_dict.get('name', '')
        })
    """
    Plot subpanel 9.2: Percent correct vs separability score.
    
    Excludes tdTomato, avoids bidirectional pairs, includes FP names in CSV.
    
    Parameters
    ----------
    all_row_data : list of dict
        List of row data dictionaries (for the 3 known rows)
    ax : matplotlib.axes.Axes
        Axes to plot on
    data_dir : str
        Path to data directory
        
    Returns
    -------
    list of dict
        List of processed pairs (for use in 9.3)
    """
    # Get all acquisition pairs (exclude tdTomato, avoid bidirectional)
    print("Subpanel 9.2: Finding all acquisition pairs (excluding tdTomato)...")
    exclude_fps = ['tdTomato', 'TdTomato', 'TDTomato', 'TDTfp']
    all_pairs = _get_all_acquisition_pairs(data_dir, exclude_fluorophores=exclude_fps, avoid_bidirectional=True,
                                          excitation_wls=[1080, 1240], filters=['BR2', 'Red', 'FarRed'])
    print(f"  Found {len(all_pairs)} valid acquisition pairs")
    
    # Process each pair
    print(f"Processing {len(all_pairs)} pairs (computing separability, classifying)...")
    separability_scores = []
    percent_corrects = []
    csv_data = []
    processed_pairs = []  # Store for 9.3
    is_known_list = []
    row_names_list = []
    
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
        
        # Combine data from both fluorophores
        ch1_combined = np.concatenate([fp1_ch1, fp2_ch1])
        ch2_combined = np.concatenate([fp1_ch2, fp2_ch2])
        true_labels = np.array([fp1] * len(fp1_ch1) + [fp2] * len(fp2_ch1), dtype=object)
        
        # Compute data vectors
        data_vectors = {}
        try:
            data_vectors[fp1] = compute_data_vector(fp1_ch1, fp1_ch2)
            data_vectors[fp2] = compute_data_vector(fp2_ch1, fp2_ch2)
        except:
            continue  # Skip if can't compute vectors
        
        if len(data_vectors) != 2:
            continue
        
        # Compute separability score
        row_dict_like = {
            'Fluorophores': [fp1, fp2],
            'Channel 1': {'Excitation wavelength': ch1_wl, 'emission filter': ch1_filter},
            'Channel 2': {'Excitation wavelength': ch2_wl, 'emission filter': ch2_filter}
        }
        ch1_pockels = pair.get('ch1_pockels')
        ch2_pockels = pair.get('ch2_pockels')
        separability_score = compute_separability_score(
            row_dict_like, fp1, fp2, data_vectors,
            N_fluorophores=cfg.N_fluorophores_default,
            ch1_pockels=ch1_pockels, ch2_pockels=ch2_pockels
        )
        if separability_score is None:
            separability_score = 0.0  # Fallback if computation fails
        
        # Subsample and classify (same as 9.1)
        intensities = np.sqrt(ch1_combined.astype(np.float64)**2 + ch2_combined.astype(np.float64)**2)
        mean_intensity = np.mean(intensities[intensities > 0])
        if mean_intensity < 100:
            continue
        
        # Subsample pixels (50 per bin)
        intensity_bins = np.logspace(1, 4, 81)
        ch1_sampled = []
        ch2_sampled = []
        labels_sampled = []
        
        for i in range(len(intensity_bins) - 1):
            bin_min = intensity_bins[i]
            bin_max = intensity_bins[i + 1]
            bin_mask = (intensities >= bin_min) & (intensities < bin_max)
            
            if np.sum(bin_mask) > 0:
                bin_ch1 = ch1_combined[bin_mask]
                bin_ch2 = ch2_combined[bin_mask]
                bin_labels = true_labels[bin_mask]
                
                n_sample = min(50, len(bin_ch1))
                if n_sample > 0:
                    indices = np.random.choice(len(bin_ch1), n_sample, replace=False)
                    ch1_sampled.append(bin_ch1[indices])
                    ch2_sampled.append(bin_ch2[indices])
                    labels_sampled.append(bin_labels[indices])
        
        if len(ch1_sampled) == 0:
            continue
        
        ch1_sampled = np.concatenate(ch1_sampled)
        ch2_sampled = np.concatenate(ch2_sampled)
        labels_sampled = np.concatenate(labels_sampled)
        
        # Classify sampled pixels
        predicted_labels = []
        for i in range(len(ch1_sampled)):
            pred_label = classify_pixel_by_angle(ch1_sampled[i], ch2_sampled[i], data_vectors)
            predicted_labels.append(pred_label)
        predicted_labels = np.array(predicted_labels, dtype=object)
        
        # Compute percent correct
        valid_pred_mask = predicted_labels != None
        if np.sum(valid_pred_mask) > 0:
            correct_mask = (predicted_labels == labels_sampled) & valid_pred_mask
            n_correct = np.sum(correct_mask)
            n_total = np.sum(valid_pred_mask)
            pct_correct = 100.0 * n_correct / n_total if n_total > 0 else 0
        else:
            pct_correct = 0
        
        # Check if this is one of the known rows
        is_known = False
        row_name = None
        for known_config in known_row_configs:
            if (known_config['fp1'] == fp1 and known_config['fp2'] == fp2 and
                known_config['ch1_wl'] == ch1_wl and 
                known_config['ch1_filter'] == ch1_filter and
                known_config['ch2_wl'] == ch2_wl and
                known_config['ch2_filter'] == ch2_filter):
                is_known = True
                row_name = known_config['name']
                break
        
        separability_scores.append(separability_score)
        percent_corrects.append(pct_correct)
        is_known_list.append(is_known)
        row_names_list.append(row_name)
        
        # Compute angles for each FP (to nearest axis) - same as 9.1
        # mCherry angle actual
        mcherry_vec_data = data_vectors.get('mCherry', data_vectors.get(fp1 if 'Cherry' in fp1 else fp2))
        if mcherry_vec_data is not None:
            mcherry_angle_rad = np.arctan2(mcherry_vec_data[1], mcherry_vec_data[0])
            mcherry_angle_actual = np.degrees(mcherry_angle_rad)
            mcherry_angle_actual = np.abs(mcherry_angle_actual) % 180
            mcherry_angle_actual = min(mcherry_angle_actual, 180 - mcherry_angle_actual)
        else:
            mcherry_angle_actual = None
        
        # mNeptune angle actual
        mneptune_vec_data = data_vectors.get('mNeptune', data_vectors.get(fp1 if 'Neptune' in fp1 else fp2))
        if mneptune_vec_data is not None:
            mneptune_angle_rad = np.arctan2(mneptune_vec_data[1], mneptune_vec_data[0])
            mneptune_angle_actual = np.degrees(mneptune_angle_rad)
            mneptune_angle_actual = np.abs(mneptune_angle_actual) % 180
            mneptune_angle_actual = min(mneptune_angle_actual, 180 - mneptune_angle_actual)
        else:
            mneptune_angle_actual = None
        
        # Get predicted vectors
        row_dict_like = {
            'Fluorophores': [fp1, fp2],
            'Channel 1': {'Excitation wavelength': ch1_wl, 'emission filter': ch1_filter},
            'Channel 2': {'Excitation wavelength': ch2_wl, 'emission filter': ch2_filter}
        }
        predicted_signals = compute_predicted_channel_signals(row_dict_like)
        predicted_vectors = {}
        for fp_name in [fp1, fp2]:
            ch1_signal = predicted_signals[fp_name]["Channel 1"]
            ch2_signal = predicted_signals[fp_name]["Channel 2"]
            pred_vec = np.array([ch1_signal, ch2_signal])
            pred_vec = pred_vec / np.linalg.norm(pred_vec)
            predicted_vectors[fp_name] = pred_vec
        
        # mCherry angle predicted
        mcherry_vec_pred = predicted_vectors.get('mCherry', predicted_vectors.get(fp1 if 'Cherry' in fp1 else fp2))
        if mcherry_vec_pred is not None:
            mcherry_angle_rad_pred = np.arctan2(mcherry_vec_pred[1], mcherry_vec_pred[0])
            mcherry_angle_predicted = np.degrees(mcherry_angle_rad_pred)
            mcherry_angle_predicted = np.abs(mcherry_angle_predicted) % 180
            mcherry_angle_predicted = min(mcherry_angle_predicted, 180 - mcherry_angle_predicted)
        else:
            mcherry_angle_predicted = None
        
        # mNeptune angle predicted
        mneptune_vec_pred = predicted_vectors.get('mNeptune', predicted_vectors.get(fp1 if 'Neptune' in fp1 else fp2))
        if mneptune_vec_pred is not None:
            mneptune_angle_rad_pred = np.arctan2(mneptune_vec_pred[1], mneptune_vec_pred[0])
            mneptune_angle_predicted = np.degrees(mneptune_angle_rad_pred)
            mneptune_angle_predicted = np.abs(mneptune_angle_predicted) % 180
            mneptune_angle_predicted = min(mneptune_angle_predicted, 180 - mneptune_angle_predicted)
        else:
            mneptune_angle_predicted = None
        
        # Compute angle between vectors
        vec1 = data_vectors[fp1]
        vec2 = data_vectors[fp2]
        dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_between = np.degrees(angle_rad)
        
        # Store for CSV (include FP names and all angles)
        # Format acquisition names with Pockels if available
        ch1_pockels = pair.get('ch1_pockels')
        ch2_pockels = pair.get('ch2_pockels')
        
        # Get power values for suffix
        power_mapping = load_pockels_power_mapping()
        ch1_power = get_power_from_pockels(ch1_wl, ch1_pockels, power_mapping) if ch1_pockels is not None else None
        ch2_power = get_power_from_pockels(ch2_wl, ch2_pockels, power_mapping) if ch2_pockels is not None else None
        
        if ch1_pockels is not None:
            power_suffix_1 = f"_{int(ch1_power)}mW" if ch1_power is not None else ""
            acq1_name_csv = f"{ch1_wl}nm_{ch1_filter}_{ch1_pockels}poc{power_suffix_1}"
        else:
            acq1_name_csv = f"{ch1_wl}nm_{ch1_filter}"
        if ch2_pockels is not None:
            power_suffix_2 = f"_{int(ch2_power)}mW" if ch2_power is not None else ""
            acq2_name_csv = f"{ch2_wl}nm_{ch2_filter}_{ch2_pockels}poc{power_suffix_2}"
        else:
            acq2_name_csv = f"{ch2_wl}nm_{ch2_filter}"
        
        # Calculate mNeptune error (signed difference: actual - predicted)
        # Positive = predicted is smaller (underestimated), Negative = predicted is larger (overestimated)
        mneptune_error = None
        if mneptune_angle_actual is not None and mneptune_angle_predicted is not None:
            mneptune_error = mneptune_angle_actual - mneptune_angle_predicted
        
        csv_data.append({
            'fp1': fp1,
            'fp2': fp2,
            'mNeptune_error': mneptune_error,
            'acquisition_1': acq1_name_csv,
            'acquisition_2': acq2_name_csv,
            'angle_between_vectors': angle_between,
            'mCherry_angle_actual': mcherry_angle_actual,
            'mCherry_angle_predicted': mcherry_angle_predicted,
            'mNeptune_angle_actual': mneptune_angle_actual,
            'mNeptune_angle_predicted': mneptune_angle_predicted,
            'separability_score': separability_score,
            'percent_correct': pct_correct
        })
        
        # Store processed pair for 9.3 and 9.4
        processed_pairs.append({
            'fp1': fp1,
            'fp2': fp2,
            'ch1_wl': ch1_wl,
            'ch1_filter': ch1_filter,
            'ch2_wl': ch2_wl,
            'ch2_filter': ch2_filter,
            'ch1_pockels': ch1_pockels,
            'ch2_pockels': ch2_pockels,
            'data_vectors': data_vectors,
            'fp1_ch1': fp1_ch1,
            'fp1_ch2': fp1_ch2,
            'fp2_ch1': fp2_ch1,
            'fp2_ch2': fp2_ch2,
            'is_known': is_known,
            'row_name': row_name,
            'predicted_vectors': {}  # Will be computed in 9.3
        })
    
    # Combine with 9.1 CSV data and save
    import pandas as pd
    csv_data_combined = []
    
    # Get 9.1 data if available
    if csv_data_9_1 is not None and len(csv_data_9_1) > 0:
        # Create a lookup key for 9.1 data
        lookup_9_1 = {}
        for row in csv_data_9_1:
            key = (row['fp1'], row['fp2'], row['acquisition_1'], row['acquisition_2'])
            lookup_9_1[key] = row
        
        # Merge 9.2 data with 9.1 data
        keys_9_2 = set()
        for row_9_2 in csv_data:
            key = (row_9_2['fp1'], row_9_2['fp2'], row_9_2['acquisition_1'], row_9_2['acquisition_2'])
            keys_9_2.add(key)
            if key in lookup_9_1:
                # Combine: use all data from 9.2 (it has all the angles), but keep separability from 9.2
                # 9.2 already has all the angle columns, so just use it
                csv_data_combined.append(row_9_2)
            else:
                # Only in 9.2 (shouldn't happen if exclusion is correct)
                csv_data_combined.append(row_9_2)
        
        # Add rows only in 9.1 (shouldn't happen since both exclude tdTomato now)
        for row_9_1 in csv_data_9_1:
            key = (row_9_1['fp1'], row_9_1['fp2'], row_9_1['acquisition_1'], row_9_1['acquisition_2'])
            if key not in keys_9_2:
                # Add with None separability_score
                combined_row = row_9_1.copy()
                combined_row['separability_score'] = None
                csv_data_combined.append(combined_row)
    else:
        # Only 9.2 data available
        csv_data_combined = csv_data
    
    # Save combined CSV
    if len(csv_data_combined) > 0:
        csv_df = pd.DataFrame(csv_data_combined)
        # Reorder columns: fp1, fp2, mNeptune_error, acquisition_1, acquisition_2, then rest
        column_order = ['fp1', 'fp2', 'mNeptune_error', 'acquisition_1', 'acquisition_2']
        remaining_cols = [c for c in csv_df.columns if c not in column_order]
        csv_df = csv_df[column_order + remaining_cols]
        csv_path = os.path.join("results", "Figure1", "subpanel9_results.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        csv_df.to_csv(csv_path, index=False)
        print(f"  Saved combined results to {csv_path}")
    
    # Plot with highlighting for known rows (same style as 9.1)
    for i, (ss, pct, is_known, name) in enumerate(zip(
        separability_scores, percent_corrects, is_known_list, row_names_list)):
        
        if is_known and name:
            # Highlight known rows with blue shades and markers
            color = row_colors.get(name.lower(), '#1f77b4')  # Default blue if name not found
            marker = row_markers.get(name.lower(), 'o')  # Default circle if name not found
            ax.scatter(ss, pct, color=color, marker=marker, s=150, label=name, zorder=3, 
                      edgecolors='black', linewidths=1.5)
        else:
            # Unlabeled dots for other pairs
            ax.scatter(ss, pct, color='gray', s=30, alpha=0.5, zorder=1)
    
    ax.set_xlabel("Separability Score", fontsize=12)
    ax.set_ylabel("Percent Correct", fontsize=12)
    ax.set_title("9.2: Percent Correct vs Separability Score", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return processed_pairs, csv_data_combined
    
    ax.set_xlabel("Separability Score", fontsize=12)
    ax.set_ylabel("Percent Correct", fontsize=12)
    ax.set_title("9.2: Percent Correct vs Separability Score", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    return processed_pairs


def _plot_9_1_percent_correct_vs_angle_separation(all_row_data, ax, data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619"):
    """
    Returns csv_data for combination with 9.2
    """
    """
    Plot subpanel 9.1: Percent correct vs angle between mean vectors.
    
    Processes ALL acquisition pairs in the data directory, not just the 3 rows.
    Highlights and labels the 3 known rows.
    
    Parameters
    ----------
    all_row_data : list of dict
        List of row data dictionaries (for the 3 known rows)
    ax : matplotlib.axes.Axes
        Axes to plot on
    data_dir : str
        Path to data directory
    """
    import glob
    
    # Use colors and markers from config (matching 9.3 and 9.4)
    row_colors = cfg.row_colors
    row_markers = cfg.row_markers
    
    # Get the 3 known row configurations for highlighting
    known_row_configs = []
    for row_data in all_row_data:
        row_dict = row_data['row_dict']
        ch1_config = row_dict['Channel 1']
        ch2_config = row_dict['Channel 2']
        fluorophores = row_dict['Fluorophores']
        known_row_configs.append({
            'fp1': fluorophores[0],
            'fp2': fluorophores[1],
            'ch1_wl': ch1_config['Excitation wavelength'],
            'ch1_filter': ch1_config['emission filter'],
            'ch2_wl': ch2_config['Excitation wavelength'],
            'ch2_filter': ch2_config['emission filter'],
            'name': row_dict.get('name', '')
        })
    
    # Get all acquisition pairs (exclude tdTomato for 9.1, avoid bidirectional)
    print("Subpanel 9.1: Finding all acquisition pairs (excluding tdTomato)...")
    exclude_fps = ['tdTomato', 'TdTomato', 'TDTomato', 'TDTfp']
    all_pairs = _get_all_acquisition_pairs(data_dir, exclude_fluorophores=exclude_fps, avoid_bidirectional=True,
                                          excitation_wls=[1080, 1240], filters=['BR2', 'Red', 'FarRed'])
    print(f"  Found {len(all_pairs)} valid acquisition pairs")
    
    # Process each pair
    print(f"Processing {len(all_pairs)} pairs (computing vectors, subsampling, classifying)...")
    angles_between_vectors = []
    percent_corrects = []
    is_known_row = []
    row_names = []
    mean_intensities = []
    csv_data = []  # Store data for CSV output
    
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
        
        # Combine data from both fluorophores
        # For each FP, ch1 comes from ch1_config, ch2 comes from ch2_config
        ch1_combined = np.concatenate([fp1_ch1, fp2_ch1])
        ch2_combined = np.concatenate([fp1_ch2, fp2_ch2])
        true_labels = np.array([fp1] * len(fp1_ch1) + [fp2] * len(fp2_ch1), dtype=object)
        
        # Compute data vectors (each FP's vector from its own ch1 and ch2 data)
        data_vectors = {}
        try:
            data_vectors[fp1] = compute_data_vector(fp1_ch1, fp1_ch2)
            data_vectors[fp2] = compute_data_vector(fp2_ch1, fp2_ch2)
        except:
            continue  # Skip if can't compute vectors
        
        if len(data_vectors) != 2:
            continue
        
        vec1 = data_vectors[fp1]
        vec2 = data_vectors[fp2]
        
        # Compute angle between vectors
        dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        # Check mean intensity - skip if too low (angle cannot be estimated well)
        intensities = np.sqrt(ch1_combined.astype(np.float64)**2 + ch2_combined.astype(np.float64)**2)
        mean_intensity = np.mean(intensities[intensities > 0])
        if mean_intensity < 100:  # Threshold for low brightness
            continue
        
        # Subsample pixels for speed (50 per bin, ~4000 total)
        # Bin by intensity and sample 50 from each bin
        intensity_bins = np.logspace(1, 4, 81)  # 80 bins
        ch1_sampled = []
        ch2_sampled = []
        labels_sampled = []
        
        for i in range(len(intensity_bins) - 1):
            bin_min = intensity_bins[i]
            bin_max = intensity_bins[i + 1]
            bin_mask = (intensities >= bin_min) & (intensities < bin_max)
            
            if np.sum(bin_mask) > 0:
                bin_ch1 = ch1_combined[bin_mask]
                bin_ch2 = ch2_combined[bin_mask]
                bin_labels = true_labels[bin_mask]
                
                n_sample = min(50, len(bin_ch1))
                if n_sample > 0:
                    indices = np.random.choice(len(bin_ch1), n_sample, replace=False)
                    ch1_sampled.append(bin_ch1[indices])
                    ch2_sampled.append(bin_ch2[indices])
                    labels_sampled.append(bin_labels[indices])
        
        if len(ch1_sampled) == 0:
            continue
        
        ch1_sampled = np.concatenate(ch1_sampled)
        ch2_sampled = np.concatenate(ch2_sampled)
        labels_sampled = np.concatenate(labels_sampled)
        
        # Classify sampled pixels
        predicted_labels = []
        for i in range(len(ch1_sampled)):
            pred_label = classify_pixel_by_angle(ch1_sampled[i], ch2_sampled[i], data_vectors)
            predicted_labels.append(pred_label)
        predicted_labels = np.array(predicted_labels, dtype=object)
        
        # Compute percent correct
        valid_pred_mask = predicted_labels != None
        if np.sum(valid_pred_mask) > 0:
            correct_mask = (predicted_labels == labels_sampled) & valid_pred_mask
            n_correct = np.sum(correct_mask)
            n_total = np.sum(valid_pred_mask)
            pct_correct = 100.0 * n_correct / n_total if n_total > 0 else 0
        else:
            pct_correct = 0
        
        # Check if this is one of the known rows
        is_known = False
        row_name = None
        for known_config in known_row_configs:
            if (known_config['fp1'] == fp1 and known_config['fp2'] == fp2 and
                known_config['ch1_wl'] == pair['ch1_wl'] and 
                known_config['ch1_filter'] == pair['ch1_filter'] and
                known_config['ch2_wl'] == pair['ch2_wl'] and
                known_config['ch2_filter'] == pair['ch2_filter']):
                is_known = True
                row_name = known_config['name']
                break
        
        # Format acquisition names (channel configurations, not fluorophores)
        # Include Pockels values if available
        ch1_pockels = pair.get('ch1_pockels')
        ch2_pockels = pair.get('ch2_pockels')
        
        # Get power values for suffix
        power_mapping = load_pockels_power_mapping()
        ch1_power = get_power_from_pockels(ch1_wl, ch1_pockels, power_mapping) if ch1_pockels is not None else None
        ch2_power = get_power_from_pockels(ch2_wl, ch2_pockels, power_mapping) if ch2_pockels is not None else None
        
        if ch1_pockels is not None:
            power_suffix_1 = f"_{int(ch1_power)}mW" if ch1_power is not None else ""
            acq1_name = f"{ch1_wl}nm_{ch1_filter}_{ch1_pockels}poc{power_suffix_1}"
        else:
            acq1_name = f"{ch1_wl}nm_{ch1_filter}"
        if ch2_pockels is not None:
            power_suffix_2 = f"_{int(ch2_power)}mW" if ch2_power is not None else ""
            acq2_name = f"{ch2_wl}nm_{ch2_filter}_{ch2_pockels}poc{power_suffix_2}"
        else:
            acq2_name = f"{ch2_wl}nm_{ch2_filter}"
        
        # Compute angles for each FP (to nearest axis)
        # mCherry angle actual
        mcherry_vec_data = data_vectors.get('mCherry', data_vectors.get(fp1 if 'Cherry' in fp1 else fp2))
        if mcherry_vec_data is not None:
            mcherry_angle_rad = np.arctan2(mcherry_vec_data[1], mcherry_vec_data[0])
            mcherry_angle_actual = np.degrees(mcherry_angle_rad)
            mcherry_angle_actual = np.abs(mcherry_angle_actual) % 180
            mcherry_angle_actual = min(mcherry_angle_actual, 180 - mcherry_angle_actual)
        else:
            mcherry_angle_actual = None
        
        # mNeptune angle actual
        mneptune_vec_data = data_vectors.get('mNeptune', data_vectors.get(fp1 if 'Neptune' in fp1 else fp2))
        if mneptune_vec_data is not None:
            mneptune_angle_rad = np.arctan2(mneptune_vec_data[1], mneptune_vec_data[0])
            mneptune_angle_actual = np.degrees(mneptune_angle_rad)
            mneptune_angle_actual = np.abs(mneptune_angle_actual) % 180
            mneptune_angle_actual = min(mneptune_angle_actual, 180 - mneptune_angle_actual)
        else:
            mneptune_angle_actual = None
        
        # Get predicted vectors for angle calculations
        row_dict_like = {
            'Fluorophores': [fp1, fp2],
            'Channel 1': {'Excitation wavelength': ch1_wl, 'emission filter': ch1_filter},
            'Channel 2': {'Excitation wavelength': ch2_wl, 'emission filter': ch2_filter}
        }
        ch1_pockels = pair.get('ch1_pockels')
        ch2_pockels = pair.get('ch2_pockels')
        predicted_signals = compute_predicted_channel_signals(row_dict_like, ch1_pockels=ch1_pockels, ch2_pockels=ch2_pockels)
        predicted_vectors = {}
        for fp_name in [fp1, fp2]:
            ch1_signal = predicted_signals[fp_name]["Channel 1"]
            ch2_signal = predicted_signals[fp_name]["Channel 2"]
            pred_vec = np.array([ch1_signal, ch2_signal])
            pred_vec = pred_vec / np.linalg.norm(pred_vec)
            predicted_vectors[fp_name] = pred_vec
        
        # mCherry angle predicted
        mcherry_vec_pred = predicted_vectors.get('mCherry', predicted_vectors.get(fp1 if 'Cherry' in fp1 else fp2))
        if mcherry_vec_pred is not None:
            mcherry_angle_rad_pred = np.arctan2(mcherry_vec_pred[1], mcherry_vec_pred[0])
            mcherry_angle_predicted = np.degrees(mcherry_angle_rad_pred)
            mcherry_angle_predicted = np.abs(mcherry_angle_predicted) % 180
            mcherry_angle_predicted = min(mcherry_angle_predicted, 180 - mcherry_angle_predicted)
        else:
            mcherry_angle_predicted = None
        
        # mNeptune angle predicted
        mneptune_vec_pred = predicted_vectors.get('mNeptune', predicted_vectors.get(fp1 if 'Neptune' in fp1 else fp2))
        if mneptune_vec_pred is not None:
            mneptune_angle_rad_pred = np.arctan2(mneptune_vec_pred[1], mneptune_vec_pred[0])
            mneptune_angle_predicted = np.degrees(mneptune_angle_rad_pred)
            mneptune_angle_predicted = np.abs(mneptune_angle_predicted) % 180
            mneptune_angle_predicted = min(mneptune_angle_predicted, 180 - mneptune_angle_predicted)
        else:
            mneptune_angle_predicted = None
        
        angles_between_vectors.append(angle_deg)
        percent_corrects.append(pct_correct)
        is_known_row.append(is_known)
        row_names.append(row_name)
        mean_intensities.append(mean_intensity)
        
        # Calculate mNeptune error (signed difference: actual - predicted)
        # Positive = predicted is smaller (underestimated), Negative = predicted is larger (overestimated)
        mneptune_error = None
        if mneptune_angle_actual is not None and mneptune_angle_predicted is not None:
            mneptune_error = mneptune_angle_actual - mneptune_angle_predicted
        
        # Store for CSV (will be combined with 9.2 data)
        csv_data.append({
            'fp1': fp1,
            'fp2': fp2,
            'mNeptune_error': mneptune_error,
            'acquisition_1': acq1_name,
            'acquisition_2': acq2_name,
            'angle_between_vectors': angle_deg,
            'mCherry_angle_actual': mcherry_angle_actual,
            'mCherry_angle_predicted': mcherry_angle_predicted,
            'mNeptune_angle_actual': mneptune_angle_actual,
            'mNeptune_angle_predicted': mneptune_angle_predicted,
            'separability_score': None,  # Will be filled from 9.2
            'percent_correct': pct_correct
        })
    
    # Plot all points
    for i, (angle, pct, is_known, name, mean_int) in enumerate(zip(
        angles_between_vectors, percent_corrects, is_known_row, row_names, mean_intensities)):
        
        if is_known and name:
            # Highlight known rows with blue shades and markers
            color = row_colors.get(name.lower(), '#1f77b4')  # Default blue if name not found
            marker = row_markers.get(name.lower(), 'o')  # Default circle if name not found
            ax.scatter(angle, pct, color=color, marker=marker, s=150, label=name, zorder=3, 
                      edgecolors='black', linewidths=1.5)
            # Removed text labels
        else:
            # Unlabeled dots for other pairs
            ax.scatter(angle, pct, color='gray', s=30, alpha=0.5, zorder=1)
    
    # Add 50% chance line (thicker to match 9.0)
    if len(angles_between_vectors) > 0:
        x_max = max(angles_between_vectors)
        ax.axhline(50, color='gray', linestyle='--', linewidth=3, alpha=0.5, label='Chance (50%)', zorder=2)
    
    # CSV will be saved in 9.2 with combined data
    
    ax.set_xlabel("Angle Between Mean Vectors (degrees)", fontsize=12)
    ax.set_ylabel("Percent Correct", fontsize=12)
    ax.set_title("9.1: Percent Correct vs Angle Between Vectors", fontsize=12, fontweight='bold')
    
    # Set axis limits based on data
    if len(angles_between_vectors) > 0:
        x_min = min(angles_between_vectors)
        x_max = max(angles_between_vectors)
        # Add some padding
        x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 1
        ax.set_xlim(max(0, x_min - x_padding), min(90, x_max + x_padding))
    else:
        ax.set_xlim(0, 90)
    
    if len(percent_corrects) > 0:
        y_min = min(percent_corrects)
        y_max = max(percent_corrects)
        # Start y-axis closer to lowest point, but ensure 50% line is visible
        y_min_lim = min(y_min - 5, 45)  # At least show down to 45% to see 50% line
        y_max_lim = min(100, y_max + 5)
        ax.set_ylim(y_min_lim, y_max_lim)
    else:
        ax.set_ylim(0, 100)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)  # Moved legend down
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_9_3_actual_vs_predicted_angle(processed_pairs, ax, data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619"):
    """
    Plot subpanel 9.3: Scatterplot of actual angle vs predicted angle.
    
    Uses pairs from 9.2. For each pair and each FP, computes:
    - Actual angle: angle of data vector to nearest axis (min(angle, 90-angle))
    - Predicted angle: angle of predicted vector to nearest axis (min(angle, 90-angle))
    - Plots one point per FP per pair
    - Uses different symbols for excitation/emission/dual and mCherry/mNeptune colors
    
    Parameters
    ----------
    processed_pairs : list of dict
        List of processed pair dictionaries from 9.2
    ax : matplotlib.axes.Axes
        Axes to plot on
    data_dir : str
        Path to data directory
    """
    actual_angles = []
    predicted_angles = []
    fp_names_list = []
    row_names_list = []
    is_known_list = []
    is_cross_wavelength_list = []
    
    # Map row names to markers
    row_markers = {
        'excitation based': 'o',  # circle
        'emission based': 's',    # square
        'dual domain': '^'        # triangle
    }
    
    fp_colors = {
        'mCherry': cfg.fluorophore_colors.get('mCherry', '#E31A1C'),
        'mNeptune': cfg.fluorophore_colors.get('mNeptune', '#4B0082')
    }
    
    for pair in processed_pairs:
        # Check if this is a cross-wavelength comparison (1080 vs 1240)
        is_cross_wavelength = (pair['ch1_wl'] != pair['ch2_wl'])
        
        fp1 = pair['fp1']
        fp2 = pair['fp2']
        data_vectors = pair['data_vectors']
        
        # Create a row_dict-like structure for computing predicted vectors
        row_dict_like = {
            'Fluorophores': [fp1, fp2],
            'Channel 1': {'Excitation wavelength': pair['ch1_wl'], 'emission filter': pair['ch1_filter']},
            'Channel 2': {'Excitation wavelength': pair['ch2_wl'], 'emission filter': pair['ch2_filter']}
        }
        
        # Get predicted vectors
        ch1_pockels = pair.get('ch1_pockels')
        ch2_pockels = pair.get('ch2_pockels')
        
        # If Pockels values are missing, try to extract from acquisition names
        if ch1_pockels is None or ch2_pockels is None:
            acq1 = pair.get('acquisition_1', '')
            acq2 = pair.get('acquisition_2', '')
            if ch1_pockels is None and 'poc' in acq1:
                ch1_pockels = extract_pockels_from_filename(acq1)
            if ch2_pockels is None and 'poc' in acq2:
                ch2_pockels = extract_pockels_from_filename(acq2)
        
        try:
            # Debug flag for specific problematic cases
            debug_case = ((pair['ch1_wl'] == 1080 and pair['ch1_filter'] == 'broad' and 
                          pair['ch2_wl'] == 1080 and pair['ch2_filter'] == 'FarRed') or
                         (pair['ch1_wl'] == 1240 and pair['ch1_filter'] == 'BR2' and 
                          pair['ch2_wl'] == 1240 and pair['ch2_filter'] == 'Red'))
            predicted_signals = compute_predicted_channel_signals(row_dict_like, ch1_pockels=ch1_pockels, ch2_pockels=ch2_pockels, debug=debug_case)
            
            predicted_vectors = {}
            for fp_name in [fp1, fp2]:
                ch1_signal = predicted_signals[fp_name]["Channel 1"]
                ch2_signal = predicted_signals[fp_name]["Channel 2"]
                pred_vec = np.array([ch1_signal, ch2_signal])
                pred_vec = pred_vec / np.linalg.norm(pred_vec)
                predicted_vectors[fp_name] = pred_vec
                
                # Debug for specific cases
                if (((pair['ch1_wl'] == 1080 and pair['ch1_filter'] == 'broad' and 
                      pair['ch2_wl'] == 1080 and pair['ch2_filter'] == 'FarRed') or
                     (pair['ch1_wl'] == 1240 and pair['ch1_filter'] == 'BR2' and 
                      pair['ch2_wl'] == 1240 and pair['ch2_filter'] == 'Red')) and
                    fp_name == 'mNeptune'):
                    angle_rad = np.arctan2(ch2_signal, ch1_signal)
                    angle_deg = np.degrees(angle_rad)
                    angle_to_axis = min(angle_deg % 180, 180 - (angle_deg % 180))
                    print(f"\nDEBUG 9.3 mNeptune: {pair['ch1_wl']}nm_{pair['ch1_filter']}_{ch1_pockels}poc vs {pair['ch2_wl']}nm_{pair['ch2_filter']}_{ch2_pockels}poc")
                    print(f"  Ch1 signal: {ch1_signal:.6f}, Ch2 signal: {ch2_signal:.6f}")
                    print(f"  Ratio (Ch2/Ch1): {ch2_signal/ch1_signal:.4f}")
                    print(f"  Predicted angle: {angle_deg:.2f}°, angle to axis: {angle_to_axis:.2f}°")
        except Exception as e:
            print(f"Error computing predicted signals: {e}")
            continue
        
        # Get row info
        is_known = pair.get('is_known', False)
        row_name = pair.get('row_name', None)
        
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
            row_names_list.append(row_name)
            is_known_list.append(is_known)
            # Store whether this is a cross-wavelength comparison
            is_cross_wavelength_list.append(is_cross_wavelength)
    
    # Plot points: all colored by FP, known rows with symbols and larger size
    for i, (actual, predicted, fp_name, row_name, is_known, is_cross_wl) in enumerate(zip(
        actual_angles, predicted_angles, fp_names_list, row_names_list, is_known_list, is_cross_wavelength_list)):
        
        # Get color for this FP (use FP color if available, otherwise gray)
        color = fp_colors.get(fp_name, 'gray')
        
        # Determine marker: star for cross-wavelength, otherwise use row type marker
        if is_cross_wl:
            marker = '*'  # Star for 1080 vs 1240 comparisons
        elif is_known and row_name:
            marker = row_markers.get(row_name.lower(), 'o')
        else:
            marker = 'o'  # Default circle
        
        if is_known and row_name:
            # Known row: use symbol for row type or star, FP color, larger size
            ax.scatter(actual, predicted, color=color, marker=marker, s=100,
                      alpha=0.7, edgecolors='black', linewidths=1, zorder=3)
        else:
            # Unknown: FP color, smaller dots
            ax.scatter(actual, predicted, color=color, marker=marker, s=30, alpha=0.5, zorder=1)
    
    # Plot diagonal line (perfect agreement, no label)
    if len(actual_angles) > 0:
        max_angle = max(max(actual_angles), max(predicted_angles))
        ax.plot([0, max_angle], [0, max_angle], 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel("Actual Angle (degrees)", fontsize=12)
    ax.set_ylabel("Predicted Angle (degrees)", fontsize=12)
    ax.set_title("9.3: Actual vs Predicted Angle", fontsize=12, fontweight='bold')
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend with row type symbols
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
              markeredgecolor='black', markeredgewidth=1, label='Excitation based'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8,
              markeredgecolor='black', markeredgewidth=1, label='Emission based'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8,
              markeredgecolor='black', markeredgewidth=1, label='Dual domain')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_9_4_actual_vs_predicted_variance(processed_pairs, ax, data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619"):
    """
    Plot subpanel 9.4: Scatterplot of actual variance vs predicted variance.
    
    For each FP in each pair, computes:
    - Actual variance: variance of pixel distances perpendicular to the data vector
    - Predicted variance: 95% perpendicular noise interval squared (variance equivalent)
    
    Uses the same N_fluorophores as in 9.2 for comparability.
    
    Parameters
    ----------
    processed_pairs : list of dict
        List of processed pair dictionaries from 9.2
    ax : matplotlib.axes.Axes
        Axes to plot on
    data_dir : str
        Path to data directory
    """
    actual_variances = []
    predicted_variances = []
    fp_names = []
    row_names_list = []
    is_known_list = []
    csv_data_variance = []  # Store variance data for CSV
    
    # Map row names to markers
    row_markers = {
        'excitation based': 'o',  # circle
        'emission based': 's',    # square
        'dual domain': '^'        # triangle
    }
    
    fp_colors = {
        'mCherry': cfg.fluorophore_colors.get('mCherry', '#E31A1C'),
        'mNeptune': cfg.fluorophore_colors.get('mNeptune', '#4B0082')
    }
    
    for pair in processed_pairs:
        fp1 = pair['fp1']
        fp2 = pair['fp2']
        data_vectors = pair['data_vectors']
        fp1_ch1 = pair['fp1_ch1']
        fp1_ch2 = pair['fp1_ch2']
        fp2_ch1 = pair['fp2_ch1']
        fp2_ch2 = pair['fp2_ch2']
        
        # Create a row_dict-like structure for computing predicted vectors
        row_dict_like = {
            'Fluorophores': [fp1, fp2],
            'Channel 1': {'Excitation wavelength': pair['ch1_wl'], 'emission filter': pair['ch1_filter']},
            'Channel 2': {'Excitation wavelength': pair['ch2_wl'], 'emission filter': pair['ch2_filter']}
        }
        
        # Get predicted signals at N fluorophores
        ch1_pockels = pair.get('ch1_pockels')
        ch2_pockels = pair.get('ch2_pockels')
        try:
            predicted_signals = compute_predicted_signals_at_N(
                row_dict_like, cfg.N_fluorophores_default,
                ch1_pockels=ch1_pockels, ch2_pockels=ch2_pockels
            )
        except:
            continue
        
        # Compute mean distances from origin for each FP
        mean_distances = {}
        for fp_name in [fp1, fp2]:
            if fp_name not in data_vectors:
                continue
            
            # Get data for this FP
            if fp_name == fp1:
                ch1_data = fp1_ch1
                ch2_data = fp1_ch2
            else:
                ch1_data = fp2_ch1
                ch2_data = fp2_ch2
            
            # Compute mean distance from origin
            distances = np.sqrt(ch1_data.astype(np.float64)**2 + ch2_data.astype(np.float64)**2)
            valid_distances = distances[distances > 10]  # Filter out near-origin
            if len(valid_distances) > 0:
                mean_distances[fp_name] = np.mean(valid_distances)
            else:
                mean_distances[fp_name] = None
        
        # For each FP, compute actual and predicted variance
        csv_rows = []  # Store data for CSV
        for fp_name in [fp1, fp2]:
            if fp_name not in data_vectors:
                continue
            
            # Get data vector
            vec_data = data_vectors[fp_name]
            
            # Get data for this FP
            if fp_name == fp1:
                ch1_data = fp1_ch1
                ch2_data = fp1_ch2
            else:
                ch1_data = fp2_ch1
                ch2_data = fp2_ch2
            
            # Get mean distance for this FP
            mean_distance_actual = mean_distances.get(fp_name)
            if mean_distance_actual is None:
                continue
            
            # Get predicted signals at default N
            if fp_name not in predicted_signals:
                continue
            
            # Scale N_fluorophores to match actual mean distance
            # The predicted distance is the magnitude of the predicted signal vector
            pred_vec_magnitude = np.linalg.norm([
                predicted_signals[fp_name]["Channel 1"],
                predicted_signals[fp_name]["Channel 2"]
            ])
            
            if pred_vec_magnitude > 0:
                # Scale factor to match actual distance
                scale_factor = mean_distance_actual / pred_vec_magnitude
                N_scaled = cfg.N_fluorophores_default * scale_factor
            else:
                N_scaled = cfg.N_fluorophores_default
            
            # Recompute predicted signals at scaled N
            predicted_signals_scaled = compute_predicted_signals_at_N(
                row_dict_like, N_scaled,
                ch1_pockels=ch1_pockels, ch2_pockels=ch2_pockels
            )
            
            if fp_name not in predicted_signals_scaled:
                continue
            
            # Compute actual variance using range around mean distance
            actual_var, computed_distance = compute_actual_variance_perpendicular(
                ch1_data, ch2_data, vec_data, 
                target_distance=mean_distance_actual, range_width=50
            )
            
            if np.isnan(actual_var) or actual_var <= 0 or np.isnan(computed_distance):
                continue
            
            # Get predicted mu values at scaled N
            mu_values = np.array([
                predicted_signals_scaled[fp_name]["Channel 1"],
                predicted_signals_scaled[fp_name]["Channel 2"]
            ])
            
            # Get predicted vector (normalized)
            pred_vec = np.array([
                predicted_signals_scaled[fp_name]["Channel 1"],
                predicted_signals_scaled[fp_name]["Channel 2"]
            ])
            pred_vec = pred_vec / np.linalg.norm(pred_vec)
            
            # Compute 95% perpendicular interval
            delta_95_perp = compute_perpendicular_variance_95_interval(pred_vec, mu_values)
            
            # Convert to variance (square the interval to get variance-like measure)
            # The 95% interval is 1.645 * std, so std = delta_95_perp / 1.645
            # variance = std^2 = (delta_95_perp / 1.645)^2
            predicted_var = (delta_95_perp / 1.645) ** 2
            
            # Get row info
            is_known = pair.get('is_known', False)
            row_name = pair.get('row_name', None)
            
            actual_variances.append(actual_var)
            predicted_variances.append(predicted_var)
            fp_names.append(fp_name)
            row_names_list.append(row_name)
            is_known_list.append(is_known)
            
            # Store for CSV
            csv_data_variance.append({
                'fp1': fp1,
                'fp2': fp2,
                'acquisition_1': _format_acquisition_name(pair['ch1_wl'], pair['ch1_filter'], pair.get('ch1_pockels')),
                'acquisition_2': _format_acquisition_name(pair['ch2_wl'], pair['ch2_filter'], pair.get('ch2_pockels')),
                'fp_name': fp_name,
                'mean_mCherry_distance': mean_distances.get('mCherry', None),
                'mean_mNeptune_distance': mean_distances.get('mNeptune', None),
                'computed_distance_from_origin': computed_distance,
                'N_fluorophores_scaled': N_scaled,
                'predicted_variance_at_computed_distance': predicted_var,
                'measured_variance_at_computed_distance': actual_var
            })
    
    if len(actual_variances) == 0:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("9.4: Actual vs Predicted Variance", fontsize=12, fontweight='bold')
        return
    
    # Plot points: all colored by FP, known rows with symbols and larger size
    for i, (predicted, actual, fp_name, row_name, is_known) in enumerate(zip(
        predicted_variances, actual_variances, fp_names, row_names_list, is_known_list)):
        
        # Get color for this FP (use FP color if available, otherwise gray)
        color = fp_colors.get(fp_name, 'gray')
        
        if is_known and row_name:
            # Known row: use symbol for row type, FP color, larger size
            marker = row_markers.get(row_name.lower(), 'o')
            ax.scatter(predicted, actual, color=color, marker=marker, s=100,
                      alpha=0.7, edgecolors='black', linewidths=1, zorder=3)
        else:
            # Unknown: FP color, smaller dots
            ax.scatter(predicted, actual, color=color, s=30, alpha=0.5, zorder=1)
    
    # Add diagonal line for perfect agreement
    if len(actual_variances) > 0:
        min_val = min(min(actual_variances), min(predicted_variances))
        max_val = max(max(actual_variances), max(predicted_variances))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5, label='Perfect agreement', zorder=1)
    
    ax.set_xlabel("Predicted Variance", fontsize=12)
    ax.set_ylabel("Actual Variance", fontsize=12)
    ax.set_title("9.4: Actual vs Predicted Variance", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    # No fixed aspect ratio - let axes scale independently for better data visibility
    ax.set_aspect('auto')  # Explicitly set to auto to allow independent scaling
    # Set axis limits: x-axis to 2000, y-axis to 20000
    if len(predicted_variances) > 0:
        ax.set_xlim(0, 2000)
    if len(actual_variances) > 0:
        ax.set_ylim(0, 20000)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Save variance data to CSV
    if len(csv_data_variance) > 0:
        import pandas as pd
        csv_df = pd.DataFrame(csv_data_variance)
        csv_path = os.path.join("results", "Figure1", "subpanel9_4_variance.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        csv_df.to_csv(csv_path, index=False)
        print(f"  Saved variance results to {csv_path}")


def generate_row_subpanels(row_dict):
    """
    Generate all subpanels for a single row configuration.
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
        
    Returns
    -------
    dict
        Dictionary of (subpanel_number, (fig, ax)) tuples
    """
    subpanels = {}
    
    subpanels[1] = subpanel_1(row_dict)
    subpanels[2] = subpanel_2(row_dict)
    subpanels[3] = subpanel_3(row_dict)
    subpanels[4] = subpanel_4(row_dict)
    subpanels[5] = subpanel_5(row_dict)
    subpanels[6] = subpanel_6(row_dict)
    subpanels[7] = subpanel_7(row_dict)
    
    return subpanels


def generate_all_subpanels(row_list):
    """
    Generate all subpanels for all rows.
    
    Parameters
    ----------
    row_list : list of dict
        List of row configuration dictionaries
        
    Returns
    -------
    dict
        Dictionary with structure: {row_index: {subpanel_number: (fig, ax)}}
    """
    all_subpanels = {}
    
    # Generate subpanels for each row
    for i, row_dict in enumerate(row_list):
        all_subpanels[i] = generate_row_subpanels(row_dict)
    
    # Generate subpanels that combine all rows (subpanel 9)
    all_subpanels['combined'] = {}
    all_subpanels['combined'][9] = subpanel_9(row_list)
    
    # Generate subpanel 8 for each row separately
    for i, row_dict in enumerate(row_list):
        all_subpanels[i][8] = subpanel_8(row_dict)
    
    return all_subpanels


def save_all_subpanels(row_list, output_dir="results/Figure1"):
    """
    Generate and save all subpanels for all rows with simple filenames.
    
    Parameters
    ----------
    row_list : list of dict
        List of row configuration dictionaries
    output_dir : str
        Directory to save figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Subpanels 1 and 2 are shared across rows (not row-specific).
    # Save them once with a "combined" prefix.
    combined_once_subpanels = {1, 2}
    combined_already_saved = set()
    
    # Generate and save subpanels for each row
    for row_idx, row_dict in enumerate(row_list):
        row_name = row_dict.get("name", f"Row{row_idx+1}").replace(" ", "_")
        subpanels = generate_row_subpanels(row_dict)
        
        for subpanel_num, result in subpanels.items():
            # Handle different return types
            # Subpanels return (fig, ax) or (fig, axes) tuple
            if isinstance(result, tuple) and len(result) == 2:
                fig, ax_or_axes = result
            else:
                # Fallback if structure is different
                fig = result if hasattr(result, 'savefig') else result[0]
            
            if subpanel_num in combined_once_subpanels:
                if subpanel_num in combined_already_saved:
                    plt.close(fig)
                    continue
                filename = f"combined_subpanel{subpanel_num}.png"
                combined_already_saved.add(subpanel_num)
            else:
                filename = f"{row_name}_subpanel{subpanel_num}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {filepath}")
    
    # Generate and save subpanel 8 for each row separately
    for row_idx, row_dict in enumerate(row_list):
        row_name = row_dict.get("name", f"Row{row_idx+1}").replace(" ", "_")
        fig8, ax8 = subpanel_8(row_dict)
        filename8 = f"{row_name}_subpanel8.png"
        filepath8 = os.path.join(output_dir, filename8)
        fig8.savefig(filepath8, dpi=300, bbox_inches='tight')
        plt.close(fig8)
        print(f"Saved: {filepath8}")
    
    # Generate and save combined subpanel 9
    fig9, axes9 = subpanel_9(row_list)
    filename9 = "combined_subpanel9.png"
    filepath9 = os.path.join(output_dir, filename9)
    fig9.savefig(filepath9, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath9}")
    
    # Save individual subplots
    subplot_names = ["9_0_percent_correct_vs_intensity", 
                     "9_1_percent_correct_vs_angle_separation",
                     "9_2_percent_correct_vs_separability",
                     "9_3_actual_vs_predicted_angle",
                     "9_4_actual_vs_predicted_variance"]
    
    for i, (ax, name) in enumerate(zip(axes9, subplot_names)):
        # Create a new figure for this subplot
        fig_single = plt.figure(figsize=(6, 5))
        ax_single = fig_single.add_subplot(111)
        
        # Copy the subplot content
        # Get the title and labels
        title = ax.get_title()
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        
        # Get limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Copy all artists (lines, scatter plots, etc.)
        for artist in ax.get_children():
            try:
                if isinstance(artist, plt.Line2D):
                    # Line plot
                    xdata, ydata = artist.get_data()
                    if len(xdata) > 0:
                        ax_single.plot(xdata, ydata, 
                                      color=artist.get_color(),
                                      linestyle=artist.get_linestyle(),
                                      linewidth=artist.get_linewidth(),
                                      alpha=artist.get_alpha(),
                                      label=artist.get_label() if artist.get_label() else '',
                                      marker=artist.get_marker() if artist.get_marker() != 'None' else None,
                                      markersize=artist.get_markersize())
                elif hasattr(artist, 'get_offsets'):
                    # Scatter plot (PathCollection)
                    offsets = artist.get_offsets()
                    if len(offsets) > 0:
                        facecolors = artist.get_facecolors()
                        edgecolors = artist.get_edgecolors()
                        sizes = artist.get_sizes()
                        linewidths = artist.get_linewidths()
                        # Get marker if available
                        marker = 'o'  # default
                        if hasattr(artist, 'get_paths') and len(artist.get_paths()) > 0:
                            # Try to infer marker from path
                            marker = 'o'  # Default to circle
                        
                        ax_single.scatter(offsets[:, 0], offsets[:, 1],
                                         c=facecolors if len(facecolors) > 0 else 'gray',
                                         s=sizes if len(sizes) > 0 else 30,
                                         alpha=artist.get_alpha() if hasattr(artist, 'get_alpha') else 1.0,
                                         edgecolors=edgecolors if len(edgecolors) > 0 else 'none',
                                         linewidths=linewidths if len(linewidths) > 0 else 0.5,
                                         marker=marker)
            except Exception as e:
                # Skip elements that can't be copied
                pass
        
        # Copy legend if it exists
        if ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax_single.legend(handles, labels, loc=ax.get_legend()._loc, fontsize=ax.get_legend().get_fontsize())
        
        # Set properties
        ax_single.set_title(title, fontsize=ax.title.get_fontsize(), fontweight=ax.title.get_fontweight())
        ax_single.set_xlabel(xlabel, fontsize=ax.xaxis.label.get_fontsize())
        ax_single.set_ylabel(ylabel, fontsize=ax.yaxis.label.get_fontsize())
        ax_single.set_xlim(xlim)
        ax_single.set_ylim(ylim)
        ax_single.grid(ax.get_grid(), alpha=0.3)
        ax_single.set_aspect(ax.get_aspect(), adjustable=ax.get_adjustable())
        
        # Copy spine visibility
        for spine_name in ['top', 'right', 'bottom', 'left']:
            ax_single.spines[spine_name].set_visible(ax.spines[spine_name].get_visible())
        
        # Save individual subplot
        filename_single = f"{name}.png"
        filepath_single = os.path.join(output_dir, filename_single)
        fig_single.savefig(filepath_single, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Saved: {filepath_single}")
    
    plt.close(fig9)


if __name__ == "__main__":
    # Generate and save all subpanels
    save_all_subpanels(row_list)

