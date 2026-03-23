"""
Shared helper functions for figure generation across figures.

This module contains reusable functions for:
- Vector computation from data
- Distance-based filtering and binning
- Pixel subsampling
- Pixel classification by angle
- Variance calculations
- Classification zone computation
- Ratio histogram plotting
- Spectra loading (2P excitation and 1P emission)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src import config as cfg


def compute_data_vector(ch1_data, ch2_data, lower_percentile=None, upper_percentile=None):
    """
    Compute unit vector from data by filtering pixels and computing mean angle.
    
    Parameters
    ----------
    ch1_data : np.ndarray
        Channel 1 pixel intensities
    ch2_data : np.ndarray
        Channel 2 pixel intensities
    lower_percentile : float, optional
        Lower percentile to filter (default from config)
    upper_percentile : float, optional
        Upper percentile to filter (default from config)
        
    Returns
    -------
    np.ndarray
        Unit vector [ch1_component, ch2_component]
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
    
    # Keep pixels in the "middle chunk"
    mask = ((ch1_data >= ch1_lower) & (ch1_data <= ch1_upper) &
            (ch2_data >= ch2_lower) & (ch2_data <= ch2_upper))
    
    ch1_filtered = ch1_data[mask]
    ch2_filtered = ch2_data[mask]
    
    # Compute mean angle (mean of normalized vectors)
    # Normalize each pixel vector
    magnitudes = np.sqrt(ch1_filtered**2 + ch2_filtered**2)
    valid_mask = magnitudes > 0
    ch1_normalized = ch1_filtered[valid_mask] / magnitudes[valid_mask]
    ch2_normalized = ch2_filtered[valid_mask] / magnitudes[valid_mask]
    
    # Mean normalized vector
    mean_ch1 = np.mean(ch1_normalized)
    mean_ch2 = np.mean(ch2_normalized)
    
    # Normalize to unit vector
    mean_magnitude = np.sqrt(mean_ch1**2 + mean_ch2**2)
    if mean_magnitude > 0:
        unit_vector = np.array([mean_ch1 / mean_magnitude, mean_ch2 / mean_magnitude])
    else:
        unit_vector = np.array([1.0, 0.0])  # Default if no valid pixels
    
    return unit_vector


def vector_angle(vector):
    """
    Compute angle of a vector from positive x-axis.
    
    Parameters
    ----------
    vector : np.ndarray
        Vector [ch1_component, ch2_component]
        
    Returns
    -------
    float
        Angle in degrees (0-360)
    """
    angle_rad = np.arctan2(vector[1], vector[0])
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg


def filter_by_distance(ch1_data, ch2_data, labels=None, max_distance=None, min_distance=0):
    """
    Filter pixels by distance from origin.
    
    Parameters
    ----------
    ch1_data : np.ndarray
        Channel 1 pixel intensities
    ch2_data : np.ndarray
        Channel 2 pixel intensities
    labels : np.ndarray, optional
        Labels for each pixel (e.g., fluorophore names)
    max_distance : float, optional
        Maximum distance from origin (default: no upper limit)
    min_distance : float
        Minimum distance from origin (default: 0)
        
    Returns
    -------
    ch1_filtered : np.ndarray
        Filtered channel 1 data
    ch2_filtered : np.ndarray
        Filtered channel 2 data
    labels_filtered : np.ndarray or None
        Filtered labels (if provided)
    distances : np.ndarray
        Distances from origin for filtered pixels
    """
    # Ensure arrays are 1D
    if len(ch1_data.shape) > 1:
        ch1_data = ch1_data.flatten()
    if len(ch2_data.shape) > 1:
        ch2_data = ch2_data.flatten()
    
    # Compute distance from origin
    ch1_float = ch1_data.astype(np.float64)
    ch2_float = ch2_data.astype(np.float64)
    distances = np.sqrt(ch1_float**2 + ch2_float**2)
    
    # Create mask
    mask = distances >= min_distance
    if max_distance is not None:
        mask = mask & (distances <= max_distance)
    
    ch1_filtered = ch1_data[mask]
    ch2_filtered = ch2_data[mask]
    distances_filtered = distances[mask]
    
    if labels is not None:
        labels_filtered = np.array(labels, dtype=object)[mask]
    else:
        labels_filtered = None
    
    return ch1_filtered, ch2_filtered, labels_filtered, distances_filtered


def bin_and_subsample_by_distance(ch1_data, ch2_data, labels, bin_width=100, 
                                   samples_per_bin=300, max_distance=None):
    """
    Bin pixels by distance from origin and subsample evenly from each bin.
    
    Samples the same number of points from each bin, separately for each
    unique label (e.g., per fluorophore).
    
    Parameters
    ----------
    ch1_data : np.ndarray
        Channel 1 pixel intensities
    ch2_data : np.ndarray
        Channel 2 pixel intensities
    labels : np.ndarray
        Labels for each pixel (e.g., fluorophore names)
    bin_width : float
        Width of each distance bin (default: 100)
    samples_per_bin : int
        Number of samples to take from each bin per label (default: 300)
    max_distance : float, optional
        Maximum distance to consider (default: computed from data)
        
    Returns
    -------
    ch1_sampled : np.ndarray
        Subsampled channel 1 data
    ch2_sampled : np.ndarray
        Subsampled channel 2 data
    labels_sampled : np.ndarray
        Subsampled labels
    """
    # Ensure arrays are 1D
    if len(ch1_data.shape) > 1:
        ch1_data = ch1_data.flatten()
    if len(ch2_data.shape) > 1:
        ch2_data = ch2_data.flatten()
    
    # Compute distances
    ch1_float = ch1_data.astype(np.float64)
    ch2_float = ch2_data.astype(np.float64)
    distances = np.sqrt(ch1_float**2 + ch2_float**2)
    
    # Determine max distance if not provided
    if max_distance is None:
        max_distance = np.max(distances)
    
    # Determine number of bins
    n_bins = int(np.ceil(max_distance / bin_width))
    
    # Get unique labels (e.g., fluorophore names)
    unique_labels = np.unique(labels)
    
    # Collect sampled points
    ch1_sampled_list = []
    ch2_sampled_list = []
    labels_sampled_list = []
    
    for bin_idx in range(n_bins):
        bin_min = bin_idx * bin_width
        bin_max = (bin_idx + 1) * bin_width
        
        # Create mask for this bin
        if bin_idx == 0:
            bin_mask = distances < bin_max
        else:
            bin_mask = (distances >= bin_min) & (distances < bin_max)
        
        if not np.any(bin_mask):
            continue
        
        ch1_bin = ch1_data[bin_mask]
        ch2_bin = ch2_data[bin_mask]
        labels_bin = np.array(labels, dtype=object)[bin_mask]
        
        # Sample separately for each label
        for label in unique_labels:
            label_mask = labels_bin == label
            if not np.any(label_mask):
                continue
            
            ch1_label = ch1_bin[label_mask]
            ch2_label = ch2_bin[label_mask]
            n_label = len(ch1_label)
            n_take = min(samples_per_bin, n_label)
            
            if n_label > n_take:
                indices = np.random.choice(n_label, n_take, replace=False)
                ch1_sampled_list.append(ch1_label[indices])
                ch2_sampled_list.append(ch2_label[indices])
                labels_sampled_list.append(np.array([label] * n_take, dtype=object))
            else:
                ch1_sampled_list.append(ch1_label)
                ch2_sampled_list.append(ch2_label)
                labels_sampled_list.append(np.array([label] * n_label, dtype=object))
    
    # Combine all sampled points
    if len(ch1_sampled_list) > 0:
        ch1_sampled = np.concatenate(ch1_sampled_list)
        ch2_sampled = np.concatenate(ch2_sampled_list)
        labels_sampled = np.concatenate(labels_sampled_list)
    else:
        # Fallback if no bins had points
        ch1_sampled = ch1_data
        ch2_sampled = ch2_data
        labels_sampled = np.array(labels, dtype=object)
    
    return ch1_sampled, ch2_sampled, labels_sampled


def classify_pixel_by_angle(ch1_val, ch2_val, vectors_dict):
    """
    Classify a pixel by finding which vector it's closest to (by angle).
    
    Parameters
    ----------
    ch1_val : float
        Channel 1 intensity
    ch2_val : float
        Channel 2 intensity
    vectors_dict : dict
        Dictionary mapping fluorophore names to unit vectors
        
    Returns
    -------
    str or None
        Name of closest fluorophore, or None if pixel is at origin
    """
    if ch1_val == 0 and ch2_val == 0:
        return None
    
    # Normalize pixel vector
    magnitude = np.sqrt(ch1_val**2 + ch2_val**2)
    pixel_vec = np.array([ch1_val / magnitude, ch2_val / magnitude])
    
    # Compute angle to each vector
    min_angle = float('inf')
    closest_fp = None
    
    for fp_name, vec in vectors_dict.items():
        # Compute angle between pixel and vector
        dot_product = np.clip(np.dot(pixel_vec, vec), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))
        # Take minimum of angle and 180-angle (0-90 range)
        angle = min(angle, 180 - angle)
        
        if angle < min_angle:
            min_angle = angle
            closest_fp = fp_name
    
    return closest_fp


def compute_classification_zone(ch1_data, ch2_data, fp_labels, fp_name, vector, 
                               percentile=80, min_distance=500):
    """
    Compute symmetric angle zone that contains a given percentile of pixels.
    
    Parameters
    ----------
    ch1_data : np.ndarray
        Channel 1 pixel intensities (subsampled)
    ch2_data : np.ndarray
        Channel 2 pixel intensities (subsampled)
    fp_labels : np.ndarray
        Labels indicating which fluorophore each pixel belongs to
    fp_name : str
        Name of fluorophore to compute zone for
    vector : np.ndarray
        Reference unit vector for this fluorophore
    percentile : float
        Percentile to include (default 80)
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
    
    # Filter by minimum distance
    distances = np.sqrt(ch1_fp.astype(np.float64)**2 + ch2_fp.astype(np.float64)**2)
    bright_mask = distances >= min_distance
    
    if not np.any(bright_mask):
        return None
    
    ch1_bright = ch1_fp[bright_mask]
    ch2_bright = ch2_fp[bright_mask]
    
    # Compute pixel angles (from x-axis, mapped to 0-90 degrees)
    pixel_angles_rad = np.arctan2(ch2_bright, ch1_bright)
    pixel_angles_deg = np.degrees(pixel_angles_rad)
    # Map all angles to 0-90 range (take absolute and fold)
    pixel_angles_deg = np.abs(pixel_angles_deg) % 180
    pixel_angles_deg = np.where(pixel_angles_deg > 90, 180 - pixel_angles_deg, pixel_angles_deg)
    
    # Compute data vector angle (from x-axis, mapped to 0-90 degrees)
    vec_angle_rad = np.arctan2(vector[1], vector[0])
    vec_angle_deg = np.degrees(vec_angle_rad)
    vec_angle_deg = np.abs(vec_angle_deg) % 180
    vec_angle_deg = vec_angle_deg if vec_angle_deg <= 90 else 180 - vec_angle_deg
    
    # Compute angular differences from vector angle
    # Handle wrap-around: if vec is at 85° and pixel is at 5°, difference is 10° not 80°
    angle_diffs = np.abs(pixel_angles_deg - vec_angle_deg)
    angle_diffs = np.minimum(angle_diffs, 90 - angle_diffs)
    
    valid_angles = angle_diffs[~np.isnan(angle_diffs)]
    
    if len(valid_angles) == 0:
        return None
    
    # Find the half-angle such that [vec_angle - half_angle, vec_angle + half_angle] 
    # contains percentile% of pixel angles (handling wrap-around)
    sorted_diffs = np.sort(valid_angles)
    target_count = int(np.ceil(len(sorted_diffs) * percentile / 100.0))
    
    # The half-angle is simply the target_count-th smallest difference
    # (since we want the smallest symmetric range that contains target_count pixels)
    half_angle = sorted_diffs[target_count - 1] if target_count <= len(sorted_diffs) else sorted_diffs[-1]
    
    # Debug output
    print(f"  compute_classification_zone({fp_name}): {len(valid_angles)} valid angles")
    print(f"    vec_angle={vec_angle_deg:.2f}°, pixel angles: min={np.min(pixel_angles_deg):.2f}°, max={np.max(pixel_angles_deg):.2f}°")
    print(f"    angle_diffs: min={np.min(valid_angles):.2f}°, max={np.max(valid_angles):.2f}°, median={np.median(valid_angles):.2f}°")
    print(f"    target_count={target_count} out of {len(sorted_diffs)}, half_angle={half_angle:.2f}°")
    
    return half_angle


def compute_actual_variance_perpendicular(ch1_data, ch2_data, vector, target_distance=None, range_width=50):
    """
    Compute actual variance of pixels perpendicular to a given vector.
    
    Selects pixels within a range around the target distance from origin,
    then computes variance of their perpendicular distances.
    
    Parameters
    ----------
    ch1_data : np.ndarray
        Channel 1 pixel intensities
    ch2_data : np.ndarray
        Channel 2 pixel intensities
    vector : np.ndarray
        Unit vector [ch1_component, ch2_component]
    target_distance : float, optional
        Target distance from origin. If None, uses mean distance of valid pixels.
    range_width : float
        Range width around target distance (default 50, so +/- 50)
        
    Returns
    -------
    float
        Variance of perpendicular distances (std^2)
    float
        Mean distance from origin of selected pixels
    """
    # Normalize vector
    vector_norm = vector / np.linalg.norm(vector)
    
    # Convert pixels to 2D points
    points = np.column_stack([ch1_data, ch2_data])
    
    # Filter out points at origin
    distances_from_origin = np.linalg.norm(points, axis=1)
    valid_mask = distances_from_origin > 10  # Same threshold as used elsewhere
    
    if np.sum(valid_mask) == 0:
        return np.nan, np.nan
    
    points_valid = points[valid_mask]
    distances_valid = distances_from_origin[valid_mask]
    
    # Determine target distance if not provided
    if target_distance is None:
        target_distance = np.mean(distances_valid)
    
    # Select pixels within range around target distance
    range_mask = (distances_valid >= target_distance - range_width) & (distances_valid <= target_distance + range_width)
    
    if np.sum(range_mask) == 0:
        # If no pixels in range, use all valid pixels
        points_in_range = points_valid
        distances_in_range = distances_valid
    else:
        points_in_range = points_valid[range_mask]
        distances_in_range = distances_valid[range_mask]
    
    if len(points_in_range) == 0:
        return np.nan, np.nan
    
    # Compute perpendicular distances for pixels in range
    # For each point, project onto vector, then compute perpendicular component
    # Perpendicular distance = ||point - (point · vector) * vector||
    projections = np.dot(points_in_range, vector_norm)
    projected_points = projections[:, np.newaxis] * vector_norm
    perpendicular_vectors = points_in_range - projected_points
    perpendicular_distances = np.linalg.norm(perpendicular_vectors, axis=1)
    
    # Compute variance
    variance = np.var(perpendicular_distances)
    mean_distance = np.mean(distances_in_range)
    
    return variance, mean_distance


def create_ratio_histogram_base(row_dict, sorted_channel_keys, sorted_fp_data, 
                                  channel_labels, fluorophores, fp_colors_dict,
                                  figsize_width=20, label_every_other=True):
    """
    Base function for creating ratio histogram plots.
    
    Handles all the common plotting logic:
    - Creates subplots
    - Plots bars
    - Sets labels and formatting
    - Adjusts x-axis width to ensure bars aren't wider than they are tall
    
    Parameters
    ----------
    row_dict : dict
        Row configuration dictionary
    sorted_channel_keys : list
        List of channel keys in sorted order
    sorted_fp_data : dict
        Dictionary of {fp_name: normalized_signal_array} for sorted channels
    channel_labels : list
        List of channel label strings for x-axis
    fluorophores : list
        List of fluorophore names
    fp_colors_dict : dict
        Dictionary mapping fluorophore names to colors (e.g., FIG_5_FP_COLORS or FIG_2_FP_COLORS)
    figsize_width : float
        Width of figure (default 20 for all channels, 12 for best channels)
    label_every_other : bool
        If True, only label every other channel on x-axis (for all channels mode)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    channel_labels : list
        Channel labels (for legend if needed)
    """
    # Create subplots: N+1 subplots (N for each FP, 1 for overlay)
    n_subplots = len(fluorophores) + 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=(figsize_width, 2 * n_subplots), sharex=True)
    
    # Get colors from provided dictionary
    colors = [fp_colors_dict.get(fp_name, "#808080") for fp_name in fluorophores]
    
    x_pos = np.arange(len(sorted_channel_keys))
    width = 1.0  # Full width for no gaps (histogram style)
    
    # Determine max y value for common y-axis
    max_y = max([max(sorted_fp_data[fp]) for fp in fluorophores if fp in sorted_fp_data], default=1.0) * 1.1
    
    # Create subplot for each fluorophore
    for fp_idx, fp_name in enumerate(fluorophores):
        ax_sub = axes[fp_idx]
        if fp_name in sorted_fp_data:
            values = sorted_fp_data[fp_name]
            display_label = fp_name.replace("GCampCa-", "GCamp Ca-")
            ax_sub.bar(x_pos, values, width,
                       color=colors[fp_idx], alpha=0.3, edgecolor='black', linewidth=1, zorder=10)
            ax_sub.set_title(display_label, fontsize=int(54*0.75), fontweight='bold')
            ax_sub.set_ylim(0, max_y)
            ax_sub.set_yticks([0, 1])
            ax_sub.tick_params(axis='y', labelsize=int(30*0.75))
            ax_sub.grid(True, alpha=0.3, axis='y')
            ax_sub.spines["top"].set_visible(False)
            ax_sub.spines["right"].set_visible(False)
    
    # Bottom subplot: overlay (all fluorophores overlaid)
    ax_bot = axes[-1]
    for fp_idx, fp_name in enumerate(fluorophores):
        if fp_name in sorted_fp_data:
            values = sorted_fp_data[fp_name]
            display_label = fp_name.replace("GCampCa-", "GCamp Ca-")
            ax_bot.bar(x_pos, values, width, 
                       color=colors[fp_idx], alpha=0.3, edgecolor='black', linewidth=1, zorder=10)
    ax_bot.set_title("Overlay", fontsize=int(54*0.75), fontweight='bold')
    ax_bot.set_ylim(0, max_y)
    ax_bot.set_yticks([0, 1])
    ax_bot.set_xticks(x_pos)
    
    # Number the channels (every other for all channels mode, all for best channels mode)
    if label_every_other:
        numbered_labels = []
        for i in range(len(channel_labels)):
            if i % 2 == 0:  # Only label even indices (0, 2, 4, ...)
                numbered_labels.append(str(i+1))
            else:
                numbered_labels.append('')  # Empty string for odd indices
    else:
        numbered_labels = [str(i+1) for i in range(len(channel_labels))]
    
    ax_bot.set_xticklabels(numbered_labels, fontsize=int(42*0.75), rotation=0, ha='center')
    ax_bot.tick_params(axis='x', labelsize=int(42*0.75))
    ax_bot.tick_params(axis='y', labelsize=int(30*0.75))
    ax_bot.grid(True, alpha=0.3, axis='y')
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    
    # Set y-axis label on middle subplot (aligned to middle, 1/2 size of current)
    # Current is 3x, so 1/2 would be 1.5x
    middle_idx = len(axes) // 2
    axes[middle_idx].set_ylabel("Relative Signal", fontsize=int(36*0.75*1.5), labelpad=20)
    
    # Adjust x-axis limits to ensure bars are not wider than they are tall
    # Add padding on both sides: for N channels, add padding to make bars appear narrower
    # This ensures bars appear narrower when there are fewer channels (like Figure 2)
    # For Figure 5 with many channels, padding is minimal. For Figure 2 with few channels, more padding.
    n_channels = len(sorted_channel_keys)
    # Current x-axis: bars at positions 0 to n_channels-1, each with width 1.0
    # To make bars narrower visually, add padding: more padding for fewer channels
    # Scale padding inversely with number of channels (more padding when fewer channels)
    if n_channels <= 10:
        # For few channels (like Figure 2), add significant padding
        x_padding = max(1.0, (10 - n_channels) * 0.2)  # More padding for fewer channels
    else:
        # For many channels (like Figure 5), minimal padding
        x_padding = 0.5
    for ax_sub in axes:
        ax_sub.set_xlim(-x_padding, n_channels - 1 + x_padding)
    
    return fig, axes, channel_labels


def load_2p_spectra_flexible(fluorophore_name, spectra_dir=None, debug=False):
    """
    Load 2P excitation spectra for fluorophores with flexible column name matching.
    
    Handles case-insensitive column names, various naming variations, and special cases.
    This is a more flexible version than the basic load_2p_spectra from figure1.
    
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
    if spectra_dir is None:
        spectra_dir = os.path.join("dev_scripts", "demo_data")
    
    # Map fluorophore names to CSV filenames
    filename_map = {
        "EBFP": "ebfp.csv",
        "tagBFP": "tagbfp.csv",
        "ECFP": "ECFP.csv",
        "GCamp Ca+": "egfp.csv",  # GCamp Ca+ uses EGFP spectra
        "GCampCa-": "GCampCa-.csv",  # New CSV with tagBFP excitation and EGFP emission
        "LSSmOrange": "lssmOrange.csv",
        "TdTomato": "TdTomato.csv",
        "mCherry": "mCherry.csv",
        "LSSmKAte": "LSS-mKate1.csv",  # Special case: no 2P, will copy from mTFP1
        "mNeptune": "mNeptune.csv",
        "YFP": "eyfp.csv",  # eYFP spectra
        "mScarlet": "mScarlet.csv"  # mScarlet spectra
    }
    
    # Debug: Print what we're looking for
    if debug:
        print(f"DEBUG load_2p_spectra_flexible: Requested fluorophore: '{fluorophore_name}'")
        print(f"DEBUG: Available in filename_map: {list(filename_map.keys())}")
    
    # Try case-insensitive matching
    matched_key = None
    if fluorophore_name in filename_map:
        matched_key = fluorophore_name
    else:
        # Try case-insensitive match
        for key in filename_map.keys():
            if key.lower() == fluorophore_name.lower():
                matched_key = key
                if debug:
                    print(f"DEBUG: Matched '{fluorophore_name}' to '{key}' (case-insensitive)")
                break
    
    if matched_key is None:
        if debug:
            print(f"ERROR: 2P spectra not available for '{fluorophore_name}'")
            print(f"  Available fluorophores: {list(filename_map.keys())}")
        raise ValueError(f"2P spectra not available for {fluorophore_name}. Available: {list(filename_map.keys())}")
    
    csv_path = os.path.join(spectra_dir, filename_map[matched_key])
    if debug:
        print(f"DEBUG: Using CSV file: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"2P spectra file not found: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Special case: LSSmKate - copy 2P column from mTFP1
    if fluorophore_name == "LSSmKAte":
        mTFP1_path = os.path.join(spectra_dir, "mtfp1.csv")
        if os.path.exists(mTFP1_path):
            mTFP1_df = pd.read_csv(mTFP1_path)
            mTFP1_df.columns = mTFP1_df.columns.str.strip()
            # Find mTFP1 2p column (handle naming variations)
            mTFP1_2p_col = None
            for col in mTFP1_df.columns:
                if "mTFP1" in col and ("2p" in col.lower() or "2P" in col):
                    mTFP1_2p_col = col
                    break
            
            if mTFP1_2p_col:
                # Align by wavelength and copy
                merged = pd.merge(df, mTFP1_df[["wavelength", mTFP1_2p_col]], 
                                 on="wavelength", how="left")
                # Create LSSmKAte 2p column from mTFP1 2p
                df["LSSmKAte 2p"] = merged[mTFP1_2p_col]
                print(f"Copied 2P spectra from mTFP1 to LSSmKAte")
            else:
                print(f"Warning: Could not find mTFP1 2p column to copy for LSSmKAte")
        else:
            print(f"Warning: Could not find mtfp1.csv to copy 2P spectra for LSSmKAte")
    
    # Extract relevant columns - handle case and naming variations
    # Find wavelength column (case-insensitive)
    wavelength_col = None
    for col in df.columns:
        if col.lower() == "wavelength":
            wavelength_col = col
            break
    if wavelength_col is None:
        raise ValueError(f"Wavelength column not found in {csv_path}. Available columns: {list(df.columns)}")
    
    # Find excitation column (2P) - try multiple variations
    excitation_col = None
    fp_name_variations = [
        fluorophore_name,
        fluorophore_name.replace("LSSmKAte", "LSS-mKate1").replace("LSSmOrange", "lssmOrange"),
        fluorophore_name.upper(),
        fluorophore_name.lower(),
        fluorophore_name.replace("tag", "Tag").replace("BFP", "BFP"),
        fluorophore_name.replace("GCamp Ca+", "EGFP"),  # GCamp Ca+ uses EGFP spectra
        fluorophore_name.replace("GCampCa-", "GCampCa-")  # Keep as-is for GCampCa-
    ]
    
    for fp_var in fp_name_variations:
        possible_cols = [
            f"{fp_var} 2p", f"{fp_var} 2P", f"{fp_var}2p", f"{fp_var}2P",
            f"{fp_var} 2p", f"{fp_var} 2P"
        ]
        for col in possible_cols:
            if col in df.columns:
                excitation_col = col
                break
        if excitation_col:
            break
    
    # Special case: LSSmKate - check for "LSSmKAte 2p" column first (before generic search)
    if not excitation_col and fluorophore_name == "LSSmKAte":
        if "LSSmKAte 2p" in df.columns:
            excitation_col = "LSSmKAte 2p"
            print(f"DEBUG: Using 'LSSmKAte 2p' column for {fluorophore_name}")
        elif "LSS-mKate1 2p" in df.columns:
            excitation_col = "LSS-mKate1 2p"
            print(f"DEBUG: Using 'LSS-mKate1 2p' column for {fluorophore_name}")
    
    # If still not found, search for any column with "2p" or "2P"
    if not excitation_col:
        alt_names = [col for col in df.columns if "2p" in col.lower()]
        if alt_names:
            excitation_col = alt_names[0]
        else:
            # Special case: Some CSVs have generic "Excitation" column (e.g., CFP.csv)
            if "Excitation" in df.columns:
                excitation_col = "Excitation"
                print(f"DEBUG: Using generic 'Excitation' column for {fluorophore_name}")
            # Special case: GCamp Ca+ uses EGFP columns
            elif fluorophore_name == "GCamp Ca+" and "EGFP 2p" in df.columns:
                excitation_col = "EGFP 2p"
            # Special case: GCampCa- has "GCampCa- 2p" column
            elif fluorophore_name == "GCampCa-":
                for col in df.columns:
                    if "GCampCa" in col and ("2p" in col.lower() or "2P" in col):
                        excitation_col = col
                        break
                if not excitation_col:
                    raise ValueError(f"2P excitation column not found for {fluorophore_name} in {csv_path}. Available columns: {list(df.columns)}")
            else:
                raise ValueError(f"2P excitation column not found for {fluorophore_name} in {csv_path}. Available columns: {list(df.columns)}")
    
    # Find emission column - try multiple variations
    emission_col = None
    for fp_var in fp_name_variations:
        possible_cols = [
            f"{fp_var} em", f"{fp_var} Em", f"{fp_var}em", f"{fp_var}Em",
            f"{fp_var} emission", f"{fp_var} Emission"
        ]
        for col in possible_cols:
            if col in df.columns:
                emission_col = col
                break
        if emission_col:
            break
    
    # Special case: GCampCa- might have "GCampCa- em" column
    if not emission_col and fluorophore_name == "GCampCa-":
        for col in df.columns:
            if "GCampCa" in col and ("em" in col.lower() or "emission" in col.lower()):
                emission_col = col
                break
    
    # Special case: GCamp Ca+ uses EGFP columns
    if not emission_col and fluorophore_name == "GCamp Ca+":
        if "EGFP em" in df.columns:
            emission_col = "EGFP em"
    
    # If still not found, search for any column with "em" or "emission"
    if not emission_col:
        alt_names = [col for col in df.columns if ("em" in col.lower() and "emission" not in col.lower()) or "emission" in col.lower()]
        if alt_names:
            emission_col = alt_names[0]
        else:
            # Special case: Some CSVs have generic "Emission" column (e.g., CFP.csv)
            if "Emission" in df.columns:
                emission_col = "Emission"
                print(f"DEBUG: Using generic 'Emission' column for {fluorophore_name}")
            else:
                raise ValueError(f"Emission column not found for {fluorophore_name} in {csv_path}. Available columns: {list(df.columns)}")
    
    # Create standardized DataFrame
    spectra_df = pd.DataFrame({
        "Wavelength": df[wavelength_col],
        "Excitation": df[excitation_col],
        "Emission": df[emission_col]
    })
    
    # Fill NaN values with 0
    spectra_df = spectra_df.fillna(0)
    
    return spectra_df

