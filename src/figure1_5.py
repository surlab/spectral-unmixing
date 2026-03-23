"""
Figure 1.5 generation for spectral unmixing methods paper.

This module generates Figure 1.5, which is based on the dual domain scatterplot
from Figure 1 but with added labels and arrows to illustrate key concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc, ConnectionPatch
from src import config as cfg
from src.figure_scatterplot_helpers import (
    compute_data_vector,
    vector_angle,
    filter_by_distance,
    bin_and_subsample_by_distance,
    compute_actual_variance_perpendicular
)
from src.figure1 import (
    load_channel_data,
    compute_predicted_channel_signals,
    _get_all_acquisition_pairs
)

# Dual domain row configuration (same as Row3_dict from figure1.py)
DUAL_DOMAIN_ROW = {
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


def create_main_figure(data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619", 
                       ax=None):
    """
    Create the main Figure 1.5 scatterplot with labeled arrows.
    
    Based on dual_domain_subpanel5 but:
    - Removes classification zones
    - Removes predicted dashed vector
    - Keeps data vector (solid arrow)
    - Adds 5 labeled arrows explaining key concepts
    
    Parameters
    ----------
    data_dir : str
        Path to data directory
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (if None, creates new figure)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    row_dict = DUAL_DOMAIN_ROW  # dual domain
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
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
    
    # Ensure arrays are 1D
    if len(ch1_combined.shape) > 1:
        ch1_combined = ch1_combined.flatten()
    if len(ch2_combined.shape) > 1:
        ch2_combined = ch2_combined.flatten()
    
    # Filter by distance (same as subpanel_5)
    max_value = 3000
    max_distance = max_value * np.sqrt(2)
    
    ch1_filtered, ch2_filtered, fp_labels_filtered, distances = filter_by_distance(
        ch1_combined, ch2_combined, labels=fp_labels, 
        max_distance=max_distance, min_distance=0
    )
    
    # Subsample using distance-based binning
    ch1_plot, ch2_plot, fp_labels_plot = bin_and_subsample_by_distance(
        ch1_filtered, ch2_filtered, fp_labels_filtered,
        bin_width=100, samples_per_bin=300, max_distance=max_distance
    )
    
    # Set axis limits
    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)
    
    # Compute data vectors
    data_vectors = {}
    for i, fp_name in enumerate(fluorophores):
        data_vec = compute_data_vector(all_ch1_data[i], all_ch2_data[i])
        data_vectors[fp_name] = data_vec
    
    # Scale vectors to reach 70th percentile, then make arrows 2x as long
    ch1_70th = np.percentile(ch1_plot, cfg.vector_scaling_percentile)
    ch2_70th = np.percentile(ch2_plot, cfg.vector_scaling_percentile)
    max_scale = max(ch1_70th, ch2_70th) * 2.0
    
    # Plot scatter points
    fp_colors = {fp: cfg.fluorophore_colors.get(fp, "#000000") for fp in fluorophores}
    colors_list = [fp_colors[label] for label in fp_labels_plot]
    ax.scatter(ch1_plot, ch2_plot, c=colors_list, alpha=0.4, s=2, zorder=2)
    
    # Plot data vectors (solid arrows only, no predicted)
    # Use smaller mutation scale for data vectors (they're thinner)
    data_arrow_mutation_scale = 18
    for fp_name in fluorophores:
        color = fp_colors[fp_name]
        data_vec = data_vectors[fp_name]
        data_end_x = data_vec[0] * max_scale
        data_end_y = data_vec[1] * max_scale
        data_arrow = FancyArrowPatch((0, 0), (data_end_x, data_end_y),
                                     arrowstyle='->', mutation_scale=data_arrow_mutation_scale,
                                     linestyle='-', linewidth=2, color=color, alpha=0.7)
        ax.add_patch(data_arrow)
    
    # Compute angle between vectors for label
    if len(fluorophores) == 2:
        vec1 = data_vectors[fluorophores[0]]
        vec2 = data_vectors[fluorophores[1]]
        angle_rad = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        vec1_angle_deg = np.degrees(np.arctan2(vec1[1], vec1[0]))
        vec2_angle_deg = np.degrees(np.arctan2(vec2[1], vec2[0]))
        
        # Draw arrow between vectors (instead of arc)
        # Draw thick arc and add arrowhead at the end
        arc_radius = max_scale * 0.5
        mid_angle_deg = (vec1_angle_deg + vec2_angle_deg) / 2
        
        # Calculate arrowhead positions first
        # Convert angles to radians first
        start_angle_rad = np.radians(vec1_angle_deg)
        end_angle_rad = np.radians(vec2_angle_deg)
        
        # Use a larger offset to make the arc noticeably shorter
        # Calculate offset as a fraction of the arc span
        total_angle_rad = abs(end_angle_rad - start_angle_rad)
        # Make arc 50% of original length (25% offset on each end)
        offset_fraction = 0.25
        arrowhead_offset_rad = total_angle_rad * offset_fraction
        
        # End arrowhead (at vec2) - arrow tip at exact end point
        before_end_angle_rad = end_angle_rad - arrowhead_offset_rad
        before_end_x = arc_radius * np.cos(before_end_angle_rad)
        before_end_y = arc_radius * np.sin(before_end_angle_rad)
        end_x = arc_radius * np.cos(end_angle_rad)
        end_y = arc_radius * np.sin(end_angle_rad)
        
        # Start arrowhead (at vec1) - arrow tip at exact start point
        after_start_angle_rad = start_angle_rad + arrowhead_offset_rad
        after_start_x = arc_radius * np.cos(after_start_angle_rad)
        after_start_y = arc_radius * np.sin(after_start_angle_rad)
        start_x = arc_radius * np.cos(start_angle_rad)
        start_y = arc_radius * np.sin(start_angle_rad)
        
        # Draw the arc shorter - from arrowhead base positions, keeping it centered
        arc_theta1_deg = np.degrees(after_start_angle_rad)
        arc_theta2_deg = np.degrees(before_end_angle_rad)
        arc = Arc((0, 0), arc_radius * 2, arc_radius * 2, 
                 angle=0, theta1=arc_theta1_deg, theta2=arc_theta2_deg, 
                 color='black', linewidth=cfg.thick_linewidth, zorder=3)
        ax.add_patch(arc)
        
        # Add arrowheads at the ends
        end_arrowhead = FancyArrowPatch((before_end_x, before_end_y),
                                        (end_x, end_y),
                                        arrowstyle='->', mutation_scale=cfg.arrow_mutation_scale,
                                        linewidth=cfg.thick_linewidth, color='black', zorder=3)
        ax.add_patch(end_arrowhead)
        
        start_arrowhead = FancyArrowPatch((after_start_x, after_start_y),
                                          (start_x, start_y),
                                          arrowstyle='->', mutation_scale=cfg.arrow_mutation_scale,
                                          linewidth=cfg.thick_linewidth, color='black', zorder=3)
        ax.add_patch(start_arrowhead)
        
        # Label angle (updated text) - positioned at (2450, 800), moved 100 right
        ax.text(2450, 800, 'angle of separation\n(emission filter and\nexcitation wavelengths)', 
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labeled arrows
    _add_labeled_arrows(ax, data_vectors, max_scale, max_value, fluorophores, fp_colors, cfg.thick_linewidth, cfg.arrow_mutation_scale)
    
    # Set labels
    ax.set_xlabel("Channel 1 Signal", fontsize=12)
    ax.set_ylabel("Channel 2 Signal", fontsize=12)
    
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Square aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    return fig, ax


def _add_labeled_arrows(ax, data_vectors, max_scale, max_value, fluorophores, fp_colors, thick_linewidth, arrow_mutation_scale):
    """
    Add 5 labeled arrows to the plot.
    
    1. Y-axis scaling arrow (Channel 2 scaling)
    2. mNeptune vector scaling arrow (pixel brightness scaling)
    3. Perpendicular variance arrow (variance around mean angle)
    4. Noise arrows near origin (5 arrows)
    
    Parameters
    ----------
    thick_linewidth : float
        Line width for thick arrows
    arrow_mutation_scale : float
        Scale factor for arrowhead size
    """
    
    # 1. Y-axis scaling arrow (Channel 2 scaling)
    # Moved to the right, X=200, from ~1500 to ~3000
    y_arrow_start = (200, 1500)
    y_arrow_end = (200, 3000)
    y_arrow = FancyArrowPatch(y_arrow_start, y_arrow_end,
                              arrowstyle='<->', mutation_scale=arrow_mutation_scale,
                              linewidth=thick_linewidth, color='black', zorder=3)
    ax.add_patch(y_arrow)
    ax.text(300, 2250, 'scaling channel 2\n(laser power, PMT amplification,\nfilter collection efficiency)',
           ha='left', va='center', fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. mNeptune vector scaling arrow (pixel brightness scaling)
    # Longer arrow, text centered on (750, 1250) and not overlapping arrow
    if 'mNeptune' in data_vectors:
        neptune_vec = data_vectors['mNeptune']
        # Position along mNeptune vector near (1500, 1500)
        center_dist = 1500
        center_x = neptune_vec[0] * center_dist
        center_y = neptune_vec[1] * center_dist
        
        # Arrow along vector direction, longer (~1100)
        arrow_length = 1100
        arrow_start_x = center_x - neptune_vec[0] * arrow_length / 2
        arrow_start_y = center_y - neptune_vec[1] * arrow_length / 2
        arrow_end_x = center_x + neptune_vec[0] * arrow_length / 2
        arrow_end_y = center_y + neptune_vec[1] * arrow_length / 2
        
        neptune_arrow = FancyArrowPatch((arrow_start_x, arrow_start_y), 
                                        (arrow_end_x, arrow_end_y),
                                        arrowstyle='<->', mutation_scale=arrow_mutation_scale,
                                        linewidth=thick_linewidth, color='black', zorder=3)
        ax.add_patch(neptune_arrow)
        # Text centered on (650, 1150) - positioned away from arrow, moved 100 left and 100 down
        ax.text(650, 1150, 'scaling pixel brightness\n(FP concentration, FP brightness,\nobjective collection efficiency,\nnet dwell, ROI size)',
               ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Perpendicular variance arrow
    # Longer arrow, text centered on (2500, 1700) and not overlapping arrow
    if 'mNeptune' in data_vectors:
        neptune_vec = data_vectors['mNeptune']
        # Position further out
        base_dist = 2500
        base_x = neptune_vec[0] * base_dist
        base_y = neptune_vec[1] * base_dist
        
        # Perpendicular direction
        perp_vec = np.array([-neptune_vec[1], neptune_vec[0]])
        # Longer arrow (~1100)
        arrow_length = 1100
        arrow_start_x = base_x - perp_vec[0] * arrow_length / 2
        arrow_start_y = base_y - perp_vec[1] * arrow_length / 2
        arrow_end_x = base_x + perp_vec[0] * arrow_length / 2
        arrow_end_y = base_y + perp_vec[1] * arrow_length / 2
        
        perp_arrow = FancyArrowPatch((arrow_start_x, arrow_start_y),
                                     (arrow_end_x, arrow_end_y),
                                     arrowstyle='<->', mutation_scale=arrow_mutation_scale,
                                     linewidth=thick_linewidth, color='black', zorder=3)
        ax.add_patch(perp_arrow)
        # Text centered on (2600, 1700) - positioned away from arrow, moved 100 right
        ax.text(2600, 1700, 'variance around mean angle\n(amplitude, proximity to 45 degrees)',
               ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Noise arrows near origin (3 arrows with different lengths and directions)
    # Thinner arrows (1/3rd of thick_linewidth)
    # Start at (500, 500) to avoid clipping by axes
    noise_linewidth = thick_linewidth / 3.0  # 1/3rd thickness
    noise_start_x, noise_start_y = 500, 500
    
    # Define 3 noise arrows with clearly different angles and lengths
    # Format: (angle_deg, length)
    noise_arrow_specs = [
        (45, 300),    # Northeast, medium length
        (200, 450),   # Southwest, longer
        (320, 200),   # Northwest, shorter
    ]
    
    for angle_deg, length in noise_arrow_specs:
        angle_rad = np.radians(angle_deg)
        arrow_end_x = noise_start_x + length * np.cos(angle_rad)
        arrow_end_y = noise_start_y + length * np.sin(angle_rad)
        
        # Thinner arrows for noise (1/3rd thickness), black color
        noise_arrow = FancyArrowPatch((noise_start_x, noise_start_y), (arrow_end_x, arrow_end_y),
                                      arrowstyle='->', mutation_scale=15,
                                      linewidth=noise_linewidth, color='black', 
                                      alpha=0.7, zorder=3)
        ax.add_patch(noise_arrow)
    
    # Single label for all noise arrows, centered on (700, 200)
    ax.text(700, 200, 'detector noise,\nbackground and dark noise',
           ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def save_main_figure(output_dir="results/Figure1_5"):
    """
    Generate and save the main Figure 1.5.
    
    Parameters
    ----------
    output_dir : str
        Directory to save the figure
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = create_main_figure()
    filepath = os.path.join(output_dir, "figure1_5_main.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def save_all_supplement_panels(fig2_data_dir="data/fig2_3color_inh_spatial_control_2p3_10072025",
                               fig1_data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619",
                               output_dir="results/Figure1_5"):
    """
    Generate and save all supplement panels for Figure 1.5.
    
    Parameters
    ----------
    fig2_data_dir : str
        Path to fig2 data directory
    fig1_data_dir : str
        Path to fig1 data directory
    output_dir : str
        Directory to save figures
    """
    print("\n" + "="*60)
    print("Generating Figure 1.5 Supplement Panels")
    print("="*60)
    
    # 1.5a: Power comparison
    print("\n1.5a: Power comparison scatterplots...")
    try:
        data_1_5a = subpanel_1_5a(fig2_data_dir, output_dir)
        # 1.5ai: Angle histograms for 1.5a
        print("\n1.5ai: Angle histograms for 1.5a...")
        try:
            subpanel_1_5ai(data_1_5a, fig2_data_dir, output_dir)
        except Exception as e:
            print(f"  Error in 1.5ai: {e}")
    except Exception as e:
        print(f"  Error in 1.5a: {e}")
    
    # 1.5b: Variance vs angle
    print("\n1.5b: Variance vs angle scatterplot...")
    try:
        subpanel_1_5b(fig1_data_dir, output_dir)
    except Exception as e:
        print(f"  Error in 1.5b: {e}")
    
    # 1.5c and 1.5d: Variance vs distance
    print("\n1.5c & 1.5d: Variance vs distance line plots...")
    try:
        subpanel_1_5c_1_5d(fig1_data_dir, output_dir)
    except Exception as e:
        print(f"  Error in 1.5c/1.5d: {e}")
    
    # 1.5g: Example scatterplots
    print("\n1.5g: Example scatterplots...")
    try:
        subpanel_1_5g(fig1_data_dir, output_dir)
    except Exception as e:
        print(f"  Error in 1.5g: {e}")
    
    # 1.5h: Cross-wavelength comparison
    print("\n1.5h: Cross-wavelength comparison...")
    try:
        subpanel_1_5h(fig1_data_dir, output_dir)
    except Exception as e:
        print(f"  Error in 1.5h: {e}")
    
    # 1.5z: mCherry 1240nm Broad vs FarRed
    print("\n1.5z: mCherry 1240nm Broad vs FarRed scatterplot...")
    try:
        subpanel_1_5z(fig1_data_dir, output_dir)
    except Exception as e:
        print(f"  Error in 1.5z: {e}")
    
    print("\n" + "="*60)
    print("Supplement panels generation complete!")
    print("="*60)


# ============================================================================
# Supplement Panels
# ============================================================================

def _find_fig2_acquisitions(data_dir="data/fig2_3color_inh_spatial_control_2p3_10072025"):
    """
    Find all valid acquisitions from fig2 data directory.
    
    Returns list of dicts with keys: filter, excitation_wl, pockels, filepath
    """
    import os
    import glob
    
    acquisitions = []
    
    if not os.path.exists(data_dir):
        print(f"Warning: fig2 data directory not found: {data_dir}")
        return acquisitions
    
    # Pattern: Filter_ExcitationWavelength_PockelsValue_PMT.tif
    # e.g., BR2EmFilt_1040nm_185poc_600pmt.tif
    pattern = os.path.join(data_dir, "*EmFilt_*nm_*poc*.tif")
    files = glob.glob(pattern)
    
    for filepath in files:
        basename = os.path.basename(filepath)
        # Parse: Filter_ExcitationWavelength_PockelsValue_PMT.tif
        parts = basename.replace('EmFilt_', '_').replace('nm_', '_').replace('poc_', '_').replace('pmt.tif', '').split('_')
        
        if len(parts) >= 3:
            filter_name = parts[0]  # e.g., BR2, Red, FarRed
            try:
                excitation_wl = int(parts[1])  # e.g., 1040
                pockels = int(parts[2])  # e.g., 185
                acquisitions.append({
                    'filter': filter_name,
                    'excitation_wl': excitation_wl,
                    'pockels': pockels,
                    'filepath': filepath
                })
            except (ValueError, IndexError):
                continue
    
    return acquisitions


def subpanel_1_5a(data_dir="data/fig2_3color_inh_spatial_control_2p3_10072025", 
                  output_dir="results/Figure1_5"):
    """
    Subpanel 1.5a: Power comparison scatterplots.
    
    First subplot: BR2EmFilt_1080nm_221 vs FarRedEmFilt_1240nm_465
    Second subplot: BR2EmFilt_1080nm_221 vs FarRedEmFilt_1240nm_594
    """
    import os
    from skimage import io
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Hardcode exact filenames (matching actual files in directory)
    ch1_filename = "BR2EmFilt_1080nm_221poc_600pmt.tif"
    ch2_filenames = [
        "FarRedEmFilt_1240nm_465poc.tif",
        "FarRedEmFilt_1240nm_594poc.tif"
    ]
    
    # Construct full paths
    ch1_path = os.path.join(data_dir, ch1_filename)
    ch2_paths = [os.path.join(data_dir, fn) for fn in ch2_filenames]
    
    # Check if files exist
    if not os.path.exists(ch1_path):
        print(f"Error: Ch1 file not found: {ch1_path}")
        return
    
    for ch2_path in ch2_paths:
        if not os.path.exists(ch2_path):
            print(f"Error: Ch2 file not found: {ch2_path}")
            return
    
    print(f"Loading Ch1: {ch1_filename}")
    print(f"Loading Ch2: {ch2_filenames}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for idx, ch2_path in enumerate(ch2_paths):
        ax = axes[idx]
        
        try:
            # Load images
            ch1_img = io.imread(ch1_path)
            ch2_img = io.imread(ch2_path)
            
            # Handle stacks: if 3D with shape (z, height, width), take max projection
            if len(ch1_img.shape) == 3:
                if ch1_img.shape[0] < ch1_img.shape[2]:  # Likely (z, height, width) stack
                    ch1_img = np.max(ch1_img, axis=0)  # Max projection across z
                else:  # Likely (height, width, channels)
                    ch1_img = ch1_img[:, :, 0]  # Take first channel
            elif len(ch1_img.shape) == 2:
                pass  # Already 2D
            
            if len(ch2_img.shape) == 3:
                if ch2_img.shape[0] < ch2_img.shape[2]:  # Likely (z, height, width) stack
                    ch2_img = np.max(ch2_img, axis=0)  # Max projection across z
                else:  # Likely (height, width, channels)
                    ch2_img = ch2_img[:, :, 0]  # Take first channel
            elif len(ch2_img.shape) == 2:
                pass  # Already 2D
            
            # Flatten images
            ch1_flat = ch1_img.flatten()
            ch2_flat = ch2_img.flatten()
            
            # Ensure same length
            min_len = min(len(ch1_flat), len(ch2_flat))
            ch1_flat = ch1_flat[:min_len]
            ch2_flat = ch2_flat[:min_len]
            
            # Use same subsampling as other scatterplots
            max_value = 3000
            ch1_filtered, ch2_filtered, _, _ = filter_by_distance(
                ch1_flat, ch2_flat, max_distance=max_value * np.sqrt(2)
            )
            
            # Create dummy labels (all same since fig2 has all FPs in same image)
            dummy_labels = np.array(['all'] * len(ch1_filtered), dtype=object)
            
            # Use bin_and_subsample_by_distance (same as other scatterplots)
            ch1_plot, ch2_plot, _ = bin_and_subsample_by_distance(
                ch1_filtered, ch2_filtered, dummy_labels,
                bin_width=100, samples_per_bin=300, max_distance=max_value * np.sqrt(2)
            )
            
            # Plot with same style as other scatterplots
            ax.scatter(ch1_plot, ch2_plot, alpha=0.4, s=2, c='gray')
            ax.set_xlabel("Channel 1 (1080nm BR2, 221poc)", fontsize=10)
            ch2_poc = ch2_filenames[idx].split('_')[2].replace('poc', '')
            ax.set_ylabel(f"Channel 2 (1240nm FarRed, {ch2_poc}poc)", fontsize=10)
            ax.set_xlim(0, max_value)
            ax.set_ylim(0, max_value)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f"Power: {ch2_poc}poc", fontsize=10)
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, f"Error loading data", ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "subpanel_1_5a_power_comparison.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")
    
    # Return data for histogram (1.5ai)
    return {
        'ch1_path': ch1_path,
        'ch2_paths': ch2_paths,
        'ch2_filenames': ch2_filenames
    }
    
    # Return data for histogram (1.5ai)
    return {
        'ch1_path': ch1_path,
        'ch2_paths': ch2_paths,
        'ch2_filenames': ch2_filenames
    }


def subpanel_1_5b(data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619",
                  output_dir="results/Figure1_5", target_distance=1250):
    """
    Subpanel 1.5b: Variance vs angle scatterplot.
    
    Variance around mean vs angle (FP angle, not separation angle) for many pairs,
    colored by the FP, variance taken at a consistent distance from the origin.
    """
    import os
    from src.figure_scatterplot_helpers import compute_actual_variance_perpendicular
    from src.figure1 import _get_all_acquisition_pairs
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all acquisition pairs - include all fluorophores (don't exclude tdTomato)
    print("Finding all acquisition pairs (including all fluorophores)...")
    all_pairs = _get_all_acquisition_pairs(data_dir, exclude_fluorophores=[], avoid_bidirectional=True)
    print(f"Found {len(all_pairs)} valid pairs")
    
    # Compute variance and angle for each pair
    # Process both fp1 and fp2 to get data for all fluorophores
    variances = []
    angles = []
    fp_names = []
    pair_info_list = []  # Store pair information for labeling outliers
    
    print(f"Computing variance at distance {target_distance} for {len(all_pairs)} pairs...")
    brightness_threshold = 500
    min_pixels_above_threshold = 100  # Minimum number of pixels above threshold
    
    for pair_idx, pair in enumerate(all_pairs):
        if (pair_idx + 1) % 50 == 0:
            print(f"  Processed {pair_idx + 1}/{len(all_pairs)} pairs...")
        
        # Check brightness for fp1
        fp1_ch1_data = pair['fp1_ch1']
        fp1_ch2_data = pair['fp1_ch2']
        fp1_ch1_bright = np.sum(fp1_ch1_data > brightness_threshold)
        fp1_ch2_bright = np.sum(fp1_ch2_data > brightness_threshold)
        
        # Skip if neither channel has enough bright pixels
        if fp1_ch1_bright < min_pixels_above_threshold and fp1_ch2_bright < min_pixels_above_threshold:
            continue
        
        # Process fp1
        vector = compute_data_vector(fp1_ch1_data, fp1_ch2_data)
        
        variance, _ = compute_actual_variance_perpendicular(
            fp1_ch1_data, fp1_ch2_data, vector, target_distance=target_distance
        )
        
        if not np.isnan(variance) and variance > 0:
            angle = vector_angle(vector)
            # Convert to angle to nearest axis: min(angle, 90-angle)
            # First map angle to 0-90 range
            angle_0_90 = angle % 180
            if angle_0_90 > 90:
                angle_0_90 = 180 - angle_0_90
            angle_to_nearest_axis = min(angle_0_90, 90 - angle_0_90)
            variances.append(variance)
            angles.append(angle_to_nearest_axis)
            fp_names.append(pair['fp1'])
            # Store pair info for labeling
            pair_info_list.append({
                'fp': pair['fp1'],
                'ch1_wl': pair['ch1_wl'],
                'ch1_filter': pair['ch1_filter'],
                'ch2_wl': pair['ch2_wl'],
                'ch2_filter': pair['ch2_filter'],
                'ch1_pockels': pair.get('ch1_pockels'),
                'ch2_pockels': pair.get('ch2_pockels')
            })
        
        # Check brightness for fp2
        fp2_ch1_data = pair['fp2_ch1']
        fp2_ch2_data = pair['fp2_ch2']
        fp2_ch1_bright = np.sum(fp2_ch1_data > brightness_threshold)
        fp2_ch2_bright = np.sum(fp2_ch2_data > brightness_threshold)
        
        # Skip if neither channel has enough bright pixels
        if fp2_ch1_bright < min_pixels_above_threshold and fp2_ch2_bright < min_pixels_above_threshold:
            continue
        
        # Process fp2 (to include all fluorophores like mNeptune and tdTomato)
        vector = compute_data_vector(fp2_ch1_data, fp2_ch2_data)
        
        variance, _ = compute_actual_variance_perpendicular(
            fp2_ch1_data, fp2_ch2_data, vector, target_distance=target_distance
        )
        
        if not np.isnan(variance) and variance > 0:
            angle = vector_angle(vector)
            # Convert to angle to nearest axis: min(angle, 90-angle)
            angle_0_90 = angle % 180
            if angle_0_90 > 90:
                angle_0_90 = 180 - angle_0_90
            angle_to_nearest_axis = min(angle_0_90, 90 - angle_0_90)
            variances.append(variance)
            angles.append(angle_to_nearest_axis)
            fp_names.append(pair['fp2'])
            # Store pair info for labeling
            pair_info_list.append({
                'fp': pair['fp2'],
                'ch1_wl': pair['ch1_wl'],
                'ch1_filter': pair['ch1_filter'],
                'ch2_wl': pair['ch2_wl'],
                'ch2_filter': pair['ch2_filter'],
                'ch1_pockels': pair.get('ch1_pockels'),
                'ch2_pockels': pair.get('ch2_pockels')
            })
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color by FP
    fp_colors = {fp: cfg.fluorophore_colors.get(fp, "#000000") for fp in set(fp_names)}
    for fp in set(fp_names):
        mask = np.array(fp_names) == fp
        ax.scatter(np.array(angles)[mask], np.array(variances)[mask], 
                  c=fp_colors[fp], label=fp, alpha=0.6, s=30)
    
    # Find and label outliers
    # mCherry at (42, 85000) and tdTomato at (42, 0)
    angles_array = np.array(angles)
    variances_array = np.array(variances)
    
    # Find mCherry outlier (angle ~42, variance ~85000)
    mcherry_mask = np.array(fp_names) == 'mCherry'
    if np.any(mcherry_mask):
        mcherry_angles = angles_array[mcherry_mask]
        mcherry_variances = variances_array[mcherry_mask]
        # Find point closest to (42, 85000)
        mcherry_distances = np.sqrt((mcherry_angles - 42)**2 + ((mcherry_variances - 85000) / 1000)**2)
        mcherry_outlier_idx = np.argmin(mcherry_distances)
        if mcherry_distances[mcherry_outlier_idx] < 5:  # Within reasonable distance
            mcherry_outlier_angle = mcherry_angles[mcherry_outlier_idx]
            mcherry_outlier_variance = mcherry_variances[mcherry_outlier_idx]
            # Find corresponding pair info
            mcherry_indices = np.where(mcherry_mask)[0]
            mcherry_pair_info = pair_info_list[mcherry_indices[mcherry_outlier_idx]]
            # Create label
            ch1_poc = mcherry_pair_info['ch1_pockels']
            ch2_poc = mcherry_pair_info['ch2_pockels']
            label = f"Ch1: {mcherry_pair_info['ch1_wl']}nm {mcherry_pair_info['ch1_filter']}"
            if ch1_poc is not None:
                label += f" {ch1_poc}poc"
            label += f"\nCh2: {mcherry_pair_info['ch2_wl']}nm {mcherry_pair_info['ch2_filter']}"
            if ch2_poc is not None:
                label += f" {ch2_poc}poc"
            ax.annotate(label, 
                       xy=(mcherry_outlier_angle, mcherry_outlier_variance),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       fontsize=8)
    
    # Find tdTomato outlier (angle ~42, variance ~0)
    tdtomato_mask = np.array(fp_names) == 'tdTomato'
    if np.any(tdtomato_mask):
        tdtomato_angles = angles_array[tdtomato_mask]
        tdtomato_variances = variances_array[tdtomato_mask]
        # Find point closest to (42, 0)
        tdtomato_distances = np.sqrt((tdtomato_angles - 42)**2 + (tdtomato_variances / 1000)**2)
        tdtomato_outlier_idx = np.argmin(tdtomato_distances)
        if tdtomato_distances[tdtomato_outlier_idx] < 5:  # Within reasonable distance
            tdtomato_outlier_angle = tdtomato_angles[tdtomato_outlier_idx]
            tdtomato_outlier_variance = tdtomato_variances[tdtomato_outlier_idx]
            # Find corresponding pair info
            tdtomato_indices = np.where(tdtomato_mask)[0]
            tdtomato_pair_info = pair_info_list[tdtomato_indices[tdtomato_outlier_idx]]
            # Create label
            ch1_poc = tdtomato_pair_info['ch1_pockels']
            ch2_poc = tdtomato_pair_info['ch2_pockels']
            label = f"Ch1: {tdtomato_pair_info['ch1_wl']}nm {tdtomato_pair_info['ch1_filter']}"
            if ch1_poc is not None:
                label += f" {ch1_poc}poc"
            label += f"\nCh2: {tdtomato_pair_info['ch2_wl']}nm {tdtomato_pair_info['ch2_filter']}"
            if ch2_poc is not None:
                label += f" {ch2_poc}poc"
            ax.annotate(label, 
                       xy=(tdtomato_outlier_angle, tdtomato_outlier_variance),
                       xytext=(10, -30), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       fontsize=8)
    
    ax.set_xlabel("Angle to Nearest Axis (degrees)", fontsize=12)
    ax.set_ylabel(f"Variance Perpendicular (at distance {target_distance})", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "subpanel_1_5b_variance_vs_angle.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def subpanel_1_5c_1_5d(data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619",
                       output_dir="results/Figure1_5", distances=None):
    """
    Subpanels 1.5c and 1.5d: Variance vs distance line plots.
    
    1.5c: Raw variance
    1.5d: Normalized to its own mean variance
    """
    import os
    from src.figure_scatterplot_helpers import compute_actual_variance_perpendicular
    from src.figure1 import _get_all_acquisition_pairs
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate distances every 250 pixels if not provided
    if distances is None:
        # Generate from 250 to 3000 in steps of 250
        distances = list(range(250, 3001, 250))
    
    # Get all acquisition pairs - include all fluorophores
    print("Finding all acquisition pairs (including all fluorophores)...")
    all_pairs = _get_all_acquisition_pairs(data_dir, exclude_fluorophores=[], avoid_bidirectional=True)
    print(f"Found {len(all_pairs)} valid pairs")
    print(f"Computing variance at {len(distances)} distances: {distances}")
    
    # Compute variance at multiple distances for each pair
    # Store by FP name so we can group and compute mean
    fp_variances = {}  # key: fp_name, value: list of lists of (distance, variance) for each pair
    
    for pair in all_pairs:
        # Process fp1
        ch1_data = pair['fp1_ch1']
        ch2_data = pair['fp1_ch2']
        vector = compute_data_vector(ch1_data, ch2_data)
        
        # Compute variance at each distance
        variances_at_distances = []
        for dist in distances:
            variance, _ = compute_actual_variance_perpendicular(
                ch1_data, ch2_data, vector, target_distance=dist
            )
            if not np.isnan(variance):
                variances_at_distances.append((dist, variance))
        
        if len(variances_at_distances) > 0:
            fp_name = pair['fp1']
            if fp_name not in fp_variances:
                fp_variances[fp_name] = []
            fp_variances[fp_name].append(variances_at_distances)
        
        # Process fp2 (to include all fluorophores like mNeptune and tdTomato)
        ch1_data = pair['fp2_ch1']
        ch2_data = pair['fp2_ch2']
        vector = compute_data_vector(ch1_data, ch2_data)
        
        # Compute variance at each distance
        variances_at_distances = []
        for dist in distances:
            variance, _ = compute_actual_variance_perpendicular(
                ch1_data, ch2_data, vector, target_distance=dist
            )
            if not np.isnan(variance):
                variances_at_distances.append((dist, variance))
        
        if len(variances_at_distances) > 0:
            fp_name = pair['fp2']
            if fp_name not in fp_variances:
                fp_variances[fp_name] = []
            fp_variances[fp_name].append(variances_at_distances)
    
    # Plot 1.5c: Raw variance
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for fp_name, var_data_list in fp_variances.items():
        color = cfg.fluorophore_colors.get(fp_name, "#000000")
        # Plot each pair's line
        for idx, var_data in enumerate(var_data_list):
            dists = [d for d, v in var_data]
            vars_raw = [v for d, v in var_data]
            # Only label the first line for each FP
            label = fp_name if idx == 0 else ""
            ax.plot(dists, vars_raw, 'o-', color=color, alpha=0.5, markersize=4, label=label)
    
    ax.set_xlabel("Distance from Origin", fontsize=12)
    ax.set_ylabel("Variance Perpendicular (raw)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "subpanel_1_5c_variance_vs_distance_raw.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")
    
    # Plot 1.5d: Normalized variance
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Collect all normalized variances for computing mean
    all_normalized_by_distance = {}  # key: distance, value: list of normalized variances
    
    for fp_name, var_data_list in fp_variances.items():
        color = cfg.fluorophore_colors.get(fp_name, "#000000")
        # Plot each pair's line
        for idx, var_data in enumerate(var_data_list):
            dists = [d for d, v in var_data]
            vars_raw = [v for d, v in var_data]
            # Normalize to mean variance for this line
            mean_var = np.mean(vars_raw) if len(vars_raw) > 0 else 1.0
            vars_norm = [v / mean_var for v in vars_raw]
            # Only label the first line for each FP
            label = fp_name if idx == 0 else ""
            ax.plot(dists, vars_norm, 'o-', color=color, alpha=0.5, markersize=4, label=label)
            
            # Collect for mean calculation
            for dist, var_norm in zip(dists, vars_norm):
                if dist not in all_normalized_by_distance:
                    all_normalized_by_distance[dist] = []
                all_normalized_by_distance[dist].append(var_norm)
    
    # Compute and plot mean line (thicker, gray, on top)
    if len(all_normalized_by_distance) > 0:
        mean_dists = sorted(all_normalized_by_distance.keys())
        mean_vars = [np.mean(all_normalized_by_distance[d]) for d in mean_dists]
        ax.plot(mean_dists, mean_vars, '-', color='gray', linewidth=5, label='Mean', zorder=10)
    
    ax.set_xlabel("Distance from Origin", fontsize=12)
    ax.set_ylabel("Variance Perpendicular (normalized to mean)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "subpanel_1_5d_variance_vs_distance_normalized.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def subpanel_1_5g(data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619",
                  output_dir="results/Figure1_5"):
    """
    Subpanel 1.5g: Example scatterplots.
    
    Two scatterplots:
    - First: 1080nm FarRed vs 1080nm Broad
    - Second: 1240nm FarRed vs 1240nm Broad
    For neptune and cherry.
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    fluorophores = ["mNeptune"]  # Only mNeptune
    comparisons = [
        {"wl": 1080, "filter1": "FarRed", "filter2": "BR2"},
        {"wl": 1240, "filter1": "FarRed", "filter2": "BR2"}
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for comp_idx, comp in enumerate(comparisons):
        ax = axes[comp_idx]
        wl = comp["wl"]
        filter1 = comp["filter1"]
        filter2 = comp["filter2"]
        
        # Load data for both filters for each fluorophore
        all_ch1_data = []
        all_ch2_data = []
        fp_labels = []
        
        for fp_name in fluorophores:
            try:
                # Channel 1: FarRed filter
                ch1_data, _ = load_channel_data(data_dir, fp_name, wl, filter1, channel_num=1)
                # Channel 2: Broad filter
                ch2_data, _ = load_channel_data(data_dir, fp_name, wl, filter2, channel_num=1)
                
                all_ch1_data.append(ch1_data)
                all_ch2_data.append(ch2_data)
                fp_labels.extend([fp_name] * len(ch1_data))
            except (ValueError, FileNotFoundError) as e:
                print(f"Warning: Could not load data for {fp_name} {wl}nm: {e}")
                continue
        
        if len(all_ch1_data) == 0:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Combine data
        ch1_combined = np.concatenate(all_ch1_data)
        ch2_combined = np.concatenate(all_ch2_data)
        
        # Filter and subsample
        max_value = 3000
        ch1_filtered, ch2_filtered, fp_labels_filtered, _ = filter_by_distance(
            ch1_combined, ch2_combined, labels=fp_labels, max_distance=max_value * np.sqrt(2)
        )
        
        ch1_plot, ch2_plot, fp_labels_plot = bin_and_subsample_by_distance(
            ch1_filtered, ch2_filtered, fp_labels_filtered,
            bin_width=100, samples_per_bin=300, max_distance=max_value * np.sqrt(2)
        )
        
        # Plot
        fp_colors = {fp: cfg.fluorophore_colors.get(fp, "#000000") for fp in fluorophores}
        colors_list = [fp_colors[label] for label in fp_labels_plot]
        ax.scatter(ch1_plot, ch2_plot, c=colors_list, alpha=0.4, s=2)
        
        ax.set_xlabel(f"Channel 1 ({wl}nm {filter1})", fontsize=10)
        ax.set_ylabel(f"Channel 2 ({wl}nm {filter2})", fontsize=10)
        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{wl}nm: {filter1} vs {filter2}", fontsize=10)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "subpanel_1_5g_example_scatterplots.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def subpanel_1_5h(data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619",
                  output_dir="results/Figure1_5"):
    """
    Subpanel 1.5h: Cross-wavelength comparison scatterplot.
    
    X-axis: 800 Broad vs 1080 Broad (same filter, different excitations)
    Y-axis: 800 Red vs 1080 Red (same filter, different excitations)
    For neptune and cherry.
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    fluorophores = ["mCherry"]  # Only mCherry
    
    # Load data for each fluorophore
    all_ch1_data = []  # 800 Broad
    all_ch2_data = []  # 1080 Broad
    all_ch3_data = []  # 800 Red
    all_ch4_data = []  # 1080 Red
    fp_labels = []
    
    for fp_name in fluorophores:
        try:
            # Channel 1: 800nm Broad
            ch1_data, _ = load_channel_data(data_dir, fp_name, 800, "BR2", channel_num=1)
            # Channel 2: 1080nm Broad
            ch2_data, _ = load_channel_data(data_dir, fp_name, 1080, "BR2", channel_num=1)
            # Channel 3: 800nm Red
            ch3_data, _ = load_channel_data(data_dir, fp_name, 800, "Red", channel_num=1)
            # Channel 4: 1080nm Red
            ch4_data, _ = load_channel_data(data_dir, fp_name, 1080, "Red", channel_num=1)
            
            # Ensure all have same length (take minimum)
            min_len = min(len(ch1_data), len(ch2_data), len(ch3_data), len(ch4_data))
            all_ch1_data.append(ch1_data[:min_len])
            all_ch2_data.append(ch2_data[:min_len])
            all_ch3_data.append(ch3_data[:min_len])
            all_ch4_data.append(ch4_data[:min_len])
            fp_labels.extend([fp_name] * min_len)
        except (ValueError, FileNotFoundError) as e:
            print(f"Warning: Could not load data for {fp_name}: {e}")
            continue
    
    if len(all_ch1_data) == 0:
        print("Error: No data loaded for subpanel 1.5h")
        return
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Subplot 1: 800 Broad vs 1080 Broad
    ax1 = axes[0]
    if len(all_ch1_data) > 0:
        ch1_combined = np.concatenate(all_ch1_data)  # 800 Broad
        ch2_combined = np.concatenate(all_ch2_data)  # 1080 Broad
        
        max_value = 3000
        ch1_filtered, ch2_filtered, fp_labels_filtered, _ = filter_by_distance(
            ch1_combined, ch2_combined, labels=fp_labels, max_distance=max_value * np.sqrt(2)
        )
        
        ch1_plot, ch2_plot, fp_labels_plot = bin_and_subsample_by_distance(
            ch1_filtered, ch2_filtered, fp_labels_filtered,
            bin_width=100, samples_per_bin=300, max_distance=max_value * np.sqrt(2)
        )
        
        fp_colors = {fp: cfg.fluorophore_colors.get(fp, "#000000") for fp in fluorophores}
        colors_list = [fp_colors[label] for label in fp_labels_plot]
        ax1.scatter(ch1_plot, ch2_plot, c=colors_list, alpha=0.4, s=2)
        ax1.set_xlabel("800nm Broad", fontsize=12)
        ax1.set_ylabel("1080nm Broad", fontsize=12)
        ax1.set_xlim(0, max_value)
        ax1.set_ylim(0, max_value)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title("800 Broad vs 1080 Broad", fontsize=12)
    
    # Subplot 2: 800 Red vs 1080 Red
    ax2 = axes[1]
    if len(all_ch3_data) > 0:
        ch3_combined = np.concatenate(all_ch3_data)  # 800 Red
        ch4_combined = np.concatenate(all_ch4_data)  # 1080 Red
        
        max_value = 3000
        ch3_filtered, ch4_filtered, fp_labels_filtered, _ = filter_by_distance(
            ch3_combined, ch4_combined, labels=fp_labels, max_distance=max_value * np.sqrt(2)
        )
        
        ch3_plot, ch4_plot, fp_labels_plot = bin_and_subsample_by_distance(
            ch3_filtered, ch4_filtered, fp_labels_filtered,
            bin_width=100, samples_per_bin=300, max_distance=max_value * np.sqrt(2)
        )
        
        fp_colors = {fp: cfg.fluorophore_colors.get(fp, "#000000") for fp in fluorophores}
        colors_list = [fp_colors[label] for label in fp_labels_plot]
        ax2.scatter(ch3_plot, ch4_plot, c=colors_list, alpha=0.4, s=2)
        ax2.set_xlabel("800nm Red", fontsize=12)
        ax2.set_ylabel("1080nm Red", fontsize=12)
        ax2.set_xlim(0, max_value)
        ax2.set_ylim(0, max_value)
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title("800 Red vs 1080 Red", fontsize=12)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "subpanel_1_5h_cross_wavelength.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def subpanel_1_5z(data_dir="data/fig1_fig2_1color_3mice_singleplane_june20250619",
                  output_dir="results/Figure1_5"):
    """
    Subpanel 1.5z: mCherry 1240nm Broad vs FarRed scatterplot.
    
    Standalone scatterplot for mCherry:
    - Channel 1: 1240nm Broad (BR2)
    - Channel 2: 1240nm FarRed
    """
    import os
    from src.figure1 import load_channel_data
    
    os.makedirs(output_dir, exist_ok=True)
    
    fluorophore = "mCherry"
    
    # Load channel data
    try:
        ch1_data, _ = load_channel_data(data_dir, fluorophore, 1240, "BR2", channel_num=1)
        ch2_data, _ = load_channel_data(data_dir, fluorophore, 1240, "FarRed", channel_num=1)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Filter and subsample (same as other scatterplots)
    max_value = 3000
    ch1_filtered, ch2_filtered, _, _ = filter_by_distance(
        ch1_data, ch2_data, max_distance=max_value * np.sqrt(2)
    )
    
    # Create dummy labels
    dummy_labels = np.array([fluorophore] * len(ch1_filtered), dtype=object)
    
    # Use bin_and_subsample_by_distance (same as other scatterplots)
    ch1_plot, ch2_plot, fp_labels_plot = bin_and_subsample_by_distance(
        ch1_filtered, ch2_filtered, dummy_labels,
        bin_width=100, samples_per_bin=300, max_distance=max_value * np.sqrt(2)
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    fp_color = cfg.fluorophore_colors.get(fluorophore, "#000000")
    ax.scatter(ch1_plot, ch2_plot, c=fp_color, alpha=0.4, s=2, label=fluorophore)
    
    ax.set_xlabel("Channel 1 (1240nm Broad)", fontsize=12)
    ax.set_ylabel("Channel 2 (1240nm FarRed)", fontsize=12)
    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{fluorophore}: 1240nm Broad vs 1240nm FarRed", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "subpanel_1_5z_mcherry_1240_broad_vs_farred.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def subpanel_1_5ai(data_1_5a, data_dir="data/fig2_3color_inh_spatial_control_2p3_10072025",
                   output_dir="results/Figure1_5"):
    """
    Subpanel 1.5ai: Histogram of angles for both panels in 1.5a.
    
    Creates histograms showing angle distribution for each scatterplot in 1.5a.
    Uses code similar to fig 1 subpanel 8.
    """
    import os
    from skimage import io
    
    os.makedirs(output_dir, exist_ok=True)
    
    ch1_path = data_1_5a['ch1_path']
    ch2_paths = data_1_5a['ch2_paths']
    ch2_filenames = data_1_5a['ch2_filenames']
    
    # Create figure with 2 subplots (one for each panel in 1.5a)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, ch2_path in enumerate(ch2_paths):
        ax = axes[idx]
        
        try:
            # Load images (same as 1.5a)
            ch1_img = io.imread(ch1_path)
            ch2_img = io.imread(ch2_path)
            
            # Handle stacks: if 3D with shape (z, height, width), take max projection
            if len(ch1_img.shape) == 3:
                if ch1_img.shape[0] < ch1_img.shape[2]:  # Likely (z, height, width) stack
                    ch1_img = np.max(ch1_img, axis=0)  # Max projection across z
                else:  # Likely (height, width, channels)
                    ch1_img = ch1_img[:, :, 0]  # Take first channel
            elif len(ch1_img.shape) == 2:
                pass  # Already 2D
            
            if len(ch2_img.shape) == 3:
                if ch2_img.shape[0] < ch2_img.shape[2]:  # Likely (z, height, width) stack
                    ch2_img = np.max(ch2_img, axis=0)  # Max projection across z
                else:  # Likely (height, width, channels)
                    ch2_img = ch2_img[:, :, 0]  # Take first channel
            elif len(ch2_img.shape) == 2:
                pass  # Already 2D
            
            # Flatten images
            ch1_flat = ch1_img.flatten()
            ch2_flat = ch2_img.flatten()
            
            # Ensure same length
            min_len = min(len(ch1_flat), len(ch2_flat))
            ch1_flat = ch1_flat[:min_len]
            ch2_flat = ch2_flat[:min_len]
            
            # Filter by distance (same as 1.5a)
            max_value = 3000
            ch1_filtered, ch2_filtered, _, _ = filter_by_distance(
                ch1_flat, ch2_flat, max_distance=max_value * np.sqrt(2)
            )
            
            # Filter out pixels at/near origin to avoid 0-degree spike
            min_distance = 10
            ch1_float = ch1_filtered.astype(np.float64)
            ch2_float = ch2_filtered.astype(np.float64)
            distances = np.sqrt(ch1_float**2 + ch2_float**2)
            valid_mask = distances >= min_distance
            
            ch1_valid = ch1_filtered[valid_mask]
            ch2_valid = ch2_filtered[valid_mask]
            
            # Compute pixel angles (from x-axis, constrained to 0-90 degrees)
            pixel_angles_rad = np.arctan2(ch2_valid, ch1_valid)
            pixel_angles_deg = np.degrees(pixel_angles_rad)
            # Map all angles to 0-90 range (take absolute and fold)
            pixel_angles_deg = np.abs(pixel_angles_deg) % 180
            pixel_angles_deg = np.where(pixel_angles_deg > 90, 180 - pixel_angles_deg, pixel_angles_deg)
            
            # Create histogram bins (0 to 90 degrees)
            bins = np.linspace(0, 90, 91)  # 1 degree bins
            
            # Create histogram
            hist, _ = np.histogram(pixel_angles_deg, bins=bins)
            max_hist_value = np.max(hist)
            bin_width = bins[1] - bins[0]
            
            ax.bar(bins[:-1], hist, width=bin_width, color='gray', alpha=0.6, edgecolor='none')
            
            # Set labels
            ch2_poc = ch2_filenames[idx].split('_')[2].replace('poc', '')
            ax.set_xlabel("Angle (degrees)", fontsize=12)
            ax.set_ylabel("Pixel Count", fontsize=12)
            ax.set_title(f"Power: {ch2_poc}poc", fontsize=12)
            ax.set_xlim(0, 90)
            ax.set_ylim(0, max_hist_value * 1.15)
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
        except Exception as e:
            print(f"Error loading data for histogram: {e}")
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, f"Error loading data", ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "subpanel_1_5ai_angle_histograms.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


if __name__ == "__main__":
    import sys
    
    # Generate and save main figure
    save_main_figure()
    
    # Optionally generate supplement panels
    if len(sys.argv) > 1 and sys.argv[1] == "--supplements":
        save_all_supplement_panels()
    # Optionally generate standalone 1.5z
    elif len(sys.argv) > 1 and sys.argv[1] == "--1.5z":
        subpanel_1_5z()

