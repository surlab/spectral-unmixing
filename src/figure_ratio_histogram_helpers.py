"""
Shared helper functions for ratio histogram generation across figures.

This module contains reusable functions for:
- Creating ratio histogram plots with multiple fluorophores
- Adjusting x-axis width to ensure bars aren't wider than they are tall
- Formatting channel labels and legends
"""

import numpy as np
import matplotlib.pyplot as plt


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



