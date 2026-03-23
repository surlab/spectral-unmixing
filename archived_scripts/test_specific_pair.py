"""
One-off script to test a specific acquisition pair with scatter plot.
Tests: 1240nm broad vs 1240nm red for mNeptune
Uses the same scatterplot function as subpanel 5
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from figure1 import subpanel_5
import src.config as cfg

# Configuration
# Testing: 1240nm_broad_377poc_20mW vs 1240nm_red_493poc_30mW
data_dir = "data/fig1_fig2_1color_3mice_singleplane_june20250619"
fp_name = "mNeptune"
ch1_wl = 1240
ch1_filter = "broad"
ch2_wl = 1240
ch2_filter = "red"

print(f"Testing pair: {fp_name}")
print(f"  Channel 1: {ch1_wl}nm, {ch1_filter}")
print(f"  Channel 2: {ch2_wl}nm, {ch2_filter}")

# Create row_dict matching subpanel 5 format
row_dict = {
    'Fluorophores': [fp_name],
    'Channel 1': {'Excitation wavelength': ch1_wl, 'emission filter': ch1_filter},
    'Channel 2': {'Excitation wavelength': ch2_wl, 'emission filter': ch2_filter},
    'name': 'test_pair'  # Optional name
}

# Use subpanel_5 to create the scatterplot
print("\nGenerating scatterplot using subpanel_5 function...")
fig, ax = subpanel_5(row_dict, ax=None, data_dir=data_dir)

# Update title to show the specific pair
ax.set_title(f'{fp_name}: {ch1_wl}nm_{ch1_filter} vs {ch2_wl}nm_{ch2_filter}', fontsize=14)

plt.tight_layout()
output_path = "results/Figure1/test_pair_scatter.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150)
print(f"\nSaved scatter plot to: {output_path}")
plt.close()

