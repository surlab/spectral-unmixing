"""
Shared figure-related configuration.

This module contains parameters that are shared across multiple figures (plotting defaults,
color schemes, filtering thresholds). They are intentionally separated from `config.py`,
which may later be used for a lightweight "app" configuration.
"""

# Percentile bounds for pixel filtering and vector computation
pixel_intensity_lower_percentile = 20  # Ignore lowest X% of pixels
pixel_intensity_upper_percentile = 95  # Ignore highest X% of pixels
vector_scaling_percentile = 70  # Scale vectors to reach this percentile
cone_coverage_percentile = 95  # Percentage of pixels that should fall inside classification cone
classification_zone_percentile = 90  # Percentage of pixels to include in classification zone (symmetric around vector)

# Separability score and variance computation parameters
N_fluorophores_default = 10000  # Default number of fluorophores in excitation volume for separability/variance calculations

# Classification plotting defaults
classification_zone_min_distance = 500
angle_histogram_bin_size_degrees = 1

# Color scheme for fluorophores in figures
fluorophore_colors = {
    "TdTomato": "#FFB000",  # More orange-gold (less yellow than #FFD700)
    "mCherry": "#E31A1C",   # Red
    "mNeptune": "#4B0082",  # Deep violet (Indigo)
}

# Colors and styles for excitation wavelength lines in plots
# Default mapping for Figure 1 wavelengths (in order 1040, 1080, 1180, 1240).
excitation_line_colors = [
    fluorophore_colors["TdTomato"],  # 1040
    "#000000",                       # 1080
    fluorophore_colors["mCherry"],   # 1180
    fluorophore_colors["mNeptune"],  # 1240
]
excitation_line_styles = ["--", "--", "--", "--"]

# Emission filter set definitions
emission_filter_sets = {
    "BR2": "Chroma-broad--t--0.csv",
    "Orange": [550, 580],
    "Red": "Chroma-at60530m--t--0.csv",
    "FarRed": "Chroma-et67050m--t--0.csv",
    "DarkBlue": [400, 440],
    "Blue": [445, 475],
    "Cyan": [475, 495],
    "NarrowGreen": "Chroma-et51020m--t--0.csv",
    "Yellow": "Chroma-et53530m--t--0.csv",
}

# Display names for filters (for plot labels)
filter_display_names = {
    "BR2": "Broad",
    "Orange": "Orange",
    "Red": "Red",
    "FarRed": "Far Red",
    "DarkBlue": "Dark Blue",
    "Blue": "Blue",
    "Cyan": "Cyan",
    "NarrowGreen": "Narrow Green",
    "Yellow": "Yellow",
}

# Display-only filter ranges for spectra visualization
emission_filter_display_ranges = {
    "BR2": [560, 700],
    "Orange": [550, 580],
    "Red": [590, 620],
    "FarRed": [645, 695],
    "DarkBlue": [400, 440],
    "Blue": [445, 475],
    "Cyan": [475, 495],
    "NarrowGreen": [500, 520],
    "Yellow": [520, 550],
}

# Colors for emission filters in spectra plots
emission_filter_colors = {
    "BR2": "#B0B0B0",
    "Orange": "#FF8C00",  # Deeper orange for better distinction
    "Red": "#E31A1C",
    "FarRed": "#4B0082",
    "DarkBlue": "#253494",
    "Blue": "#3182BD",
    "Cyan": "#41B6C4",
    "NarrowGreen": "#31A354",
    "Yellow": "#FFB000",
}

# Figure 1 zoom settings (used in subpanels 5 and 6)
figure1_zoom_factor = 5
figure1_zoom_pick_percentile = 99.5

# Figure 1.5 arrow styling
thick_linewidth = 18
arrow_mutation_scale = 50

# Row type colors and markers
row_colors = {
    "excitation based": "#2166ac",
    "emission based": "#4393c3",
    "dual domain": "#92c5de",
}
row_markers = {
    "excitation based": "o",
    "emission based": "s",
    "dual domain": "^",
}

