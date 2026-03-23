"""
Script to create GCampCa- CSV file.
GCampCa- combines:
- 2P excitation: from tagBFP
- 1P excitation: from tagBFP  
- Emission: from EGFP
"""

import pandas as pd
import os

# Paths
demo_data_dir = "dev_scripts/demo_data"
tagbfp_path = os.path.join(demo_data_dir, "tagbfp.csv")
egfp_path = os.path.join(demo_data_dir, "egfp.csv")
output_path = os.path.join(demo_data_dir, "GCampCa-.csv")

# Read source files
print(f"Reading {tagbfp_path}...")
tagbfp_df = pd.read_csv(tagbfp_path)
tagbfp_df.columns = tagbfp_df.columns.str.strip()

print(f"Reading {egfp_path}...")
egfp_df = pd.read_csv(egfp_path)
egfp_df.columns = egfp_df.columns.str.strip()

# Get wavelength column (use tagBFP as base)
wavelength_col = tagbfp_df["wavelength"].copy()

# Get 2P excitation from tagBFP
tagbfp_2p_col = None
for col in tagbfp_df.columns:
    if "TagBFP" in col and ("2p" in col.lower() or "2P" in col):
        tagbfp_2p_col = col
        break

if not tagbfp_2p_col:
    raise ValueError("Could not find TagBFP 2p column")

# Get 1P excitation from tagBFP
tagbfp_ex_col = None
for col in tagbfp_df.columns:
    if "TagBFP" in col and ("ex" in col.lower() and "2p" not in col.lower()):
        tagbfp_ex_col = col
        break

if not tagbfp_ex_col:
    raise ValueError("Could not find TagBFP ex column")

# Get emission from EGFP
egfp_em_col = None
for col in egfp_df.columns:
    if "EGFP" in col and ("em" in col.lower() or "emission" in col.lower()):
        egfp_em_col = col
        break

if not egfp_em_col:
    raise ValueError("Could not find EGFP em column")

print(f"Using columns:")
print(f"  TagBFP 2p: {tagbfp_2p_col}")
print(f"  TagBFP ex: {tagbfp_ex_col}")
print(f"  EGFP em: {egfp_em_col}")

# Merge dataframes on wavelength
merged = pd.merge(
    tagbfp_df[["wavelength", tagbfp_2p_col, tagbfp_ex_col]],
    egfp_df[["wavelength", egfp_em_col]],
    on="wavelength",
    how="outer"
).sort_values("wavelength")

# Create new dataframe with GCampCa- columns
gcampca_minus_df = pd.DataFrame({
    "wavelength": merged["wavelength"],
    "GCampCa- 2p": merged[tagbfp_2p_col],
    "GCampCa- ex": merged[tagbfp_ex_col],
    "GCampCa- em": merged[egfp_em_col]
})

# Fill NaN values with empty strings (to match CSV format)
gcampca_minus_df = gcampca_minus_df.fillna("")

# Save to CSV
gcampca_minus_df.to_csv(output_path, index=False)
print(f"\nCreated {output_path}")
print(f"Shape: {gcampca_minus_df.shape}")
print(f"Wavelength range: {gcampca_minus_df['wavelength'].min():.0f}-{gcampca_minus_df['wavelength'].max():.0f}nm")
