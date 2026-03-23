"""
Script to update LSS-mKate1.csv to include mTFP1's 2P excitation spectra.
This modifies the CSV file directly so the 2P data is permanently stored.
"""

import pandas as pd
import os

# Paths
demo_data_dir = "dev_scripts/demo_data"
lssmkate_path = os.path.join(demo_data_dir, "LSS-mKate1.csv")
mtfp1_path = os.path.join(demo_data_dir, "mtfp1.csv")

# Read LSS-mKate1 CSV
print(f"Reading {lssmkate_path}...")
lssmkate_df = pd.read_csv(lssmkate_path)
lssmkate_df.columns = lssmkate_df.columns.str.strip()

# Read mTFP1 CSV
print(f"Reading {mtfp1_path}...")
mtfp1_df = pd.read_csv(mtfp1_path)
mtfp1_df.columns = mtfp1_df.columns.str.strip()

# Find mTFP1 2p column
mtfp1_2p_col = None
for col in mtfp1_df.columns:
    if "mTFP1" in col and ("2p" in col.lower() or "2P" in col):
        mtfp1_2p_col = col
        break

if not mtfp1_2p_col:
    raise ValueError("Could not find mTFP1 2p column")

print(f"Using mTFP1 column: {mtfp1_2p_col}")

# Merge dataframes on wavelength using outer join to include all wavelengths
# This ensures we get wavelengths from both dataframes
merged = pd.merge(
    lssmkate_df,
    mtfp1_df[["wavelength", mtfp1_2p_col]],
    on="wavelength",
    how="outer"
).sort_values("wavelength").reset_index(drop=True)

# Create LSSmKAte 2p column from mTFP1 2p
merged["LSSmKAte 2p"] = merged[mtfp1_2p_col]

# Interpolate missing values for wavelengths between existing data points
# This fills in gaps where mTFP1 has data but LSS-mKate1 doesn't have that wavelength
merged["LSSmKAte 2p"] = merged["LSSmKAte 2p"].interpolate(method='linear', limit_direction='both')

# Fill remaining NaN values with 0 (for wavelengths outside the mTFP1 range)
merged["LSSmKAte 2p"] = merged["LSSmKAte 2p"].fillna(0)

# Drop the temporary mTFP1 column if it was added
if mtfp1_2p_col in merged.columns and mtfp1_2p_col != "LSSmKAte 2p":
    merged = merged.drop(columns=[mtfp1_2p_col])

# Fill NaN values in other columns (emission, 1P excitation) with 0 for new wavelengths
for col in merged.columns:
    if col not in ["wavelength", "LSSmKAte 2p"]:
        merged[col] = merged[col].fillna(0)

# Save updated CSV
merged.to_csv(lssmkate_path, index=False)
print(f"\nUpdated {lssmkate_path}")
print(f"Added 'LSSmKAte 2p' column from mTFP1")
print(f"Shape: {merged.shape}")
print(f"Wavelength range: {merged['wavelength'].min():.0f}-{merged['wavelength'].max():.0f}nm")
print(f"LSSmKAte 2p max value: {merged['LSSmKAte 2p'].max():.6f}")

