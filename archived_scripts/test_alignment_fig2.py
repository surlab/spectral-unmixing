"""
Test script to run TIFF alignment on fig2_3color_inh_spatial_control_2p3_10072025 directory.

This tests alignment on stacks.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.alignment import process_data_directory

if __name__ == "__main__":
    # Process only the fig2 directory
    data_dir = os.path.join("data", "fig2_3color_inh_spatial_control_2p3_10072025")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"Testing alignment on stacks: {data_dir}")
    print("This may take a while...")
    
    results = process_data_directory(
        data_dir_path=data_dir,
        align_xy_first=True,  # Use XY alignment first (more accurate but slower)
        preserve_metadata=True
    )
    
    print(f"\nCompleted processing {len(results)} directories")
    for dir_name, df in results.items():
        print(f"  {dir_name}: {len(df)} acquisitions processed")





