"""
Script to run TIFF alignment on all directories in data folder.

This script processes all image directories, aligning them to 1080nm reference images
and saving aligned images with new naming conventions.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.alignment import process_data_directory

if __name__ == "__main__":
    # Process all directories in data folder
    data_dir = os.path.join("data")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"Processing all directories in {data_dir}")
    print("This may take a while...")
    
    results = process_data_directory(
        data_dir_path=data_dir,
        align_xy_first=True,  # Use XY alignment first (more accurate but slower)
        preserve_metadata=True
    )
    
    print(f"\nCompleted processing {len(results)} directories")
    for dir_name, df in results.items():
        print(f"  {dir_name}: {len(df)} acquisitions processed")





