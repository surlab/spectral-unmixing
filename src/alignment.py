"""
TIFF image alignment module.

This module provides functions for aligning TIFF images to a reference image,
handling both single images and stacks. Alignment is done using phase cross-correlation
for XY shifts, with optional Z-plane matching for stacks.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import tifffile as tf
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_pockels_from_folder_name(folder_name):
    """
    Extract pockels value from folder name.
    
    Parameters
    ----------
    folder_name : str
        Folder name like "BR2EmFilt_1080nm_230poc_z13_580pmt-007"
        
    Returns
    -------
    int or None
        Pockels value (e.g., 230) or None if not found
    """
    match = re.search(r'(\d+)poc', folder_name)
    if match:
        return int(match.group(1))
    return None


def find_1080_reference(directory_path):
    """
    Find the 1080nm reference image in a directory.
    
    Priority:
    1. BR2EmFilt_1080nm with highest pockels
    2. Any 1080nm with highest pockels
    
    Parameters
    ----------
    directory_path : str
        Path to directory containing image subdirectories
        
    Returns
    -------
    str
        Path to the Ch1 reference image file
    dict
        Dictionary with reference info (filter, wavelength, pockels, z_plane)
        
    Raises
    ------
    ValueError
        If no 1080nm reference is found
    """
    # Find all subdirectories
    subdirs = [d for d in os.listdir(directory_path) 
               if os.path.isdir(os.path.join(directory_path, d))]
    
    # Filter for 1080nm directories
    ref_1080_dirs = []
    for subdir in subdirs:
        if '1080nm' in subdir:
            pockels = extract_pockels_from_folder_name(subdir)
            if pockels is not None:
                ref_1080_dirs.append((subdir, pockels))
    
    if len(ref_1080_dirs) == 0:
        raise ValueError(f"No 1080nm reference found in {directory_path}")
    
    # Sort by: 1) BR2EmFilt first, 2) highest pockels
    def sort_key(item):
        subdir, pockels = item
        is_br2 = 'BR2EmFilt' in subdir
        return (not is_br2, -pockels)  # False (BR2) sorts before True
    
    ref_1080_dirs.sort(key=sort_key)
    ref_subdir = ref_1080_dirs[0][0]
    ref_pockels = ref_1080_dirs[0][1]
    
    # Find Ch1 file in reference directory
    ref_dir_path = os.path.join(directory_path, ref_subdir)
    ch1_files = glob.glob(os.path.join(ref_dir_path, "*_Ch1_*.ome.tif"))
    
    if len(ch1_files) == 0:
        raise ValueError(f"No Ch1 file found in reference directory {ref_dir_path}")
    
    ref_file = ch1_files[0]
    
    # Extract filter name
    filter_name = "BR2EmFilt" if "BR2EmFilt" in ref_subdir else "Unknown"
    
    # Extract z plane if present
    z_match = re.search(r'z(\d+)', ref_subdir)
    z_plane = int(z_match.group(1)) if z_match else None
    
    ref_info = {
        'filter': filter_name,
        'wavelength': 1080,
        'pockels': ref_pockels,
        'z_plane': z_plane,
        'subdir': ref_subdir
    }
    
    logger.info(f"Found 1080nm reference: {ref_subdir} (pockels={ref_pockels})")
    
    return ref_file, ref_info


def load_image_or_stack(file_path):
    """
    Load a TIFF image, handling both single images and stacks.
    
    Parameters
    ----------
    file_path : str
        Path to TIFF file
        
    Returns
    -------
    np.ndarray
        Image array. Shape: (height, width) for single image,
        (z, height, width) for stack
    bool
        True if stack, False if single image
    """
    image = tf.imread(file_path)
    
    # Handle different shapes
    if len(image.shape) == 2:
        return image, False
    elif len(image.shape) == 3:
        # Could be (z, height, width) or (height, width, channels)
        if image.shape[0] < image.shape[2]:
            # Likely (z, height, width)
            return image, True
        else:
            # Likely (height, width, channels) - take first channel
            return image[:, :, 0], False
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")


def align_xy_to_reference(moving_image, reference_image, upsample_factor=10):
    """
    Align a moving image to a reference image using phase cross-correlation.
    
    Parameters
    ----------
    moving_image : np.ndarray
        2D image to align
    reference_image : np.ndarray
        2D reference image
    upsample_factor : int
        Upsampling factor for sub-pixel registration
        
    Returns
    -------
    np.ndarray
        Aligned image
    tuple
        Shift (y_shift, x_shift) applied
    float
        Correlation coefficient after alignment
    """
    shift_estimate, error, phasediff = phase_cross_correlation(
        reference_image, moving_image, upsample_factor=upsample_factor
    )
    
    aligned_image = shift(moving_image, shift=shift_estimate, mode='nearest')
    
    # Compute correlation after alignment
    corr_coef = np.corrcoef(
        aligned_image.flatten(), 
        reference_image.flatten()
    )[0, 1]
    
    return aligned_image, shift_estimate, corr_coef


def find_best_z_match(moving_stack, reference_image, align_xy_first=True, 
                      upsample_factor=10):
    """
    Find the best matching Z plane in a stack to a reference image.
    
    Parameters
    ----------
    moving_stack : np.ndarray
        3D stack with shape (z, height, width)
    reference_image : np.ndarray
        2D reference image
    align_xy_first : bool
        If True, align XY first then find best Z (slower but more accurate).
        If False, find best Z first then align XY (faster).
    upsample_factor : int
        Upsampling factor for sub-pixel registration
        
    Returns
    -------
    np.ndarray
        Best matching aligned image
    int
        Z plane index of best match
    float
        Correlation coefficient of best match
    """
    if align_xy_first:
        # Align each plane to reference, then find best match
        best_corr = -np.inf
        best_z = 0
        best_aligned = None
        
        for z_idx in range(moving_stack.shape[0]):
            aligned, _, corr = align_xy_to_reference(
                moving_stack[z_idx, :, :], 
                reference_image, 
                upsample_factor
            )
            if corr > best_corr:
                best_corr = corr
                best_z = z_idx
                best_aligned = aligned
        
        return best_aligned, best_z, best_corr
    else:
        # Find best Z first (no alignment), then align
        best_corr = -np.inf
        best_z = 0
        
        for z_idx in range(moving_stack.shape[0]):
            # Quick correlation without alignment
            corr = np.corrcoef(
                moving_stack[z_idx, :, :].flatten(),
                reference_image.flatten()
            )[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_z = z_idx
        
        # Now align the best match
        aligned, _, final_corr = align_xy_to_reference(
            moving_stack[best_z, :, :],
            reference_image,
            upsample_factor
        )
        
        return aligned, best_z, final_corr


def extract_target_z_planes(stack, every_nth=10, exclude_edges=True):
    """
    Extract target Z planes from a reference stack (every nth plane).
    
    Parameters
    ----------
    stack : np.ndarray
        3D stack with shape (z, height, width)
    every_nth : int
        Extract every nth plane (e.g., 10 means planes 0, 10, 20, ...)
    exclude_edges : bool
        If True, exclude first and last planes to avoid edge effects
        
    Returns
    -------
    list
        List of Z plane indices to use as targets
    """
    num_planes = stack.shape[0]
    
    if exclude_edges:
        # Exclude first and last plane
        start_idx = 1
        end_idx = num_planes - 1
    else:
        start_idx = 0
        end_idx = num_planes
    
    target_indices = list(range(start_idx, end_idx, every_nth))
    
    if len(target_indices) == 0:
        # If no targets after exclusion, use middle plane
        target_indices = [num_planes // 2]
        logger.warning(f"Only 1 target Z plane available after edge exclusion")
    elif len(target_indices) == 1:
        logger.warning(f"Only 1 target Z plane available")
    
    return target_indices


def generate_output_filename(folder_name, z_plane=None):
    """
    Generate output filename from folder name.
    
    Format: filterName_excNM_valPoc[_zZ].tif
    
    Parameters
    ----------
    folder_name : str
        Original folder name (e.g., "BR2EmFilt_1080nm_230poc_z13_580pmt-007")
    z_plane : int, optional
        Z plane number to include in filename
        
    Returns
    -------
    str
        Output filename (e.g., "BR2EmFilt_1080nm_230poc.tif" or 
        "BR2EmFilt_1080nm_230poc_z13.tif")
    """
    # Extract filter name, wavelength, and pockels
    filter_match = re.search(r'([A-Za-z]+EmFilt)', folder_name)
    wavelength_match = re.search(r'(\d+)nm', folder_name)
    pockels_match = re.search(r'(\d+)poc', folder_name)
    
    if not all([filter_match, wavelength_match, pockels_match]):
        # Fallback: use folder name as base
        base_name = folder_name.split('-')[0]  # Remove trailing number
        if z_plane is not None:
            return f"{base_name}_z{z_plane}.tif"
        return f"{base_name}.tif"
    
    filter_name = filter_match.group(1)
    wavelength = wavelength_match.group(1)
    pockels = pockels_match.group(1)
    
    if z_plane is not None:
        filename = f"{filter_name}_{wavelength}nm_{pockels}poc_z{z_plane}.tif"
    else:
        filename = f"{filter_name}_{wavelength}nm_{pockels}poc.tif"
    
    return filename


def align_single_image_to_reference(image_path, reference_path, output_path, 
                                     preserve_metadata=True):
    """
    Align a single image to a reference and save.
    
    Parameters
    ----------
    image_path : str
        Path to image to align
    reference_path : str
        Path to reference image
    output_path : str
        Path to save aligned image
    preserve_metadata : bool
        If True, try to preserve TIFF metadata
        
    Returns
    -------
    dict
        Alignment info (shift, correlation, z_plane if applicable)
    """
    logger.info(f"Aligning {os.path.basename(image_path)} to reference")
    
    moving_image, is_stack = load_image_or_stack(image_path)
    if is_stack:
        raise ValueError(f"Expected single image, got stack: {image_path}")
    
    ref_image, ref_is_stack = load_image_or_stack(reference_path)
    if ref_is_stack:
        # Use middle plane of reference stack
        mid_z = ref_image.shape[0] // 2
        ref_image = ref_image[mid_z, :, :]
        logger.info(f"Using middle plane (z={mid_z}) of reference stack")
    
    aligned, shift_est, corr = align_xy_to_reference(moving_image, ref_image)
    
    logger.info(f"  Shift: {shift_est}, Correlation: {corr:.4f}")
    
    # Save aligned image
    if preserve_metadata:
        try:
            with tf.TiffFile(image_path) as tif:
                metadata = {}
                if hasattr(tif, 'ome_metadata'):
                    metadata['ome_metadata'] = tif.ome_metadata
                tf.imwrite(output_path, aligned.astype('uint16'), **metadata)
        except:
            # Fallback if metadata preservation fails
            tf.imwrite(output_path, aligned.astype('uint16'))
    else:
        tf.imwrite(output_path, aligned.astype('uint16'))
    
    return {
        'shift': shift_est,
        'correlation': corr,
        'z_plane': None
    }


def align_stack_to_reference(stack_path, reference_path, output_path,
                             align_xy_first=True, preserve_metadata=True):
    """
    Align a stack to a reference, finding best Z match.
    
    Parameters
    ----------
    stack_path : str
        Path to stack to align
    reference_path : str
        Path to reference image
    output_path : str
        Path to save aligned single image (best match)
    align_xy_first : bool
        If True, align XY first then find best Z
    preserve_metadata : bool
        If True, try to preserve TIFF metadata
        
    Returns
    -------
    dict
        Alignment info (shift, correlation, z_plane)
    """
    logger.info(f"Aligning stack {os.path.basename(stack_path)} to reference")
    
    moving_stack, is_stack = load_image_or_stack(stack_path)
    if not is_stack:
        raise ValueError(f"Expected stack, got single image: {stack_path}")
    
    ref_image, ref_is_stack = load_image_or_stack(reference_path)
    if ref_is_stack:
        # Use middle plane of reference stack
        mid_z = ref_image.shape[0] // 2
        ref_image = ref_image[mid_z, :, :]
        logger.info(f"Using middle plane (z={mid_z}) of reference stack")
    
    aligned, best_z, corr = find_best_z_match(
        moving_stack, ref_image, align_xy_first=align_xy_first
    )
    
    logger.info(f"  Best Z: {best_z}, Correlation: {corr:.4f}")
    
    # Save aligned image
    if preserve_metadata:
        try:
            with tf.TiffFile(stack_path) as tif:
                metadata = {}
                if hasattr(tif, 'ome_metadata'):
                    metadata['ome_metadata'] = tif.ome_metadata
                tf.imwrite(output_path, aligned.astype('uint16'), **metadata)
        except:
            tf.imwrite(output_path, aligned.astype('uint16'))
    else:
        tf.imwrite(output_path, aligned.astype('uint16'))
    
    return {
        'shift': None,  # Shift is computed per plane
        'correlation': corr,
        'z_plane': best_z
    }


def process_directory_with_reference(directory_path, reference_path, reference_info,
                                     output_base_dir, align_xy_first=True,
                                     preserve_metadata=True):
    """
    Process all images in a directory, aligning them to a reference.
    
    Parameters
    ----------
    directory_path : str
        Directory containing image subdirectories
    reference_path : str
        Path to reference image file
    reference_info : dict
        Reference info dictionary
    output_base_dir : str
        Base directory for output (one level up from subdirectories)
    align_xy_first : bool
        For stacks, whether to align XY first
    preserve_metadata : bool
        Whether to preserve TIFF metadata
        
    Returns
    -------
    pd.DataFrame
        DataFrame with alignment results (acquisition, z_plane, correlation, etc.)
    """
    results = []
    
    # Load reference image once
    ref_image, ref_is_stack = load_image_or_stack(reference_path)
    
    # Get all subdirectories (acquisition folders)
    # Filter out any that don't look like acquisition folders (e.g., "References" folders)
    subdirs = [d for d in os.listdir(directory_path) 
               if os.path.isdir(os.path.join(directory_path, d))
               and not d.startswith('.')  # Skip hidden directories
               and d.lower() != 'references']  # Skip References folders
    
    # Process reference first (save without alignment)
    ref_subdir = reference_info['subdir']
    
    if ref_is_stack:
        # Extract every 10th plane, excluding edges
        # We'll determine which target planes are actually valid after processing other stacks
        # (some may be excluded if their best matches are at edges of other stacks)
        target_z_planes = extract_target_z_planes(ref_image, every_nth=10, 
                                                  exclude_edges=True)
        # Don't save reference yet - wait until we know which target planes are valid
    else:
        # Save single reference image
        output_filename = generate_output_filename(ref_subdir)
        output_path = os.path.join(output_base_dir, output_filename)
        tf.imwrite(output_path, ref_image.astype('uint16'))
        logger.info(f"Saved reference image: {output_filename}")
    
    # Track invalid target planes (those with edge matches in any stack)
    # This ensures all stacks end up with the same number of planes
    invalid_target_planes = set()  # Target planes to exclude (have edge matches in at least one stack)
    saved_stacks = {}  # Store saved stack paths and aligned planes for filtering
    
    # Process all subdirectories
    total_subdirs = len([s for s in subdirs if s != ref_subdir])
    logger.info(f"\nProcessing {total_subdirs} acquisitions...")
    
    for idx, subdir in enumerate([s for s in subdirs if s != ref_subdir], 1):
        logger.info(f"\n[{idx}/{total_subdirs}] Processing acquisition: {subdir}")
        subdir_path = os.path.join(directory_path, subdir)
        ch1_files = glob.glob(os.path.join(subdir_path, "*_Ch1_*.ome.tif"))
        
        if len(ch1_files) == 0:
            logger.warning(f"  No Ch1 files found in {subdir}, skipping")
            continue
        
        image_file = ch1_files[0]
        image, is_stack = load_image_or_stack(image_file)
        
        if is_stack:
            if ref_is_stack:
                # Both are stacks: align all target planes, track edge matches
                num_planes_moving = image.shape[0]
                aligned_planes = []
                target_z_indices = []  # Track which target_z each aligned plane corresponds to
                
                logger.info(f"  Processing {subdir}: aligning {len(target_z_planes)} target planes...")
                
                for idx, target_z in enumerate(target_z_planes, 1):
                    logger.info(f"    Matching target plane {idx}/{len(target_z_planes)} (z={target_z})...")
                    target_ref = ref_image[target_z, :, :]
                    aligned, best_z, corr = find_best_z_match(
                        image, target_ref, align_xy_first=align_xy_first
                    )
                    
                    # Check if best match is at edge - mark target plane as invalid for ALL stacks
                    if best_z == 0 or best_z == (num_planes_moving - 1):
                        invalid_target_planes.add(target_z)
                        logger.warning(f"      Target plane {target_z}: edge match (z={best_z}), "
                                     f"will exclude from all stacks")
                    else:
                        # Only add if not at edge (we'll filter later)
                        aligned_planes.append(aligned)
                        target_z_indices.append(target_z)
                        logger.info(f"      Matched to z={best_z} (corr={corr:.4f})")
                    
                    results.append({
                        'acquisition': subdir,
                        'reference_z': target_z,
                        'selected_z': best_z,
                        'correlation': corr
                    })
                
                # Save compiled stack (may include invalid planes, will filter later)
                logger.info(f"  Saving aligned stack for {subdir}...")
                compiled_stack = np.stack(aligned_planes, axis=0)
                output_filename = generate_output_filename(subdir)
                output_path = os.path.join(output_base_dir, output_filename)
                tf.imwrite(output_path, compiled_stack.astype('uint16'))
                
                # Store for later filtering
                saved_stacks[subdir] = {
                    'path': output_path,
                    'stack': compiled_stack,
                    'target_z_indices': target_z_indices
                }
                
                logger.info(f"  Saved aligned stack: {output_filename} "
                          f"({len(aligned_planes)} planes, will filter invalid planes)")
            else:
                # Moving is stack, reference is single: find best Z match
                info = align_stack_to_reference(
                    image_file, reference_path, 
                    os.path.join(output_base_dir, generate_output_filename(subdir)),
                    align_xy_first=align_xy_first, preserve_metadata=preserve_metadata
                )
                results.append({
                    'acquisition': subdir,
                    'reference_z': None,
                    'selected_z': info['z_plane'],
                    'correlation': info['correlation']
                })
        else:
            # Single image: align to reference
            if ref_is_stack:
                # Use middle plane of reference stack
                mid_z = ref_image.shape[0] // 2
                target_ref = ref_image[mid_z, :, :]
                # Create temporary reference file path for alignment function
                aligned, shift_est, corr = align_xy_to_reference(image, target_ref)
                
                output_filename = generate_output_filename(subdir)
                output_path = os.path.join(output_base_dir, output_filename)
                tf.imwrite(output_path, aligned.astype('uint16'))
                
                results.append({
                    'acquisition': subdir,
                    'reference_z': mid_z,
                    'selected_z': None,
                    'correlation': corr
                })
            else:
                info = align_single_image_to_reference(
                    image_file, reference_path,
                    os.path.join(output_base_dir, generate_output_filename(subdir)),
                    preserve_metadata=preserve_metadata
                )
                results.append({
                    'acquisition': subdir,
                    'reference_z': None,
                    'selected_z': None,
                    'correlation': info['correlation']
                })
    
    # Filter out invalid target planes from all saved stacks and resave
    if ref_is_stack and len(invalid_target_planes) > 0:
        # Determine valid target planes
        valid_target_planes = [z for z in target_z_planes if z not in invalid_target_planes]
        
        if len(valid_target_planes) == 0:
            logger.warning(f"All target planes have edge matches in at least one stack!")
            logger.warning(f"Using all extracted planes as fallback (may include edge effects)")
            valid_target_planes = target_z_planes
        else:
            logger.info(f"\nFiltering stacks: keeping {len(valid_target_planes)} valid target planes "
                      f"(excluding {len(invalid_target_planes)} due to edge matches)")
        
        # Filter and resave all stacks
        logger.info(f"Resaving {len(saved_stacks)} stacks with filtered planes...")
        for idx, (subdir, stack_info) in enumerate(saved_stacks.items(), 1):
            logger.info(f"  Resaving {idx}/{len(saved_stacks)}: {subdir}...")
            # Find indices of valid target planes in this stack
            valid_indices = [i for i, tz in enumerate(stack_info['target_z_indices']) 
                           if tz in valid_target_planes]
            
            if len(valid_indices) == 0:
                logger.warning(f"    No valid planes for {subdir} after filtering")
                continue
            
            # Extract valid planes
            filtered_stack = stack_info['stack'][valid_indices, :, :]
            
            # Resave with only valid planes
            tf.imwrite(stack_info['path'], filtered_stack.astype('uint16'))
            logger.info(f"    Resaved: {len(valid_indices)} valid planes "
                      f"(removed {len(stack_info['target_z_indices']) - len(valid_indices)} invalid)")
        
        # Save reference stack with only valid target planes (same as all other stacks)
        logger.info(f"  Saving reference stack with {len(valid_target_planes)} valid planes...")
        reduced_stack = ref_image[valid_target_planes, :, :]
        output_filename = generate_output_filename(ref_subdir)
        output_path = os.path.join(output_base_dir, output_filename)
        tf.imwrite(output_path, reduced_stack.astype('uint16'))
        
        logger.info(f"  Saved reference stack: {output_filename} with {len(valid_target_planes)} planes "
                  f"(same as all other aligned stacks)")
    elif ref_is_stack:
        # No invalid planes, save reference stack normally
        reduced_stack = ref_image[target_z_planes, :, :]
        output_filename = generate_output_filename(ref_subdir)
        output_path = os.path.join(output_base_dir, output_filename)
        tf.imwrite(output_path, reduced_stack.astype('uint16'))
        
        logger.info(f"Saved reference stack: {output_filename} with {len(target_z_planes)} planes")
    
    return pd.DataFrame(results)


def process_data_directory(data_dir_path, align_xy_first=True, preserve_metadata=True):
    """
    Process all directories in the data folder.
    
    Parameters
    ----------
    data_dir_path : str
        Path to data directory (e.g., "data")
    align_xy_first : bool
        For stacks, whether to align XY first
    preserve_metadata : bool
        Whether to preserve TIFF metadata
        
    Returns
    -------
    dict
        Dictionary mapping directory names to alignment DataFrames
    """
    all_results = {}
    
    # Find all top-level directories in data
    top_level_dirs = [d for d in os.listdir(data_dir_path)
                     if os.path.isdir(os.path.join(data_dir_path, d))]
    
    # Check if we're already at a dataset level (contains acquisition folders directly)
    # vs. at the data/ level (contains dataset directories)
    # If we see acquisition folders (EmFilt_, nm_, _poc), we're at the level that contains them
    # If we see _mouse or FOV, we're one level up and need to go deeper
    has_acquisition_folders = any(any(x in d for x in ['EmFilt_', 'nm_', '_poc']) for d in top_level_dirs)
    has_fluorophore_dirs = any('_mouse' in d.lower() for d in top_level_dirs) or \
                          any('fov' in d.lower() for d in top_level_dirs)
    
    if has_acquisition_folders and not has_fluorophore_dirs:
        # We're at the level that directly contains acquisition folders (like mCherry_mouse or fig2 after restructure)
        # Reuse the existing process_directory_with_reference function
        logger.info(f"Detected acquisition-level directory: processing acquisition folders directly")
        
        try:
            # Find 1080 reference
            ref_path, ref_info = find_1080_reference(data_dir_path)
        except ValueError as e:
            logger.error(f"No 1080nm reference found in {data_dir_path}: {e}")
            return all_results
        
        # Process using the existing function
        output_dir = data_dir_path
        ref_image_temp, ref_is_stack_temp = load_image_or_stack(ref_path)
        
        results_df = process_directory_with_reference(
            data_dir_path, ref_path, ref_info, output_dir,
            align_xy_first=align_xy_first, preserve_metadata=preserve_metadata
        )
        
        # Save CSV if we have stack data
        if len(results_df) > 0:
            csv_path = os.path.join(output_dir, 'z_plane_selections.csv')
            if 'reference_z' in results_df.columns and 'selected_z' in results_df.columns:
                pivot_df = results_df.pivot_table(
                    index='acquisition',
                    columns='reference_z',
                    values='selected_z',
                    aggfunc='first'
                )
                if ref_is_stack_temp:
                    target_z_planes = extract_target_z_planes(ref_image_temp, every_nth=10,
                                                              exclude_edges=True)
                    ref_data = {z: z for z in target_z_planes}
                    ref_row = pd.Series(ref_data, name=ref_info['subdir'])
                    for z in target_z_planes:
                        if z not in pivot_df.columns:
                            pivot_df[z] = None
                    pivot_df = pd.concat([pd.DataFrame([ref_row]), pivot_df])
                pivot_df.to_csv(csv_path)
            else:
                results_df.to_csv(csv_path, index=False)
            logger.info(f"Saved Z plane selections to {csv_path}")
        
        all_results[os.path.basename(data_dir_path)] = results_df
        return all_results
    
    elif has_fluorophore_dirs:
        # We're at dataset level with fluorophore/FOV subdirectories - process those
        # Skip: acquisition folders, junk folders, hidden folders, References folders
        skip_patterns = ['EmFilt_', 'nm_', '_poc', 'junk', '.', 'References']
        fluorophore_dirs = [d for d in top_level_dirs
                           if os.path.isdir(os.path.join(data_dir_path, d))
                           and not any(x.lower() in d.lower() for x in skip_patterns)]
        
        for subdir in fluorophore_dirs:
            subdir_path = os.path.join(data_dir_path, subdir)
            
            try:
                logger.info(f"Looking for 1080nm reference in: {subdir_path}")
                # Find 1080 reference
                try:
                    ref_path, ref_info = find_1080_reference(subdir_path)
                except ValueError as e:
                    logger.warning(f"No 1080nm reference found in {subdir_path}: {e}")
                    logger.warning(f"  Skipping this directory.")
                    continue
                
                # Output goes directly in the fluorophore directory
                output_dir = subdir_path
                
                logger.info(f"Processing {subdir_path}")
                
                # Load reference to check if it's a stack
                ref_image_temp, ref_is_stack_temp = load_image_or_stack(ref_path)
                
                # Process directory
                results_df = process_directory_with_reference(
                    subdir_path, ref_path, ref_info, output_dir,
                    align_xy_first=align_xy_first, preserve_metadata=preserve_metadata
                )
                
                # Save CSV with Z plane selections if we have stack data
                if len(results_df) > 0:
                    csv_path = os.path.join(output_dir, 'z_plane_selections.csv')
                    # If we have reference_z and selected_z columns, create pivot table
                    if 'reference_z' in results_df.columns and 'selected_z' in results_df.columns:
                        # Create pivot: acquisitions as rows, reference Z planes as columns
                        pivot_df = results_df.pivot_table(
                            index='acquisition',
                            columns='reference_z',
                            values='selected_z',
                            aggfunc='first'
                        )
                        # Add reference row showing which Z planes were used (10, 20, 30, etc.)
                        if ref_is_stack_temp:
                            target_z_planes = extract_target_z_planes(ref_image_temp, every_nth=10,
                                                                      exclude_edges=True)
                            # Create a row with reference Z planes as values
                            ref_data = {z: z for z in target_z_planes}
                            ref_row = pd.Series(ref_data, name=ref_info['subdir'])
                            # Ensure all columns exist
                            for z in target_z_planes:
                                if z not in pivot_df.columns:
                                    pivot_df[z] = None
                            pivot_df = pd.concat([pd.DataFrame([ref_row]), pivot_df])
                        pivot_df.to_csv(csv_path)
                    else:
                        results_df.to_csv(csv_path, index=False)
                    logger.info(f"Saved Z plane selections to {csv_path}")
                
                all_results[subdir] = results_df
                
            except ValueError as e:
                logger.error(f"Error processing {subdir_path}: {e}")
                logger.error(f"  This directory will be skipped. Make sure it contains a 1080nm reference image.")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {subdir_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        return all_results
    
    # Otherwise, we're at data/ level - iterate through dataset directories
    for top_dir in top_level_dirs:
        top_dir_path = os.path.join(data_dir_path, top_dir)
        
        # Find fluorophore/mouse/FOV directories
        subdirs = [d for d in os.listdir(top_dir_path)
                  if os.path.isdir(os.path.join(top_dir_path, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(top_dir_path, subdir)
            
            # Skip if this looks like an acquisition subdirectory (has filter and wavelength in name)
            # We only want to process parent directories like "mCherry_mouse", "mNeptune_mouse", "FOV1", etc.
            # Acquisition folders have patterns like "BR2EmFilt_1080nm_230poc_z13_580pmt-007"
            # Also skip junk folders and other non-data directories
            skip_patterns = ['EmFilt_', 'nm_', '_poc', 'junk', '.', 'References']
            if any(x.lower() in subdir.lower() for x in skip_patterns):
                logger.debug(f"Skipping subdirectory (will be processed as part of parent or is non-data): {subdir}")
                continue
            
            try:
                logger.info(f"Looking for 1080nm reference in: {subdir_path}")
                # Find 1080 reference
                try:
                    ref_path, ref_info = find_1080_reference(subdir_path)
                except ValueError as e:
                    logger.warning(f"No 1080nm reference found in {subdir_path}: {e}")
                    logger.warning(f"  Skipping this directory.")
                    continue
                
                # Output goes directly in the fluorophore directory (one level up from subdirectories)
                # e.g., data/fig1/.../mCherry_mouse/ (not in BR2EmFilt_... subdirectory)
                output_dir = subdir_path
                
                logger.info(f"Processing {subdir_path}")
                
                # Load reference to check if it's a stack
                ref_image_temp, ref_is_stack_temp = load_image_or_stack(ref_path)
                
                # Process directory
                results_df = process_directory_with_reference(
                    subdir_path, ref_path, ref_info, output_dir,
                    align_xy_first=align_xy_first, preserve_metadata=preserve_metadata
                )
                
                # Save CSV with Z plane selections if we have stack data
                if len(results_df) > 0:
                    csv_path = os.path.join(output_dir, 'z_plane_selections.csv')
                    # If we have reference_z and selected_z columns, create pivot table
                    if 'reference_z' in results_df.columns and 'selected_z' in results_df.columns:
                        # Create pivot: acquisitions as rows, reference Z planes as columns
                        pivot_df = results_df.pivot_table(
                            index='acquisition',
                            columns='reference_z',
                            values='selected_z',
                            aggfunc='first'
                        )
                        # Add reference row showing which Z planes were used (10, 20, 30, etc.)
                        if ref_is_stack_temp:
                            target_z_planes = extract_target_z_planes(ref_image_temp, every_nth=10,
                                                                      exclude_edges=True)
                            # Create a row with reference Z planes as values
                            ref_data = {z: z for z in target_z_planes}
                            ref_row = pd.Series(ref_data, name=ref_info['subdir'])
                            # Ensure all columns exist
                            for z in target_z_planes:
                                if z not in pivot_df.columns:
                                    pivot_df[z] = None
                            pivot_df = pd.concat([pd.DataFrame([ref_row]), pivot_df])
                        pivot_df.to_csv(csv_path)
                    else:
                        results_df.to_csv(csv_path, index=False)
                    logger.info(f"Saved Z plane selections to {csv_path}")
                
                all_results[subdir] = results_df
                
            except ValueError as e:
                logger.error(f"Error processing {subdir_path}: {e}")
                logger.error(f"  This directory will be skipped. Make sure it contains a 1080nm reference image.")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {subdir_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
    
    return all_results

