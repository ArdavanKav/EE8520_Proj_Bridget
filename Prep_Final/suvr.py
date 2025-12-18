#!/usr/bin/env python3
"""SUVR computation."""

import numpy as np
import nibabel as nib
from pathlib import Path


def compute_suvr(pet_path, mask_path, output_path, apply_mask=False):
    """Compute SUVR using cerebellar gray matter reference.
    
    Args:
        pet_path: Path to PET image
        mask_path: Path to brain mask (used for reference region only)
        output_path: Output path for SUVR image
        apply_mask: If True, apply brain mask to output. Default False.
    """
    
    # Load PET
    pet_img = nib.load(pet_path)
    pet_data = pet_img.get_fdata()
    
    # Load brain mask for reference region
    brain_mask = None
    if mask_path and Path(mask_path).exists():
        mask_img = nib.load(mask_path)
        brain_mask = mask_img.get_fdata() > 0
    
    # Create cerebellar mask (MNI coordinates)
    cerebellum_mni = {
        'left': {'x': (10, 30), 'y': (100, 130), 'z': (30, 60)},
        'right': {'x': (152, 172), 'y': (100, 130), 'z': (30, 60)}
    }
    
    cereb_mask = np.zeros(pet_data.shape, dtype=bool)
    for region in cerebellum_mni.values():
        cereb_mask[
            region['x'][0]:region['x'][1],
            region['y'][0]:region['y'][1],
            region['z'][0]:region['z'][1]
        ] = True
    
    # Intersect with brain mask for reference region
    if brain_mask is not None:
        cereb_mask = cereb_mask & brain_mask
    
    # Get reference value
    cereb_values = pet_data[cereb_mask]
    if len(cereb_values) == 0:
        raise ValueError("No cerebellar voxels found")
    
    ref_value = np.mean(cereb_values)
    
    # Compute SUVR (no masking - keep all voxels)
    suvr_data = pet_data / ref_value
    
    # Only apply mask if requested
    if apply_mask and brain_mask is not None:
        suvr_data[~brain_mask] = 0
    
    # Save
    nib.save(nib.Nifti1Image(suvr_data.astype(np.float32), pet_img.affine), output_path)
    
    # Calculate global cortical SUVR for QC
    if brain_mask is not None:
        cortical_mask = brain_mask & (pet_data > 0)
        global_suvr = np.mean(suvr_data[cortical_mask])
    else:
        global_suvr = np.mean(suvr_data[pet_data > 0])
    
    return {
        'global_suvr': float(global_suvr),
        'ref_value': float(ref_value)
    }
