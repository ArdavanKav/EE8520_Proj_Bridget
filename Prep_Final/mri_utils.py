#!/usr/bin/env python3
"""MRI preprocessing utilities."""

import os
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage


def run_fsl(cmd, description=""):
    """Run FSL command."""
    fsldir = os.environ.get('FSLDIR', '/home/ardkav/fsl')
    env = os.environ.copy()
    env['FSLDIR'] = fsldir
    env['PATH'] = f"{fsldir}/bin:{env.get('PATH', '')}"
    env['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
    
    if not cmd[0].startswith('/'):
        cmd[0] = f"{fsldir}/bin/{cmd[0]}"
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed: {result.stderr}")
    return result


def skull_strip(input_path, output_dir, bet_frac=0.35):
    """Skull strip MRI using BET."""
    output_dir = Path(output_dir)
    brain_path = output_dir / "brain.nii.gz"
    mask_path = output_dir / "brain_mask.nii.gz"
    
    # BET with solid mask
    run_fsl(['bet', str(input_path), str(brain_path), '-f', str(bet_frac), 
             '-B', '-m'], "BET skull stripping")
    
    # Load and solidify mask
    img = nib.load(input_path)
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata() > 0
    
    # Fill holes
    mask_filled = ndimage.binary_fill_holes(mask_data)
    
    # Smooth mask to reduce sharp edges
    mask_smoothed = ndimage.gaussian_filter(mask_filled.astype(np.float32), sigma=1.5)
    mask_smoothed = mask_smoothed > 0.5
    
    # Apply to brain
    brain_data = img.get_fdata()
    brain_data[~mask_smoothed] = 0
    
    # Save
    nib.save(nib.Nifti1Image(brain_data.astype(np.float32), img.affine), brain_path)
    nib.save(nib.Nifti1Image(mask_smoothed.astype(np.float32), mask_img.affine), mask_path)
    
    # Calculate volume
    voxel_vol = np.abs(np.linalg.det(img.affine[:3, :3]))
    volume_ml = np.sum(mask_smoothed) * voxel_vol / 1000
    
    # Bias correction
    brain_bc_path = output_dir / "brain_bc.nii.gz"
    run_fsl(['fast', '-B', '-o', str(output_dir / "fast"), str(brain_path)], 
            "Bias correction")
    
    # Rename output
    restored = output_dir / "fast_restore.nii.gz"
    if restored.exists():
        restored.rename(brain_bc_path)
    
    return {
        'brain': str(brain_path),
        'brain_bc': str(brain_bc_path),
        'mask': str(mask_path),
        'volume_ml': volume_ml
    }
