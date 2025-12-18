#!/usr/bin/env python3
"""PET preprocessing utilities."""

import os
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path


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


def motion_correct(input_path, output_dir):
    """Motion correct PET frames."""
    output_dir = Path(output_dir)
    pet_img = nib.load(input_path)
    pet_data = pet_img.get_fdata()
    
    n_frames = pet_data.shape[3] if len(pet_data.shape) == 4 else 1
    
    if n_frames > 1:
        mc_output = output_dir / "pet_mc.nii.gz"
        run_fsl(['mcflirt', '-in', str(input_path), '-out', str(mc_output),
                 '-plots'], "Motion correction")
        
        # Get motion params
        params_file = output_dir / "pet_mc.par"
        if params_file.exists():
            params = np.loadtxt(params_file)
            max_motion = np.max(np.abs(params[:, 3:6]))
        else:
            max_motion = 0.0
            
        pet_img = nib.load(mc_output)
        pet_data = pet_img.get_fdata()
    else:
        max_motion = 0.0
    
    # Average frames
    avg_data = np.mean(pet_data, axis=3) if len(pet_data.shape) == 4 else pet_data
    avg_path = output_dir / "pet_avg.nii.gz"
    nib.save(nib.Nifti1Image(avg_data.astype(np.float32), pet_img.affine), avg_path)
    
    return {
        'avg': str(avg_path),
        'max_motion_mm': max_motion,
        'n_frames': n_frames
    }


def coregister_to_mri(pet_path, mri_path, output_path):
    """Coregister PET to MRI."""
    import ants
    
    pet_img = ants.image_read(pet_path)
    mri_img = ants.image_read(mri_path)
    
    reg = ants.registration(
        fixed=mri_img,
        moving=pet_img,
        type_of_transform='Rigid',
        verbose=False
    )
    
    ants.image_write(reg['warpedmovout'], output_path)
    
    # Calculate correlation
    pet_data = reg['warpedmovout'].numpy()
    mri_data = mri_img.numpy()
    mask = (pet_data > 0) & (mri_data > 0)
    
    if np.sum(mask) > 0:
        corr = np.corrcoef(pet_data[mask].flatten(), mri_data[mask].flatten())[0, 1]
    else:
        corr = 0.0
    
    return {'correlation': corr}
