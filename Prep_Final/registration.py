#!/usr/bin/env python3
"""Registration utilities."""

import numpy as np
import ants
from pathlib import Path
from scipy import ndimage


def register_to_mni(brain_path, mask_path, template_path, output_dir):
    """Register brain to MNI space."""
    output_dir = Path(output_dir)
    
    # Load images
    brain_img = ants.image_read(brain_path)
    template_img = ants.image_read(template_path)
    
    # Register
    reg = ants.registration(
        fixed=template_img,
        moving=brain_img,
        type_of_transform='SyN',
        syn_metric='mattes',
        syn_sampling=32,
        reg_iterations=(100, 70, 50, 20),
        verbose=False
    )
    
    # Save registered image
    mni_path = output_dir / "mri_mni.nii.gz"
    ants.image_write(reg['warpedmovout'], str(mni_path))
    
    # Calculate correlation
    warped_data = reg['warpedmovout'].numpy()
    template_data = template_img.numpy()
    mask = (warped_data > 0) & (template_data > 0)
    
    if np.sum(mask) > 0:
        corr = np.corrcoef(warped_data[mask].flatten(), template_data[mask].flatten())[0, 1]
    else:
        corr = 0.0
    
    # Transform mask to MNI
    mask_img = ants.image_read(mask_path)
    mask_mni = ants.apply_transforms(
        fixed=template_img,
        moving=mask_img,
        transformlist=reg['fwdtransforms'],
        interpolator='nearestNeighbor'
    )
    
    # Fill holes in mask
    mask_data = mask_mni.numpy() > 0.5
    mask_filled = ndimage.binary_fill_holes(mask_data)
    mask_mni = ants.from_numpy(
        mask_filled.astype(np.float32),
        origin=mask_mni.origin,
        spacing=mask_mni.spacing,
        direction=mask_mni.direction
    )
    
    mask_mni_path = output_dir / "brain_mask_mni.nii.gz"
    ants.image_write(mask_mni, str(mask_mni_path))
    
    return {
        'mri_mni': str(mni_path),
        'mask_mni': str(mask_mni_path),
        'transforms': reg['fwdtransforms'],
        'correlation': corr
    }


def apply_transform_to_pet(pet_path, transforms, template_path, output_path):
    """Apply MRI-to-MNI transforms to PET."""
    pet_img = ants.image_read(pet_path)
    template_img = ants.image_read(template_path)
    
    pet_mni = ants.apply_transforms(
        fixed=template_img,
        moving=pet_img,
        transformlist=transforms,
        interpolator='linear'
    )
    
    ants.image_write(pet_mni, output_path)
