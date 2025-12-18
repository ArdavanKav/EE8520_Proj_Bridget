#!/usr/bin/env python3
"""Final preparation utilities for model input."""

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import zoom


def resample_to_spacing(img_path, output_path, target_spacing=(1.5, 1.5, 1.5), order=3, smooth_mask=False):
    """Resample image to target spacing.
    
    Args:
        order: Interpolation order. Use 0 for masks, 3 for images.
        smooth_mask: If True, apply slight smoothing to mask edges.
    """
    img = nib.load(img_path)
    data = img.get_fdata()
    affine = img.affine.copy()
    
    # Current spacing
    current_spacing = np.abs(np.diag(affine[:3, :3]))
    
    # Zoom factors
    zoom_factors = current_spacing / np.array(target_spacing)
    
    # Resample
    resampled = zoom(data, zoom_factors, order=order)
    
    # For masks, optionally smooth then threshold
    if order == 0:
        if smooth_mask:
            resampled = ndimage.gaussian_filter(resampled.astype(np.float32), sigma=0.8)
        resampled = (resampled > 0.5).astype(np.float32)
    
    # Update affine
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = np.sign(affine[i, i]) * target_spacing[i]
    
    # Save
    nib.save(nib.Nifti1Image(resampled.astype(np.float32), new_affine), output_path)
    
    return output_path


def resample_to_grid(img_path, output_path, target_shape=(160, 160, 96), target_spacing=(1.5, 1.5, 1.5)):
    """Resample image to specific grid size and spacing."""
    img = nib.load(img_path)
    data = img.get_fdata()
    affine = img.affine.copy()
    
    # Current spacing and shape
    current_spacing = np.abs(np.diag(affine[:3, :3]))
    current_shape = np.array(data.shape[:3])
    
    # Calculate zoom to get target spacing, then resize to target shape
    spacing_zoom = current_spacing / np.array(target_spacing)
    intermediate_shape = (current_shape * spacing_zoom).astype(int)
    
    # First zoom for spacing
    temp = zoom(data, spacing_zoom, order=3)
    
    # Then resize to target shape
    shape_zoom = np.array(target_shape) / np.array(temp.shape[:3])
    resampled = zoom(temp, shape_zoom, order=3)
    
    # Update affine for new spacing
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = np.sign(affine[i, i]) * target_spacing[i]
    
    # Center the new image
    old_center = affine[:3, 3] + (current_shape * current_spacing) / 2
    new_center = old_center
    new_affine[:3, 3] = new_center - (np.array(target_shape) * np.array(target_spacing)) / 2
    
    nib.save(nib.Nifti1Image(resampled.astype(np.float32), new_affine), output_path)
    
    return output_path


def apply_gaussian_smoothing(img_path, output_path, fwhm_mm=8.0):
    """Apply Gaussian smoothing for scanner harmonization."""
    img = nib.load(img_path)
    data = img.get_fdata()
    affine = img.affine
    
    # Convert FWHM to sigma in voxels
    spacing = np.abs(np.diag(affine[:3, :3]))
    sigma_voxels = (fwhm_mm / 2.355) / spacing
    
    # Apply smoothing
    smoothed = ndimage.gaussian_filter(data, sigma=sigma_voxels)
    
    nib.save(nib.Nifti1Image(smoothed.astype(np.float32), affine), output_path)
    
    return output_path


def normalize_suvr_0_to_1(img_path, mask_path, output_path):
    """Normalize SUVR values to 0-1 range within brain mask."""
    img = nib.load(img_path)
    data = img.get_fdata()
    
    mask = nib.load(mask_path).get_fdata() > 0
    
    # Get brain values
    brain_values = data[mask]
    
    # Normalize to 0-1
    vmin = np.percentile(brain_values, 1)
    vmax = np.percentile(brain_values, 99)
    
    normalized = np.clip((data - vmin) / (vmax - vmin + 1e-8), 0, 1)
    normalized[~mask] = 0
    
    nib.save(nib.Nifti1Image(normalized.astype(np.float32), img.affine), output_path)
    
    return output_path


def normalize_suvr_0_to_1_no_mask(img_path, mask_path, output_path):
    """Normalize SUVR values to 0-1 range using brain values but keep all voxels."""
    img = nib.load(img_path)
    data = img.get_fdata()
    
    mask = nib.load(mask_path).get_fdata() > 0
    
    # Get brain values for normalization parameters
    brain_values = data[mask]
    
    # Normalize to 0-1 using brain statistics
    vmin = np.percentile(brain_values, 1)
    vmax = np.percentile(brain_values, 99)
    
    normalized = np.clip((data - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    nib.save(nib.Nifti1Image(normalized.astype(np.float32), img.affine), output_path)
    
    return output_path


def apply_brain_mask(img_path, mask_path, output_path):
    """Apply brain mask to image."""
    img = nib.load(img_path)
    data = img.get_fdata()
    
    mask = nib.load(mask_path).get_fdata() > 0
    
    data[~mask] = 0
    
    nib.save(nib.Nifti1Image(data.astype(np.float32), img.affine), output_path)
    
    return output_path


def crop_to_size(img_path, output_path, target_shape=(120, 128, 120)):
    """
    Crop image from (121, 145, 121) to target shape.
    
    Cropping strategy for (121, 145, 121) -> (120, 128, 120):
    - Dim 0: 121 -> 120: crop 1 pixel from end
    - Dim 1: 145 -> 128: crop 8 from start, 9 from end (total 17)
    - Dim 2: 121 -> 120: crop 1 pixel from end
    """
    img = nib.load(img_path)
    data = img.get_fdata()
    affine = img.affine.copy()
    
    current_shape = data.shape
    
    # Calculate crop amounts for each dimension
    crops = []
    for i in range(3):
        diff = current_shape[i] - target_shape[i]
        if diff > 0:
            start = diff // 2
            end = current_shape[i] - (diff - start)
            crops.append(slice(start, end))
        elif diff < 0:
            # Need padding, not cropping
            crops.append(slice(0, current_shape[i]))
        else:
            crops.append(slice(0, current_shape[i]))
    
    cropped = data[crops[0], crops[1], crops[2]]
    
    # Pad if any dimension is smaller
    if cropped.shape != target_shape:
        padded = np.zeros(target_shape, dtype=np.float32)
        for i in range(3):
            if cropped.shape[i] < target_shape[i]:
                pad_start = (target_shape[i] - cropped.shape[i]) // 2
                # This would need more complex slicing
        # For now, just use what we have
        padded[:cropped.shape[0], :cropped.shape[1], :cropped.shape[2]] = cropped
        cropped = padded
    
    nib.save(nib.Nifti1Image(cropped.astype(np.float32), affine), output_path)
    
    return output_path


def resize_to_shape(img_path, output_path, target_shape=(80, 80, 80)):
    """Resize image to target shape."""
    img = nib.load(img_path)
    data = img.get_fdata()
    affine = img.affine.copy()
    
    current_shape = np.array(data.shape)
    zoom_factors = np.array(target_shape) / current_shape
    
    resized = zoom(data, zoom_factors, order=3)
    
    # Clip negative values (can occur at boundaries from cubic interpolation)
    resized = np.clip(resized, 0, None)
    
    # Update affine for new spacing
    current_spacing = np.abs(np.diag(affine[:3, :3]))
    new_spacing = current_spacing * current_shape / np.array(target_shape)
    
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = np.sign(affine[i, i]) * new_spacing[i]
    
    nib.save(nib.Nifti1Image(resized.astype(np.float32), new_affine), output_path)
    
    return output_path


def resize_mask_to_shape(mask_path, output_path, target_shape=(80, 80, 80)):
    """Resize binary mask to target shape using nearest neighbor interpolation."""
    img = nib.load(mask_path)
    data = img.get_fdata()
    affine = img.affine.copy()
    
    current_shape = np.array(data.shape)
    zoom_factors = np.array(target_shape) / current_shape
    
    # Use order=0 (nearest neighbor) for binary mask
    resized = zoom(data, zoom_factors, order=0)
    
    # Ensure binary
    resized = (resized > 0.5).astype(np.float32)
    
    # Update affine for new spacing
    current_spacing = np.abs(np.diag(affine[:3, :3]))
    new_spacing = current_spacing * current_shape / np.array(target_shape)
    
    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = np.sign(affine[i, i]) * new_spacing[i]
    
    nib.save(nib.Nifti1Image(resized, new_affine), output_path)
    
    return output_path
