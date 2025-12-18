#!/usr/bin/env python3
"""
Visualization script for final preprocessed MRI and PET outputs.
Creates 3-row visualization: MRI slices, split view, PET slices.
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def load_nifti(filepath):
    """Load NIfTI file and return data array."""
    img = nib.load(filepath)
    return img.get_fdata()

def create_split_image(mri_slice, pet_slice):
    """Create a split image: right half MRI, left half PET."""
    h, w = mri_slice.shape
    
    # Normalize MRI to 0-1
    mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
    
    # Normalize PET to 0-1 (already should be, but ensure)
    pet_norm = np.clip(pet_slice, 0, 1)
    
    # Convert MRI to RGB (grayscale)
    mri_rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)
    
    # Convert PET to RGB using jet colormap (blue→cyan→green→yellow→red)
    cmap = plt.cm.jet
    pet_rgb = cmap(pet_norm)[:, :, :3]  # Remove alpha channel
    
    # Create split image: left half PET, right half MRI
    split_img = np.zeros((h, w, 3))
    mid = w // 2
    split_img[:, :mid, :] = pet_rgb[:, :mid, :]   # Left half: PET
    split_img[:, mid:, :] = mri_rgb[:, mid:, :]   # Right half: MRI
    
    return split_img

def create_visualization(mri_data, pet_data, output_path, subject_id, session_date):
    """Create 3-row visualization with 5 axial slices per row."""
    
    # Get 5 evenly spaced axial slice indices
    n_slices = 5
    z_dim = mri_data.shape[2]
    # Skip edge slices, take from 20% to 80% of the volume
    start_idx = int(z_dim * 0.2)
    end_idx = int(z_dim * 0.8)
    slice_indices = np.linspace(start_idx, end_idx, n_slices, dtype=int)
    
    # Create figure with 3 rows x 5 columns
    fig, axes = plt.subplots(3, n_slices, figsize=(15, 10))
    
    # Normalize MRI globally for consistent display
    mri_min, mri_max = mri_data.min(), mri_data.max()
    
    for col, z_idx in enumerate(slice_indices):
        # Extract axial slices
        mri_slice = np.rot90(mri_data[:, :, z_idx])
        pet_slice = np.rot90(pet_data[:, :, z_idx])
        
        # Normalize MRI slice
        mri_norm = (mri_slice - mri_min) / (mri_max - mri_min + 1e-8)
        
        # Row 1: MRI (grayscale)
        axes[0, col].imshow(mri_norm, cmap='gray', aspect='equal', vmin=0, vmax=1)
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel('MRI', fontsize=14, fontweight='bold')
        axes[0, col].set_title(f'Slice {z_idx}', fontsize=10)
        
        # Row 2: Split view (left=PET, right=MRI)
        split_img = create_split_image(mri_slice, pet_slice)
        axes[1, col].imshow(split_img, aspect='equal')
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_ylabel('PET | MRI', fontsize=14, fontweight='bold')
        
        # Row 3: PET (jet colormap - blue→cyan→green→yellow→red)
        axes[2, col].imshow(pet_slice, cmap='jet', aspect='equal', vmin=0, vmax=1)
        axes[2, col].axis('off')
        if col == 0:
            axes[2, col].set_ylabel('PET', fontsize=14, fontweight='bold')
    
    # Add row labels on the left
    fig.text(0.02, 0.78, 'MRI', ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)
    fig.text(0.02, 0.50, 'PET | MRI', ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)
    fig.text(0.02, 0.22, 'PET', ha='center', va='center', fontsize=14, fontweight='bold', rotation=90, color='orangered')
    
    # Add title
    fig.suptitle(f'Subject: {subject_id} | Session: {session_date}\nFinal Preprocessed Output (80×80×80) - Axial Views',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0.04, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Visualization saved to: {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_final.py <subject_id> <session_date>")
        print("Example: python visualize_final.py MCSA_00003 1953-12-31")
        sys.exit(1)
    
    subject_id = sys.argv[1]
    session_date = sys.argv[2]
    
    # Define paths
    base_dir = "/data/shared_data/Alz/MCSA/processed"
    session_dir = os.path.join(base_dir, subject_id, session_date)
    
    mri_path = os.path.join(session_dir, "MRI_final.nii.gz")
    pet_path = os.path.join(session_dir, "PET_final.nii.gz")
    output_path = os.path.join(session_dir, "visualization_final.png")
    
    # Check if files exist
    if not os.path.exists(mri_path):
        print(f"Error: MRI file not found: {mri_path}")
        sys.exit(1)
    if not os.path.exists(pet_path):
        print(f"Error: PET file not found: {pet_path}")
        sys.exit(1)
    
    # Load data
    print(f"Loading MRI: {mri_path}")
    mri_data = load_nifti(mri_path)
    print(f"  Shape: {mri_data.shape}")
    
    print(f"Loading PET: {pet_path}")
    pet_data = load_nifti(pet_path)
    print(f"  Shape: {pet_data.shape}")
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(mri_data, pet_data, output_path, subject_id, session_date)

if __name__ == "__main__":
    main()
