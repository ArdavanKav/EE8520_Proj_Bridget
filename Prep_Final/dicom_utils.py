"""DICOM conversion."""

import subprocess
from pathlib import Path


def convert_dicom_to_nifti(dicom_dir, output_path):
    """Convert DICOM to NIfTI."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'dcm2niix',
        '-z', 'y',
        '-f', output_path.stem,
        '-o', str(output_path.parent),
        str(dicom_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"DICOM conversion failed: {result.stderr}")
    
    # Find the actual output file
    nii_files = list(output_path.parent.glob(f"{output_path.stem}*.nii.gz"))
    
    if not nii_files:
        raise RuntimeError("No NIfTI file created")
    
    # Rename to expected output
    if nii_files[0] != output_path:
        nii_files[0].rename(output_path)
    
    return str(output_path)
