#!/usr/bin/env python3
"""Main preprocessing pipeline with parallel processing support."""

import os
import sys
import yaml
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import dicom_utils
import mri_utils
import pet_utils
import registration
import suvr
import final_prep


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_sessions(mcsa_root):
    """Find matched MRI-PET sessions."""
    mcsa_root = Path(mcsa_root)
    
    sessions = []
    for subject_dir in sorted(mcsa_root.glob("MCSA_*")):
        subject_id = subject_dir.name
        
        # Find MRI sessions
        mri_dir = subject_dir / "Sag_3D_MP-RAGE"
        if not mri_dir.exists():
            continue
        
        mri_sessions = {}
        for session_dir in mri_dir.iterdir():
            if session_dir.is_dir():
                date_str = session_dir.name.split('_')[0]
                dicom_dirs = [d for d in session_dir.iterdir() if d.is_dir()]
                if dicom_dirs:
                    mri_sessions[date_str] = str(dicom_dirs[0])
        
        # Find PET sessions
        pet_dir = subject_dir / "TR_BRAIN_3D_PIB_IR_CTAC"
        if not pet_dir.exists():
            continue
        
        pet_sessions = {}
        for session_dir in pet_dir.iterdir():
            if session_dir.is_dir():
                date_str = session_dir.name.split('_')[0]
                dicom_dirs = [d for d in session_dir.iterdir() if d.is_dir()]
                if dicom_dirs:
                    pet_sessions[date_str] = str(dicom_dirs[0])
        
        # Match by date
        for date_str in mri_sessions:
            if date_str in pet_sessions:
                sessions.append({
                    'subject_id': subject_id,
                    'session_date': date_str,
                    'mri': mri_sessions[date_str],
                    'pet': pet_sessions[date_str]
                })
    
    return sessions


def process_session(session, config, template_dir):
    """Process single session."""
    logger = logging.getLogger(__name__)
    
    subject_id = session['subject_id']
    session_date = session['session_date']
    
    logger.info(f"Processing {subject_id} / {session_date}")
    
    # Setup directories
    output_dir = Path(config['data']['output_dir']) / subject_id / session_date
    work_dir = Path(f"/tmp/prep_{subject_id}_{session_date}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 0: Convert DICOM to NIfTI
        logger.info("  [0/6] Converting DICOM...")
        mri_nii = work_dir / "mri.nii.gz"
        pet_nii = work_dir / "pet.nii.gz"
        
        dicom_utils.convert_dicom_to_nifti(session['mri'], mri_nii)
        dicom_utils.convert_dicom_to_nifti(session['pet'], pet_nii)
        
        # Step 1: MRI skull stripping
        logger.info("  [1/6] Skull stripping...")
        mri_results = mri_utils.skull_strip(
            str(mri_nii),
            work_dir,
            config['processing']['bet_frac']
        )
        
        if mri_results['volume_ml'] < config['qc']['min_brain_volume_ml']:
            raise ValueError(f"Brain volume too small: {mri_results['volume_ml']:.1f} ml")
        
        # Step 2: Register to MNI
        logger.info("  [2/6] Registering to MNI...")
        template_path = template_dir / config['processing']['mni_template']
        
        reg_results = registration.register_to_mni(
            mri_results['brain_bc'],
            mri_results['mask'],
            str(template_path),
            work_dir
        )
        
        if reg_results['correlation'] < config['qc']['min_correlation']:
            raise ValueError(f"Poor registration: {reg_results['correlation']:.3f}")
        
        # Step 3: PET motion correction
        logger.info("  [3/6] Motion correction...")
        pet_results = pet_utils.motion_correct(str(pet_nii), work_dir)
        
        if pet_results['max_motion_mm'] > config['qc']['max_motion_mm']:
            logger.warning(f"High motion: {pet_results['max_motion_mm']:.2f} mm")
        
        # Step 4: Coregister PET to MRI
        logger.info("  [4/6] Coregistering PET to MRI...")
        pet_coreg_path = work_dir / "pet_coreg.nii.gz"
        
        coreg_results = pet_utils.coregister_to_mri(
            pet_results['avg'],
            mri_results['brain'],
            str(pet_coreg_path)
        )
        
        # Step 5: Transform PET to MNI
        logger.info("  [5/6] Transforming PET to MNI...")
        pet_mni_path = work_dir / "pet_mni.nii.gz"
        
        registration.apply_transform_to_pet(
            str(pet_coreg_path),
            reg_results['transforms'],
            str(template_path),
            str(pet_mni_path)
        )
        
        # Step 6: Compute SUVR
        logger.info("  [6/12] Computing SUVR...")
        suvr_path = work_dir / "pet_suvr_mni.nii.gz"
        
        suvr_results = suvr.compute_suvr(
            str(pet_mni_path),
            reg_results['mask_mni'],
            str(suvr_path)
        )
        
        # Step 7: Resample MRI to 1.5mm
        logger.info("  [7/12] Resampling MRI to 1.5mm...")
        mri_15mm_path = work_dir / "mri_mni_1.5mm.nii.gz"
        final_prep.resample_to_spacing(
            reg_results['mri_mni'],
            str(mri_15mm_path),
            target_spacing=(1.5, 1.5, 1.5)
        )
        
        # Step 8: Resample PET to 1.5mm voxel size
        logger.info("  [8/10] Resampling PET to 1.5mm...")
        pet_15mm_path = work_dir / "pet_suvr_1.5mm.nii.gz"
        final_prep.resample_to_spacing(
            str(suvr_path),
            str(pet_15mm_path),
            target_spacing=(1.5, 1.5, 1.5)
        )
        
        # Step 9: Apply 4mm FWHM smoothing to PET
        logger.info("  [9/11] Applying 4mm FWHM smoothing...")
        pet_smooth_path = work_dir / "pet_suvr_4mm.nii.gz"
        final_prep.apply_gaussian_smoothing(
            str(pet_15mm_path),
            str(pet_smooth_path),
            fwhm_mm=4.0
        )
        
        # Step 10: Normalize SUVR to 0-1 (without masking)
        logger.info("  [10/11] Normalizing SUVR to 0-1...")
        # Resample mask to match PET (use nearest neighbor with smoothing)
        mask_15mm_path = work_dir / "mask_1.5mm.nii.gz"
        final_prep.resample_to_spacing(
            reg_results['mask_mni'],
            str(mask_15mm_path),
            target_spacing=(1.5, 1.5, 1.5),
            order=0,
            smooth_mask=True  # Smooth mask edges
        )
        
        pet_norm_path = work_dir / "pet_suvr_norm_unmasked.nii.gz"
        final_prep.normalize_suvr_0_to_1_no_mask(
            str(pet_smooth_path),
            str(mask_15mm_path),
            str(pet_norm_path)
        )
        
        # Step 11: Apply brain mask to both MRI and PET (before cropping)
        logger.info("  [11/14] Applying brain mask...")
        mri_masked_path = work_dir / "mri_masked.nii.gz"
        final_prep.apply_brain_mask(
            str(mri_15mm_path),
            str(mask_15mm_path),
            str(mri_masked_path)
        )
        
        pet_masked_path = work_dir / "pet_masked.nii.gz"
        final_prep.apply_brain_mask(
            str(pet_norm_path),
            str(mask_15mm_path),
            str(pet_masked_path)
        )
        
        # Step 12: Crop to (120, 128, 120)
        logger.info("  [12/14] Cropping to 120x128x120...")
        mri_cropped_path = work_dir / "mri_cropped.nii.gz"
        final_prep.crop_to_size(
            str(mri_masked_path),
            str(mri_cropped_path),
            target_shape=(120, 128, 120)
        )
        
        pet_cropped_path = work_dir / "pet_cropped.nii.gz"
        final_prep.crop_to_size(
            str(pet_masked_path),
            str(pet_cropped_path),
            target_shape=(120, 128, 120)
        )
        
        # Step 13: Resize MRI to 80x80x80
        logger.info("  [13/14] Resizing MRI to 80x80x80...")
        mri_final_path = work_dir / "MRI_final.nii.gz"
        final_prep.resize_to_shape(
            str(mri_cropped_path),
            str(mri_final_path),
            target_shape=(80, 80, 80)
        )
        
        # Step 14: Resize PET to 80x80x80
        logger.info("  [14/14] Resizing PET to 80x80x80...")
        pet_final_path = work_dir / "PET_final.nii.gz"
        final_prep.resize_to_shape(
            str(pet_cropped_path),
            str(pet_final_path),
            target_shape=(80, 80, 80)
        )
        
        # Save only final outputs
        shutil.copy(mri_final_path, output_dir / "MRI_final.nii.gz")
        shutil.copy(pet_final_path, output_dir / "PET_final.nii.gz")
        
        logger.info(f"  ✓ Complete - SUVR: {suvr_results['global_suvr']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return False
    
    finally:
        # Cleanup
        if work_dir.exists():
            shutil.rmtree(work_dir)


def process_session_wrapper(session, config_path, template_dir):
    """Wrapper for parallel processing - loads config in each worker."""
    # Setup logging for this worker
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load config in worker
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return process_session(session, config, Path(template_dir))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MRI-PET Preprocessing Pipeline')
    parser.add_argument('--subject', type=str, help='Process single subject')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--limit', type=int, help='Limit number of sessions to process')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    
    # Load config
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yaml"
    config = load_config(config_path)
    
    # Setup template directory
    template_dir = script_dir / "templates"
    if not template_dir.exists():
        template_dir = Path("/home/ardkav/Desktop/MRI2PET/preprocessing/templates")
    
    # Find sessions
    logger.info("Finding MRI-PET sessions...")
    sessions = find_sessions(config['data']['mcsa_root'])
    
    logger.info(f"Found {len(sessions)} sessions with both T1-MRI and PiB-PET")
    
    if len(sessions) == 0:
        logger.error("No sessions found")
        return 1
    
    # Filter by subject if requested
    if args.subject:
        sessions = [s for s in sessions if s['subject_id'] == args.subject]
        logger.info(f"Filtered to subject: {args.subject} ({len(sessions)} sessions)")
    
    # Limit number of sessions if requested
    if args.limit:
        sessions = sessions[:args.limit]
        logger.info(f"Limited to first {args.limit} sessions")
    
    # Process sessions
    success = 0
    failed = 0
    failed_cases = []
    
    if args.workers > 1:
        # Parallel processing
        logger.info(f"Processing {len(sessions)} sessions with {args.workers} workers...")
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_session_wrapper, 
                    session, 
                    str(config_path), 
                    str(template_dir)
                ): session 
                for session in sessions
            }
            
            for future in as_completed(futures):
                session = futures[future]
                try:
                    result = future.result()
                    if result:
                        success += 1
                    else:
                        failed += 1
                        failed_cases.append(f"{session['subject_id']},{session['session_date']}")
                except Exception as e:
                    logger.error(f"Worker error for {session['subject_id']}: {e}")
                    failed += 1
                    failed_cases.append(f"{session['subject_id']},{session['session_date']},{str(e)}")
    else:
        # Sequential processing
        for session in sessions:
            if process_session(session, config, template_dir):
                success += 1
            else:
                failed += 1
                failed_cases.append(f"{session['subject_id']},{session['session_date']}")
    
    logger.info(f"\nSummary: {success} successful, {failed} failed out of {len(sessions)}")
    
    # Save failed cases to file
    if failed_cases:
        failed_file = script_dir / "failed_cases.csv"
        with open(failed_file, 'w') as f:
            f.write("subject_id,session_date,error\n")
            for case in failed_cases:
                f.write(case + "\n")
        logger.info(f"Failed cases saved to: {failed_file}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
