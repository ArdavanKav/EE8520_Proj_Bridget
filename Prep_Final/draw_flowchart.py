#!/usr/bin/env python3
"""
Draw preprocessing pipeline flowchart and save as PNG.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def draw_flowchart():
    fig, ax = plt.subplots(1, 1, figsize=(20, 28))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 28)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    mri_color = '#4A90D9'       # Blue for MRI
    pet_color = '#E74C3C'       # Red for PET
    combined_color = '#9B59B6'  # Purple for combined
    output_color = '#27AE60'    # Green for output
    input_color = '#F39C12'     # Orange for input
    
    def draw_box(x, y, w, h, text, color, fontsize=9, text_color='white'):
        """Draw a rounded box with text."""
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.05,rounding_size=0.3",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        
        # Handle multiline text
        lines = text.split('\n')
        line_height = fontsize * 0.035
        start_y = y + (len(lines) - 1) * line_height / 2
        for i, line in enumerate(lines):
            ax.text(x, start_y - i * line_height, line, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color=text_color)
    
    def draw_arrow(x1, y1, x2, y2, color='black'):
        """Draw an arrow between two points."""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    def draw_curved_arrow(x1, y1, x2, y2, color='black', connectionstyle="arc3,rad=0.3"):
        """Draw a curved arrow."""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                  connectionstyle=connectionstyle))
    
    # Title
    ax.text(10, 27.5, 'MRI-PET Preprocessing Pipeline', ha='center', va='center',
           fontsize=20, fontweight='bold')
    ax.text(10, 27, 'Prep_Final - SiM2P Project', ha='center', va='center',
           fontsize=14, style='italic', color='gray')
    
    # ============ INPUT SECTION ============
    y_input = 26
    draw_box(6, y_input, 3.5, 0.8, 'T1-weighted MRI\n(DICOM)', input_color, fontsize=10)
    draw_box(14, y_input, 3.5, 0.8, 'PiB-PET\n(DICOM)', input_color, fontsize=10)
    
    # ============ STEP 0 ============
    y0 = 24.5
    draw_box(10, y0, 8, 0.8, 'STEP 0: DICOM to NIfTI Conversion (dcm2niix)', combined_color)
    draw_arrow(6, y_input - 0.4, 8, y0 + 0.4)
    draw_arrow(14, y_input - 0.4, 12, y0 + 0.4)
    
    # Split into MRI and PET paths
    draw_arrow(8, y0 - 0.4, 5, 23.2)
    draw_arrow(12, y0 - 0.4, 15, 22.2)
    
    # ============ MRI PATH (Left side) ============
    # Step 1
    y1 = 22.8
    draw_box(5, y1, 4, 1.0, 'STEP 1: Skull Stripping\nFSL BET (f=0.35)\nBias correction', mri_color)
    
    # Step 2
    y2 = 21.2
    draw_box(5, y2, 4, 1.0, 'STEP 2: MNI Registration\nANTsPy SyN\nMNI152 1mm template', mri_color)
    draw_arrow(5, y1 - 0.5, 5, y2 + 0.5)
    
    # ============ PET PATH (Right side) ============
    # Step 3
    y3 = 21.8
    draw_box(15, y3, 4, 0.8, 'STEP 3: Motion Correction\nFrame averaging', pet_color)
    
    # Step 4
    y4 = 20.4
    draw_box(15, y4, 4, 0.8, 'STEP 4: Coregister to MRI\nANTsPy Rigid', pet_color)
    draw_arrow(15, y3 - 0.4, 15, y4 + 0.4)
    
    # Arrow from MRI to PET coregistration
    draw_curved_arrow(7, y1 - 0.5, 13, y4, color=mri_color, connectionstyle="arc3,rad=-0.3")
    
    # ============ COMBINED PATH ============
    # Step 5
    y5 = 19
    draw_box(10, y5, 6, 0.8, 'STEP 5: Transform PET to MNI\nApply MRI→MNI transforms', combined_color)
    draw_arrow(5, y2 - 0.5, 8, y5 + 0.4)
    draw_arrow(15, y4 - 0.4, 12, y5 + 0.4)
    
    # Step 6
    y6 = 17.6
    draw_box(10, y6, 6, 0.9, 'STEP 6: Compute SUVR\nCerebellar reference region\nSUVR = PET / mean(cerebellum)', combined_color)
    draw_arrow(10, y5 - 0.4, 10, y6 + 0.45)
    
    # Split again
    draw_arrow(8, y6 - 0.45, 5, 16.2)
    draw_arrow(12, y6 - 0.45, 15, 16.2)
    
    # ============ PARALLEL PROCESSING ============
    # Step 7 (MRI)
    y7 = 15.8
    draw_box(5, y7, 4, 0.8, 'STEP 7: Resample MRI\n1.5mm isotropic', mri_color)
    
    # Step 8 (PET)
    y8 = 15.8
    draw_box(15, y8, 4, 0.8, 'STEP 8: Resample PET\n1.5mm isotropic', pet_color)
    
    # Step 9
    y9 = 14.4
    draw_box(15, y9, 4, 0.8, 'STEP 9: Smoothing\n4mm FWHM Gaussian', pet_color)
    draw_arrow(15, y8 - 0.4, 15, y9 + 0.4)
    
    # Step 10
    y10 = 13
    draw_box(15, y10, 4, 0.8, 'STEP 10: Normalize SUVR\n0-1 range (1-99%ile)', pet_color)
    draw_arrow(15, y9 - 0.4, 15, y10 + 0.4)
    
    # ============ MASKING ============
    # Step 11
    y11 = 11.4
    draw_box(10, y11, 8, 0.9, 'STEP 11: Apply Brain Mask\nResample mask (nearest neighbor)\nSmooth edges (σ=0.8), threshold 0.5', combined_color)
    draw_arrow(5, y7 - 0.4, 8, y11 + 0.45)
    draw_arrow(15, y10 - 0.4, 12, y11 + 0.45)
    
    # ============ CROPPING ============
    # Step 12
    y12 = 9.8
    draw_box(10, y12, 8, 0.9, 'STEP 12: Crop to 120×128×120\nCenter-aligned cropping\nRemove excess background', combined_color)
    draw_arrow(10, y11 - 0.45, 10, y12 + 0.45)
    
    # Split for final resize
    draw_arrow(8, y12 - 0.45, 5, 8.2)
    draw_arrow(12, y12 - 0.45, 15, 8.2)
    
    # ============ FINAL RESIZE ============
    # Step 13 (MRI)
    y13 = 7.8
    draw_box(5, y13, 4, 0.9, 'STEP 13: Resize MRI\n80×80×80\nCubic interpolation', mri_color)
    
    # Step 14 (PET)
    y14 = 7.8
    draw_box(15, y14, 4, 0.9, 'STEP 14: Resize PET\n80×80×80\nCubic interpolation', pet_color)
    
    # ============ OUTPUT ============
    y_out = 5.8
    draw_box(5, y_out, 4, 1.2, 'MRI_final.nii.gz\n80×80×80\n2.25mm isotropic', output_color, fontsize=10)
    draw_box(15, y_out, 4, 1.2, 'PET_final.nii.gz\n80×80×80\n2.25mm isotropic', output_color, fontsize=10)
    draw_arrow(5, y13 - 0.45, 5, y_out + 0.6)
    draw_arrow(15, y14 - 0.45, 15, y_out + 0.6)
    
    # Final output box
    y_final = 4
    draw_box(10, y_final, 12, 1.5, 'FINAL OUTPUT\nMNI152 Space | Brain-masked | Ready for SiM2P Training', output_color, fontsize=12)
    draw_arrow(5, y_out - 0.6, 6, y_final + 0.75)
    draw_arrow(15, y_out - 0.6, 14, y_final + 0.75)
    
    # ============ LEGEND ============
    legend_y = 1.5
    ax.text(10, legend_y + 1, 'Legend', ha='center', fontsize=12, fontweight='bold')
    
    # Legend boxes
    legend_items = [
        (3, 'Input Data', input_color),
        (6, 'MRI Processing', mri_color),
        (9, 'PET Processing', pet_color),
        (12, 'Combined Steps', combined_color),
        (15, 'Output', output_color),
    ]
    
    for x, label, color in legend_items:
        box = FancyBboxPatch((x - 0.8, legend_y - 0.3), 1.6, 0.6,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, legend_y, label, ha='center', va='center', fontsize=8, 
               fontweight='bold', color='white')
    
    # Add processing info
    ax.text(10, 0.5, 'Parallel Processing: 96 workers | Dataset: 2821 sessions | ~5 min/case',
           ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    
    # Save
    output_path = '/home/ardkav/Desktop/MRI2PET/Prep_Final/pipeline_flowchart.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Flowchart saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    draw_flowchart()
