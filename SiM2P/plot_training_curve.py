#!/usr/bin/env python3
"""
Plot training curves from SiM2P log file.

Usage:
    python plot_training_curve.py                    # Auto-find latest log
    python plot_training_curve.py --log path/to/log.txt
    python plot_training_curve.py --workdir path/to/workdir
"""

import argparse
import re
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_log_file(log_path):
    """Parse training log file and extract metrics."""
    metrics = {
        'step': [],
        'epoch': [],
        'loss': [],
        'xs_mse': [],
        'mse': [],
        'lg_loss_scale': [],
        'grad_norm': [],
        'test_loss': [],
        'test_xs_mse': [],
    }
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    current_metrics = {}
    
    for line in lines:
        line = line.strip()
        
        # Parse key-value pairs like "| step           | 100      |"
        match = re.match(r'\|\s*(\w+)\s*\|\s*([^\|]+)\s*\|', line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            
            # Try to convert to number
            try:
                if '%' in value:
                    value = float(value.replace('%', ''))
                elif 'e' in value.lower() or '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            
            current_metrics[key] = value
        
        # When we see "----" separator, save the accumulated metrics
        if line.startswith('---') and len(current_metrics) > 0:
            if 'step' in current_metrics:
                for key in metrics:
                    if key in current_metrics:
                        metrics[key].append(current_metrics[key])
                    elif key.startswith('test_') and key[5:] in current_metrics:
                        metrics[key].append(current_metrics[key[5:]])
            current_metrics = {}
    
    # Convert to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key]) if len(metrics[key]) > 0 else np.array([])
    
    return metrics


def plot_training_curves(metrics, output_path, title="Training Curves"):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    steps = metrics['step']
    
    if len(steps) == 0:
        print("No training data found in log file!")
        return
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    if len(metrics['loss']) > 0:
        valid_mask = ~np.isnan(metrics['loss'].astype(float))
        if valid_mask.any():
            ax.plot(steps[valid_mask], metrics['loss'].astype(float)[valid_mask], 'b-', label='loss', alpha=0.8)
    if len(metrics['xs_mse']) > 0:
        ax.plot(steps, metrics['xs_mse'], 'r-', label='xs_mse', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: xs_mse with smoothing
    ax = axes[0, 1]
    if len(metrics['xs_mse']) > 0:
        xs_mse = metrics['xs_mse'].astype(float)
        ax.plot(steps, xs_mse, 'r-', alpha=0.3, label='xs_mse (raw)')
        
        # Smoothed curve (moving average)
        window = min(20, len(xs_mse) // 5 + 1)
        if window > 1:
            smoothed = np.convolve(xs_mse, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            ax.plot(smooth_steps, smoothed, 'r-', linewidth=2, label=f'xs_mse (smoothed, w={window})')
    ax.set_xlabel('Step')
    ax.set_ylabel('xs_mse')
    ax.set_title('Reconstruction Error (xs_mse)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss scale (FP16 stability indicator)
    ax = axes[1, 0]
    if len(metrics['lg_loss_scale']) > 0:
        ax.plot(steps, metrics['lg_loss_scale'], 'g-', label='lg_loss_scale')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Danger zone (scale < 0)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Log Loss Scale')
    ax.set_title('FP16 Loss Scale (should stay positive)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Epoch progress and gradient norm
    ax = axes[1, 1]
    if len(metrics['grad_norm']) > 0:
        ax.plot(steps, metrics['grad_norm'], 'purple', alpha=0.8, label='grad_norm')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add epoch markers
    if len(metrics['epoch']) > 0:
        epochs = metrics['epoch'].astype(int)
        epoch_changes = np.where(np.diff(epochs) > 0)[0] + 1
        for idx in epoch_changes:
            for ax in axes.flatten():
                ax.axvline(x=steps[idx], color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training curve saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total steps: {steps[-1] if len(steps) > 0 else 0}")
    print(f"Total epochs: {int(metrics['epoch'][-1]) if len(metrics['epoch']) > 0 else 0}")
    if len(metrics['xs_mse']) > 0:
        print(f"Initial xs_mse: {metrics['xs_mse'][0]:.6f}")
        print(f"Final xs_mse: {metrics['xs_mse'][-1]:.6f}")
        print(f"Improvement: {(1 - metrics['xs_mse'][-1]/metrics['xs_mse'][0])*100:.1f}%")
    print("="*50)


def find_latest_log(base_dir):
    """Find the latest log file in the workdir."""
    # Look for log.txt in workdir subdirectories
    pattern = os.path.join(base_dir, "SiM2P_workdir", "*", "log.txt")
    logs = glob.glob(pattern)
    
    if not logs:
        # Try direct log files
        pattern = os.path.join(base_dir, "*.log")
        logs = glob.glob(pattern)
    
    if not logs:
        return None
    
    # Return most recently modified
    return max(logs, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description='Plot SiM2P training curves')
    parser.add_argument('--log', type=str, help='Path to log file')
    parser.add_argument('--workdir', type=str, help='Path to workdir (will find log.txt inside)')
    parser.add_argument('--output', type=str, default='training_curve.png', help='Output image path')
    args = parser.parse_args()
    
    # Find log file
    if args.log:
        log_path = args.log
    elif args.workdir:
        log_path = os.path.join(args.workdir, 'log.txt')
    else:
        # Auto-find
        log_path = find_latest_log('/home/ardkav/Desktop/MRI2PET/SiM2P')
    
    if not log_path or not os.path.exists(log_path):
        print(f"Error: Log file not found: {log_path}")
        return
    
    print(f"Parsing log file: {log_path}")
    
    # Parse and plot
    metrics = parse_log_file(log_path)
    
    # Determine output path
    if args.output == 'training_curve.png':
        output_path = os.path.join(os.path.dirname(log_path), 'training_curve.png')
    else:
        output_path = args.output
    
    title = f"Training Curves - {os.path.basename(os.path.dirname(log_path))}"
    plot_training_curves(metrics, output_path, title)


if __name__ == '__main__':
    main()
