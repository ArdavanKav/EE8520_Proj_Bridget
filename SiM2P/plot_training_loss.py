#!/usr/bin/env python3
"""Plot training loss from step_losses.log"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the log file
log_path = '/home/ardkav/Desktop/MRI2PET/SiM2P/SiM2P_workdir/mcsa_mri2pet80_DiT-XL_4_1gpu/step_losses.log'
df = pd.read_csv(log_path)

# Group by step and take the mean (since there are 2 microbatches per step)
df_grouped = df.groupby('step').agg({
    'loss': 'mean',
    'mse': 'mean', 
    'xs_mse': 'mean',
    'has_nan': 'any',
    'has_inf': 'any'
}).reset_index()

print(f"Training Progress Summary:")
print(f"  Total steps logged: {df_grouped['step'].max()}")
print(f"  Initial loss: {df_grouped['loss'].iloc[0]:.4f}")
print(f"  Current loss: {df_grouped['loss'].iloc[-1]:.4f}")
print(f"  Min loss: {df_grouped['loss'].min():.4f}")
print(f"  Any NaN: {df_grouped['has_nan'].any()}")
print(f"  Any Inf: {df_grouped['has_inf'].any()}")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss over steps
ax1 = axes[0, 0]
ax1.plot(df_grouped['step'], df_grouped['loss'], 'b-', linewidth=1, alpha=0.7)
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss vs Step')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Plot 2: xs_mse (reconstruction quality)
ax2 = axes[0, 1]
ax2.plot(df_grouped['step'], df_grouped['xs_mse'], 'g-', linewidth=1, alpha=0.7)
ax2.set_xlabel('Step')
ax2.set_ylabel('xs_mse')
ax2.set_title('Reconstruction MSE (xs_mse) vs Step')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# Plot 3: Loss with moving average
ax3 = axes[1, 0]
window = min(10, len(df_grouped))
loss_ma = df_grouped['loss'].rolling(window=window, min_periods=1).mean()
ax3.plot(df_grouped['step'], df_grouped['loss'], 'b-', linewidth=0.5, alpha=0.3, label='Raw')
ax3.plot(df_grouped['step'], loss_ma, 'r-', linewidth=2, label=f'MA({window})')
ax3.set_xlabel('Step')
ax3.set_ylabel('Loss')
ax3.set_title('Training Loss with Moving Average')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Both metrics together (normalized)
ax4 = axes[1, 1]
ax4.plot(df_grouped['step'], df_grouped['loss'] / df_grouped['loss'].iloc[0], 'b-', linewidth=1.5, label='Loss (normalized)')
ax4.plot(df_grouped['step'], df_grouped['xs_mse'] / df_grouped['xs_mse'].iloc[0], 'g-', linewidth=1.5, label='xs_mse (normalized)')
ax4.set_xlabel('Step')
ax4.set_ylabel('Normalized Value (relative to start)')
ax4.set_title('Normalized Loss & xs_mse')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ardkav/Desktop/MRI2PET/SiM2P/training_loss_plot.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: /home/ardkav/Desktop/MRI2PET/SiM2P/training_loss_plot.png")
plt.show()
