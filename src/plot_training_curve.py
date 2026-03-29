"""Generate training curve comparison: baseline (10 ep) vs improved (15 ep) for Fold 1."""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

LOG_PATH = 'results/fold_1/train.log'
OUT_PATH = 'results/visualizations/fold1_training_curves.png'

pattern = re.compile(
    r'Epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+val_dice=([\d.]+)'
)

# Collect all runs from the log
all_runs = []
current_run = []
with open(LOG_PATH) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            ep, loss, dice = int(m.group(1)), float(m.group(2)), float(m.group(3))
            if ep == 1 and current_run:
                all_runs.append(current_run)
                current_run = []
            current_run.append((ep, loss, dice))
if current_run:
    all_runs.append(current_run)

# Pick baseline (10 ep) and improved (15 ep) runs
baseline = [r for r in all_runs if len(r) == 10][-1]
improved = [r for r in all_runs if len(r) == 15][-1]

def unzip(run):
    epochs      = [r[0] for r in run]
    train_losses = [r[1] for r in run]
    val_dices    = [r[2] for r in run]
    return epochs, train_losses, val_dices

ep_b, loss_b, dice_b = unzip(baseline)
ep_i, loss_i, dice_i = unzip(improved)

# Plot
fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(11, 4))

# --- Loss ---
ax_loss.plot(ep_b, loss_b, color='#d62728', marker='o', linewidth=2,
             markersize=5, linestyle='--', label='Baseline (10 ep)')
ax_loss.plot(ep_i, loss_i, color='#8c0a0a', marker='o', linewidth=2,
             markersize=5, label='Improved (15 ep)')
ax_loss.set_xlabel('Epoch', fontsize=11)
ax_loss.set_ylabel('Training Loss (Dice + CE)', fontsize=11)
ax_loss.set_title('Training Loss', fontsize=12)
ax_loss.legend(fontsize=10)
ax_loss.set_xticks(ep_i)
ax_loss.grid(True, alpha=0.3)

# --- Val Dice ---
ax_dice.plot(ep_b, dice_b, color='#1f77b4', marker='s', linewidth=2,
             markersize=5, linestyle='--', label='Baseline (10 ep)')
ax_dice.plot(ep_i, dice_i, color='#0a4a8c', marker='s', linewidth=2,
             markersize=5, label='Improved (15 ep)')
ax_dice.axhline(max(dice_b), color='#1f77b4', linewidth=1, linestyle=':',
                alpha=0.7, label=f'Baseline best: {max(dice_b):.4f}')
ax_dice.axhline(max(dice_i), color='#0a4a8c', linewidth=1, linestyle=':',
                alpha=0.7, label=f'Improved best: {max(dice_i):.4f}')
ax_dice.set_xlabel('Epoch', fontsize=11)
ax_dice.set_ylabel('Mean Validation Dice (ET/TC/WT)', fontsize=11)
ax_dice.set_title('Validation Dice', fontsize=12)
ax_dice.set_ylim(0.4, 1.0)
ax_dice.set_xticks(ep_i)
ax_dice.legend(fontsize=9)
ax_dice.grid(True, alpha=0.3)

fig.suptitle('Fold 1 — Baseline vs Improved Training Curves', fontsize=13, fontweight='bold')
fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved → {OUT_PATH}')
