"""Generate training loss + validation Dice curve for Fold 1 from train.log."""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

LOG_PATH = 'results/fold_1/train.log'
OUT_PATH = 'results/visualizations/fold1_training_curves.png'

# Parse the last complete run from the log
epochs, train_losses, val_dices = [], [], []
pattern = re.compile(
    r'Epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+val_dice=([\d.]+)'
)

# Collect all runs then keep only the last one
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

last_run = all_runs[-1]
epochs      = [r[0] for r in last_run]
train_losses = [r[1] for r in last_run]
val_dices    = [r[2] for r in last_run]

# Plot
fig, ax1 = plt.subplots(figsize=(6, 4))

color_loss = '#d62728'
color_dice = '#1f77b4'

ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Training Loss (Dice + CE)', color=color_loss, fontsize=11)
ax1.plot(epochs, train_losses, color=color_loss, marker='o', linewidth=2,
         markersize=5, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.set_ylim(0, max(train_losses) * 1.15)

ax2 = ax1.twinx()
ax2.set_ylabel('Mean Validation Dice (ET/TC/WT)', color=color_dice, fontsize=11)
ax2.plot(epochs, val_dices, color=color_dice, marker='s', linewidth=2,
         linestyle='--', markersize=5, label='Val Dice')
ax2.tick_params(axis='y', labelcolor=color_dice)
ax2.set_ylim(0, 1.0)

ax1.set_xticks(epochs)
ax1.set_title('Fold 1 — Training Curves (10 epochs)', fontsize=12)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f'Saved → {OUT_PATH}')
