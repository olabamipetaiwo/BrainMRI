import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet3d import UNet3D
from dataset import BraTSDataset, load_dataset_splits
from utils import map_labels


# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------
#  Label → RGB (uint8)
LABEL_COLORS = np.array([
    [0,   0,   0  ],   # 0: background (black)
    [0,   180, 255],   # 1: edema      (cyan-blue)
    [255, 165, 0  ],   # 2: non-enhancing tumor (orange)
    [255, 50,  50 ],   # 3: enhancing tumor (red)
], dtype=np.uint8)

LEGEND_PATCHES = [
    mpatches.Patch(color=LABEL_COLORS[1] / 255., label='Edema'),
    mpatches.Patch(color=LABEL_COLORS[2] / 255., label='Non-enhancing tumor'),
    mpatches.Patch(color=LABEL_COLORS[3] / 255., label='Enhancing tumor'),
]


def label_to_rgb(label_2d: np.ndarray) -> np.ndarray:
    """Convert a (H, W) integer label slice to an (H, W, 3) RGB image."""
    rgb = np.zeros((*label_2d.shape, 3), dtype=np.uint8)
    for c, color in enumerate(LABEL_COLORS):
        rgb[label_2d == c] = color
    return rgb


# ---------------------------------------------------------------------------
# Per-subject visualisation
# ---------------------------------------------------------------------------
def visualize_subject(image: np.ndarray, gt_label: np.ndarray,
                      pred_label: np.ndarray, subject_id: str,
                      save_dir: str, slice_idx: int = None):
    """Save a 4-panel figure: FLAIR | T1w | Ground Truth | Prediction.

    Args:
        image      : (4, H, W, D) float numpy
        gt_label   : (H, W, D) int numpy
        pred_label : (H, W, D) int numpy
        slice_idx  : axial (D-axis) slice to show; defaults to mid-slice
    """
    os.makedirs(save_dir, exist_ok=True)

    if slice_idx is None:
        slice_idx = image.shape[3] // 2

    flair  = image[0, :, :, slice_idx]
    t1w    = image[1, :, :, slice_idx]
    gt_sl  = gt_label[:, :, slice_idx]
    pr_sl  = pred_label[:, :, slice_idx]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f'{subject_id}  —  axial slice {slice_idx}', fontsize=12)

    axes[0].imshow(flair.T, cmap='gray', origin='lower')
    axes[0].set_title('FLAIR')

    axes[1].imshow(t1w.T, cmap='gray', origin='lower')
    axes[1].set_title('T1w')

    axes[2].imshow(label_to_rgb(gt_sl.T), origin='lower')
    axes[2].set_title('Ground Truth')

    axes[3].imshow(label_to_rgb(pr_sl.T), origin='lower')
    axes[3].set_title('Prediction')

    for ax in axes:
        ax.axis('off')

    fig.legend(handles=LEGEND_PATCHES, loc='lower center',
               ncol=3, fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = os.path.join(save_dir, f'{subject_id}_slice{slice_idx:03d}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {save_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Visualize BraTS U-Net predictions')
    p.add_argument('--data_dir', default='data')
    p.add_argument('--fold',        type=int, default=1)
    p.add_argument('--output_dir',  default='results')
    p.add_argument('--vis_dir',     default='results/visualizations')
    p.add_argument('--n_subjects',  type=int, default=5)
    p.add_argument('--slice_idx',   type=int, default=None,
                   help='Fixed axial slice (default: mid-slice per subject)')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits = load_dataset_splits(args.data_dir)
    _, val_subjects = splits[args.fold - 1]
    subset = val_subjects[:args.n_subjects]

    ckpt_path = os.path.join(args.output_dir, f'fold_{args.fold}', 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    model = UNet3D(in_channels=4, num_classes=4, base_ch=32).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    val_ds     = BraTSDataset(args.data_dir, subset, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            logits = model(images.to(device))
            pred   = logits.argmax(dim=1)[0].cpu().numpy()
            image_np = images[0].numpy()
            gt_np    = labels[0].numpy()
            subject_id = f'fold{args.fold}_sub{i:03d}'

            visualize_subject(
                image_np, gt_np, pred,
                subject_id, args.vis_dir,
                slice_idx=args.slice_idx,
            )


if __name__ == '__main__':
    main()
