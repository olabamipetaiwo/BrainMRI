import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet3d import UNet3D
from dataset import BraTSDataset, load_dataset_splits
from utils import get_logger, map_labels

try:
    from monai.losses import DiceLoss as MonaiDiceLoss
    _monai_available = True
except ImportError:
    _monai_available = False


# ---------------------------------------------------------------------------
# Simple pure-PyTorch soft Dice loss (fallback if MONAI unavailable)
# ---------------------------------------------------------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """logits: (B,C,H,W,D)  targets: (B,H,W,D) long"""
        n_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        # one-hot encode targets
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, targets.unsqueeze(1), 1)
        # skip background class (index 0)
        num = (2 * (probs * onehot)[:, 1:].sum(dim=(2, 3, 4)) + self.smooth)
        denom = (probs[:, 1:] + onehot[:, 1:]).sum(dim=(2, 3, 4)) + self.smooth
        return 1.0 - (num / denom).mean()


def build_loss_fn():
    if _monai_available:
        return MonaiDiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    return SoftDiceLoss()


# ---------------------------------------------------------------------------
# Validation Dice (ET/TC/WT regions)
# ---------------------------------------------------------------------------
def region_dice(pred_labels_np, gt_labels_np):
    """Return mean Dice over ET/TC/WT for a single subject."""
    pm = map_labels(pred_labels_np)
    gm = map_labels(gt_labels_np)
    scores = []
    for region in ('ET', 'TC', 'WT'):
        p = pm[region].astype(bool)
        g = gm[region].astype(bool)
        inter = (p & g).sum()
        denom = p.sum() + g.sum()
        scores.append(1.0 if denom == 0 else 2 * inter / denom)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_fold(data_dir, fold_idx, splits, device, args):
    fold_dir = os.path.join(args.output_dir, f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)

    logger = get_logger(
        f'fold_{fold_idx}',
        log_file=os.path.join(fold_dir, 'train.log'),
    )

    train_subjects, val_subjects = splits[fold_idx - 1]
    logger.info(f'Fold {fold_idx}: {len(train_subjects)} train  /  '
                f'{len(val_subjects)} val')

    train_ds = BraTSDataset(data_dir, train_subjects, augment=True)
    val_ds   = BraTSDataset(data_dir, val_subjects,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, args.workers // 2), pin_memory=True,
    )

    model     = UNet3D(in_channels=4, num_classes=4, base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, verbose=True,
    )

    dice_loss_fn = build_loss_fn()
    ce_loss_fn   = nn.CrossEntropyLoss()

    best_dice    = -1.0
    ckpt_path    = os.path.join(fold_dir, 'best_model.pth')

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        train_losses = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)

            dloss  = dice_loss_fn(logits, labels.unsqueeze(1))
            celoss = ce_loss_fn(logits, labels)
            loss   = 0.5 * dloss + 0.5 * celoss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---- validate ----
        model.eval()
        val_dices = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                pred   = logits.argmax(dim=1)[0].cpu().numpy()
                gt     = labels[0].numpy()
                val_dices.append(region_dice(pred, gt))

        mean_val_dice = float(np.mean(val_dices))
        scheduler.step(1.0 - mean_val_dice)

        logger.info(
            f'Epoch {epoch:4d}/{args.epochs}  '
            f'train_loss={np.mean(train_losses):.4f}  '
            f'val_dice={mean_val_dice:.4f}'
        )

        if mean_val_dice > best_dice:
            best_dice = mean_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, ckpt_path)
            logger.info(f'  >> saved best model  (dice={best_dice:.4f})')

    logger.info(f'Fold {fold_idx} finished — best val dice: {best_dice:.4f}')

    # Persist best Dice for reference
    with open(os.path.join(fold_dir, 'best_dice.json'), 'w') as f:
        json.dump({'fold': fold_idx, 'best_val_dice': best_dice}, f, indent=2)

    return best_dice


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Train 3D U-Net on BraTS data')
    p.add_argument('--data_dir',    default='.',
                   help='Path to Task01_BrainTumour/')
    p.add_argument('--fold',        type=int, default=1,
                   help='Fold index (1-5); use 0 to run all folds')
    p.add_argument('--epochs',      type=int, default=200)
    p.add_argument('--batch_size',  type=int, default=1)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--output_dir',  default='results')
    p.add_argument('--workers',     type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    splits = load_dataset_splits(args.data_dir)

    folds = range(1, 6) if args.fold == 0 else [args.fold]
    for fold in folds:
        train_fold(args.data_dir, fold, splits, device, args)


if __name__ == '__main__':
    main()
