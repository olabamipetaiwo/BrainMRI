import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet3d import UNet3D
from dataset import BraTSDataset, load_dataset_splits
from utils import get_logger, map_labels

try:
    from medpy.metric import binary as medpy_binary
    _HAS_MEDPY = True
except ImportError:
    _HAS_MEDPY = False
    print('WARNING: medpy not installed — HD95 will be skipped.')


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 1.0   # both empty → perfect agreement
    return float(2 * inter / denom)


def compute_hd95(pred_bin: np.ndarray, gt_bin: np.ndarray,
                 spacing=(1.0, 1.0, 1.0)) -> float:
    if not _HAS_MEDPY:
        return float('nan')
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('nan')   # undefined if one mask is empty
    try:
        return float(medpy_binary.hd95(
            pred_bin, gt_bin, voxelspacing=spacing, connectivity=1))
    except Exception:
        return float('nan')


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------
def evaluate_fold(data_dir, fold_idx, splits, device, args):
    fold_dir  = os.path.join(args.output_dir, f'fold_{fold_idx}')
    ckpt_path = os.path.join(fold_dir, 'best_model.pth')
    logger    = get_logger(
        f'eval_{fold_idx}',
        log_file=os.path.join(fold_dir, 'eval.log'),
    )

    if not os.path.exists(ckpt_path):
        logger.error(f'Checkpoint not found: {ckpt_path} — skip fold {fold_idx}')
        return None

    _, val_subjects = splits[fold_idx - 1]
    val_ds = BraTSDataset(data_dir, val_subjects, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = UNet3D(in_channels=4, num_classes=4, base_ch=32).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info(f'Loaded fold {fold_idx} checkpoint (epoch {ckpt["epoch"]})')

    bucket = {r: {'dice': [], 'hd95': []} for r in ('ET', 'TC', 'WT')}

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            logits = model(images.to(device))
            pred   = logits.argmax(dim=1)[0].cpu().numpy()   # (H,W,D)
            gt     = labels[0].numpy()

            pred_map = map_labels(pred)
            gt_map   = map_labels(gt)

            for region in ('ET', 'TC', 'WT'):
                pm = pred_map[region].astype(bool)
                gm = gt_map[region].astype(bool)
                bucket[region]['dice'].append(compute_dice(pm, gm))
                bucket[region]['hd95'].append(compute_hd95(pm, gm))

            if (i + 1) % 20 == 0:
                logger.info(f'  {i+1}/{len(val_loader)} subjects done')

    # Summary
    summary = {}
    header = f'{"Region":>6}  {"Dice":>8}  {"HD95":>8}'
    logger.info(f'\nFold {fold_idx} — {len(val_subjects)} val subjects')
    logger.info(header)
    logger.info('-' * len(header))

    for region in ('ET', 'TC', 'WT'):
        dices = [v for v in bucket[region]['dice'] if not np.isnan(v)]
        hd95s = [v for v in bucket[region]['hd95'] if not np.isnan(v)]
        mean_dice = float(np.mean(dices)) if dices else float('nan')
        mean_hd95 = float(np.mean(hd95s)) if hd95s else float('nan')
        summary[region] = {'dice': mean_dice, 'hd95': mean_hd95}
        logger.info(f'{region:>6}  {mean_dice:>8.4f}  {mean_hd95:>8.2f}')

    out_path = os.path.join(fold_dir, 'metrics.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f'Metrics saved → {out_path}')

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Evaluate 3D U-Net on BraTS folds')
    p.add_argument('--data_dir', default='data')
    p.add_argument('--fold',       type=int, default=1,
                   help='Fold index (1-5); 0 = all folds')
    p.add_argument('--output_dir', default='results')
    return p.parse_args()


def print_cv_table(all_results):
    cols = ('ET', 'TC', 'WT')
    header = (f'{"Fold":<5}' +
              ''.join(f'  {c+" Dice":>10}' for c in cols) +
              ''.join(f'  {c+" HD95":>10}' for c in cols))
    print('\n=== Cross-Validation Summary ===')
    print(header)
    print('-' * len(header))

    agg = {r: {'dice': [], 'hd95': []} for r in cols}
    for fold, res in sorted(all_results.items()):
        row = f'{fold:<5}'
        for r in cols:
            row += f'  {res[r]["dice"]:>10.4f}'
            agg[r]['dice'].append(res[r]['dice'])
        for r in cols:
            row += f'  {res[r]["hd95"]:>10.2f}'
            agg[r]['hd95'].append(res[r]['hd95'])
        print(row)

    print('-' * len(header))
    row = f'{"Avg":<5}'
    for r in cols:
        vals = [v for v in agg[r]['dice'] if not np.isnan(v)]
        row += f'  {np.mean(vals):>10.4f}'
    for r in cols:
        vals = [v for v in agg[r]['hd95'] if not np.isnan(v)]
        row += f'  {np.mean(vals):>10.2f}'
    print(row)


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = load_dataset_splits(args.data_dir)

    folds = range(1, 6) if args.fold == 0 else [args.fold]
    all_results = {}
    for fold in folds:
        result = evaluate_fold(args.data_dir, fold, splits, device, args)
        if result:
            all_results[fold] = result

    if len(all_results) > 1:
        print_cv_table(all_results)

        agg_path = os.path.join(args.output_dir, 'cv_results.json')
        with open(agg_path, 'w') as f:
            json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
        print(f'\nAggregated results saved → {agg_path}')


if __name__ == '__main__':
    main()
