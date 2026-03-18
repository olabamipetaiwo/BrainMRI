#!/usr/bin/env bash
# =============================================================================
# Brain Tumor Segmentation — 3D U-Net
# CAP 5516 – Medical Image Computing (Spring 2026)
#
# Usage:
#   bash run_experiments.sh              # run all folds (default)
#   bash run_experiments.sh --fold 1     # run a single fold only
#   bash run_experiments.sh --epochs 400 # override epochs
# =============================================================================

set -euo pipefail

# ── Defaults ────────
DATA_DIR="data"
OUTPUT_DIR="results"
VIS_DIR="results/visualizations"
EPOCHS=10
BATCH_SIZE=1
LR=1e-4
WORKERS=4
N_VIS=5        # subjects to visualize per fold
FOLD=0         # 0 = all folds

# ── Argument parsing ─
while [[ $# -gt 0 ]]; do
  case $1 in
    --fold)        FOLD="$2";       shift 2 ;;
    --epochs)      EPOCHS="$2";     shift 2 ;;
    --batch_size)  BATCH_SIZE="$2"; shift 2 ;;
    --lr)          LR="$2";         shift 2 ;;
    --workers)     WORKERS="$2";    shift 2 ;;
    --n_vis)       N_VIS="$2";      shift 2 ;;
    --data_dir)    DATA_DIR="$2";   shift 2 ;;
    --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Helpers ──────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Determine which folds to run
if [[ "$FOLD" -eq 0 ]]; then
  FOLDS=(1 2 3 4 5)
else
  FOLDS=("$FOLD")
fi

# ── Sanity checks ────
if [[ ! -f "$DATA_DIR/dataset.json" ]]; then
  echo "ERROR: dataset.json not found in '$DATA_DIR/'. Check --data_dir."
  exit 1
fi

if ! python -c "import torch" &>/dev/null; then
  echo "ERROR: PyTorch is not installed. Run: pip install torch torchvision"
  exit 1
fi

DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
log "Device: $DEVICE"
log "Folds to run: ${FOLDS[*]}"
log "Epochs: $EPOCHS | Batch: $BATCH_SIZE | LR: $LR"
echo "============================================================"

# ── Step 1: Train ────
for FOLD_ID in "${FOLDS[@]}"; do
  log ">>> TRAINING fold $FOLD_ID / ${#FOLDS[@]}"
  python src/train.py \
    --data_dir    "$DATA_DIR"   \
    --fold        "$FOLD_ID"    \
    --epochs      "$EPOCHS"     \
    --batch_size  "$BATCH_SIZE" \
    --lr          "$LR"         \
    --output_dir  "$OUTPUT_DIR" \
    --workers     "$WORKERS"
  log "Fold $FOLD_ID training done."
  echo "------------------------------------------------------------"
done

# ── Step 2: Evaluate ─
log ">>> EVALUATING all trained folds"
python src/evaluate.py \
  --data_dir   "$DATA_DIR"   \
  --fold       0             \
  --output_dir "$OUTPUT_DIR"
echo "------------------------------------------------------------"

# ── Step 3: Visualize 
for FOLD_ID in "${FOLDS[@]}"; do
  log ">>> VISUALIZING fold $FOLD_ID"
  python src/visualize.py \
    --data_dir   "$DATA_DIR"   \
    --fold       "$FOLD_ID"    \
    --output_dir "$OUTPUT_DIR" \
    --vis_dir    "$VIS_DIR"    \
    --n_subjects "$N_VIS"
done

echo "============================================================"
log "All done."
log "  Checkpoints : $OUTPUT_DIR/fold_N/best_model.pth"
log "  Metrics     : $OUTPUT_DIR/fold_N/metrics.json"
log "  CV summary  : $OUTPUT_DIR/cv_results.json"
log "  Plots       : $VIS_DIR/"
