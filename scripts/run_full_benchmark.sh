#!/usr/bin/env bash
# scripts/run_full_benchmark.sh
# ──────────────────────────────────────────────────────────────
# Runs the complete HTSPF Phase 3 benchmark suite:
#   - All HTSPF ablations × 3 datasets × 5 seeds
#   - All baselines × their paired datasets × 5 seeds
#
# Usage:
#   ./scripts/run_full_benchmark.sh             # full run
#   ./scripts/run_full_benchmark.sh --fast      # dry-run (2 epochs per model)
# ──────────────────────────────────────────────────────────────

set -e  # Exit immediately on error

FAST_FLAG=""
if [[ "$1" == "--fast" ]]; then
  FAST_FLAG="--fast"
  echo ">>> DRY RUN MODE: 2 epochs per experiment"
fi

CONFIG="configs/experiment.yaml"
OUTPUT_DIR="results/raw"

SEEDS=(0 1 2 3 4)

# ── HTSPF Ablations (all three datasets) ──
ABLATIONS=("HTSPF_Full" "HTSPF_noHCAA" "HTSPF_noASG" "HTSPF_noLDWT" "HTSPF_noConflict")
ABLATION_DATASETS=("cifar100" "ford_a" "ethanol_concentration")

echo ""
echo "========================================================"
echo "  HTSPF ABLATIONS"
echo "========================================================"

for model in "${ABLATIONS[@]}"; do
  for dataset in "${ABLATION_DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo ">>> $model | $dataset | seed=$seed"
      python src/train.py \
        --model   "$model" \
        --dataset "$dataset" \
        --seed    "$seed" \
        --config  "$CONFIG" \
        --output  "$OUTPUT_DIR" \
        $FAST_FLAG
    done
  done
done

# ── Baselines ──
echo ""
echo "========================================================"
echo "  BASELINES"
echo "========================================================"

# Vision baselines (CIFAR-100)
VISION_BASELINES=("resnet18" "vit_small" "perceiver_io")
for model in "${VISION_BASELINES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo ">>> $model | cifar100 | seed=$seed"
    python src/train.py \
      --model   "$model" \
      --dataset "cifar100" \
      --seed    "$seed" \
      --config  "$CONFIG" \
      --output  "$OUTPUT_DIR" \
      $FAST_FLAG
  done
done

# Time-series baselines (UCR datasets)
TS_DATASETS=("ford_a" "ethanol_concentration")
for dataset in "${TS_DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo ">>> inception_time | $dataset | seed=$seed"
    python src/train.py \
      --model   "inception_time" \
      --dataset "$dataset" \
      --seed    "$seed" \
      --config  "$CONFIG" \
      --output  "$OUTPUT_DIR" \
      $FAST_FLAG
  done
done

echo ""
echo "========================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results saved to: $OUTPUT_DIR"
echo "  Run: python scripts/compile_results.py to generate tables"
echo "========================================================"
