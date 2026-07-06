#!/usr/bin/env bash
# scripts/run_timeseries_benchmark.sh
# ──────────────────────────────────────────────────────────────
# Runs the HTSPF time-series benchmark only (FordA + EthanolConcentration).
# Designed for M1 MacBook Air — each dataset is fast on MPS.
# Est. wall time: ~3 hours (50 epochs × 6 models × 5 seeds × 2 datasets)
#
# Usage:
#   ./scripts/run_timeseries_benchmark.sh          # full run
#   ./scripts/run_timeseries_benchmark.sh --fast   # 2-epoch dry-run
# ──────────────────────────────────────────────────────────────

set -e

FAST_FLAG=""
if [[ "$1" == "--fast" ]]; then
  FAST_FLAG="--fast"
  echo ">>> DRY RUN MODE: 2 epochs per experiment"
fi

CONFIG="configs/experiment.yaml"
OUTPUT_DIR="results/raw"
SEEDS=(0 1 2 3 4)
TS_DATASETS=("ford_a" "ethanol_concentration")
ABLATIONS=("HTSPF_Full" "HTSPF_noHCAA" "HTSPF_noASG" "HTSPF_noLDWT" "HTSPF_noConflict")

echo ""
echo "========================================================"
echo "  HTSPF TIME-SERIES BENCHMARK (Local M1 Run)"
echo "  Datasets: FordA, EthanolConcentration"
echo "  Models:   5 HTSPF variants + InceptionTime baseline"
echo "  Seeds:    5 (0-4)"
echo "========================================================"
echo ""

# ── HTSPF Ablations ──
echo "=== HTSPF Ablations ==="
for model in "${ABLATIONS[@]}"; do
  for dataset in "${TS_DATASETS[@]}"; do
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

# ── InceptionTime Baseline ──
echo ""
echo "=== InceptionTime Baseline ==="
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
echo "  TIME-SERIES BENCHMARK COMPLETE"
echo "  Results: $OUTPUT_DIR"
echo "  Next: run vision benchmark on Kaggle/Colab GPU"
echo "  Then: python scripts/compile_results.py"
echo "========================================================"
