#!/usr/bin/env python3
"""
scripts/compile_results.py — Results Aggregator and Table Generator.

Reads all per-seed JSON files from results/raw/, aggregates them into
mean ± std tables, runs paired t-tests, and outputs:
  - results/summary.json       (machine-readable)
  - results/final_table_*.tex  (camera-ready LaTeX, one per dataset)
  - Console summary            (quick human-readable overview)

Usage:
    python scripts/compile_results.py
    python scripts/compile_results.py --raw_dir results/raw --out_dir results
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import aggregate_seed_results, format_results_table, load_all_results, paired_ttest


REFERENCE_MODEL = "HTSPF_Full"

DATASET_DISPLAY_NAMES = {
    "cifar100":               "CIFAR-100",
    "ford_a":                 "FordA (UCR)",
    "ethanol_concentration":  "EthanolConcentration (UCR)",
    "ag_news":                "AG News (NLP)",
    "ravdess":                "RAVDESS (Audio)",
}


def compile_results(raw_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # ── Load all raw JSON results ──
    all_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    if not all_files:
        print(f"No result files found in {raw_dir}. Run the benchmark first.")
        return

    # Group by dataset
    by_dataset = {}
    for filename in sorted(all_files):
        with open(os.path.join(raw_dir, filename)) as f:
            data = json.load(f)
        dataset = data["dataset"]
        model   = data["model"]
        acc     = data.get("test_acc", 0.0)
        sparsity = data.get("sparsity", 1.0)
        gflops   = data.get("gflops", None)
        by_dataset.setdefault(dataset, {}).setdefault(model, []).append({
            "acc": acc / 100.0,
            "sparsity": sparsity,
            "gflops": gflops
        })

    # ── Process each dataset ──
    full_summary = {}

    for dataset_name, raw_results in sorted(by_dataset.items()):
        display = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
        print(f"\n{'='*60}")
        print(f"  {display}")
        print(f"{'='*60}")

        # Check we have 5 seeds
        for model, accs in raw_results.items():
            if len(accs) < 5:
                print(f"  [WARNING] {model} only has {len(accs)}/5 seeds.")

        aggregated = aggregate_seed_results(raw_results)

        # Console summary
        ref_available = REFERENCE_MODEL in aggregated
        for model_name, stats in sorted(aggregated.items(),
                                         key=lambda x: -x[1]["mean"]):
            marker = "◆" if model_name == REFERENCE_MODEL else " "
            print(f" {marker} {model_name:<30} "
                  f"{stats['mean']:.2f} ± {stats['std']:.2f}%")

        # Statistical significance vs. HTSPF_Full
        if ref_available:
            ref_raw = aggregated[REFERENCE_MODEL]["raw"]
            print(f"\n  Paired t-tests vs. {REFERENCE_MODEL}:")
            for model_name, stats in aggregated.items():
                if model_name == REFERENCE_MODEL:
                    continue
                t_result = paired_ttest(
                    [r / 100 for r in ref_raw],
                    [r / 100 for r in stats["raw"]]
                )
                sig = "✓ significant" if t_result["significant"] else "✗ not significant"
                delta = t_result["delta_mean"]
                delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
                print(f"    {model_name:<30} Δ={delta_str}pp  p={t_result['p_value']:.4f}  {sig}")

        # LaTeX table
        if ref_available:
            latex = format_results_table(aggregated, display, REFERENCE_MODEL)
            tex_path = os.path.join(out_dir, f"final_table_{dataset_name}.tex")
            with open(tex_path, "w") as f:
                f.write(latex)
            print(f"\n  LaTeX table saved to: {tex_path}")

        full_summary[dataset_name] = aggregated

    # ── Save machine-readable summary ──
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\n\nFull summary saved to: {summary_path}")
    print("Compilation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile HTSPF benchmark results")
    parser.add_argument("--raw_dir", default="results/raw")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    compile_results(args.raw_dir, args.out_dir)
