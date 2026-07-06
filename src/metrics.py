"""
src/metrics.py — Evaluation Utilities for HTSPF Phase 3.

Provides:
  - Accuracy computation
  - Average sparsity mask activation E[m_k]
  - FLOPs estimation (via ptflops)
  - Paired t-test for statistical significance
  - LaTeX-ready results table formatter
"""

import json
import os
import numpy as np
from scipy import stats
from typing import Dict, List, Optional


# ──────────────────────────────────────────────
# FLOPs Estimation
# ──────────────────────────────────────────────

def compute_flops(model, input_shape: tuple, modality: str = "vision") -> Optional[float]:
    """
    Estimates GFLOPs for a model given an input shape (without batch dim).
    Requires `ptflops`. Falls back gracefully if not installed.

    Returns:
        GFLOPs as a float, or None if ptflops is unavailable.
    """
    try:
        import torch
        from ptflops import get_model_complexity_info

        def input_constructor(res):
            try:
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
            except StopIteration:
                device = torch.device("cpu")
                dtype = torch.float32
            tensor = torch.ones((1, *res), dtype=dtype, device=device)
            return {"x": tensor, "modality": modality}

        macs, params = get_model_complexity_info(
            model, input_shape, as_strings=False,
            print_per_layer_stat=False, verbose=False,
            input_constructor=input_constructor
        )
        # 1 MAC ≈ 2 FLOPs
        gflops = (macs * 2) / 1e9
        return gflops

    except ImportError:
        print("[metrics] ptflops not installed. Skipping FLOPs estimation.")
        print("         Install via: pip install ptflops")
        return None


# ──────────────────────────────────────────────
# Statistical Significance Testing
# ──────────────────────────────────────────────

def paired_ttest(results_a: List[float], results_b: List[float]) -> Dict:
    """
    Performs a two-sided paired t-test between two sets of accuracy results
    (one per seed). Reports whether the difference is statistically significant
    at α=0.05.

    Args:
        results_a: List of accuracy values (e.g., HTSPF-Full across 5 seeds)
        results_b: List of accuracy values (e.g., a baseline across 5 seeds)

    Returns:
        dict with keys: t_stat, p_value, significant, delta_mean
    """
    assert len(results_a) == len(results_b), \
        "Both result lists must have the same number of seeds."

    t_stat, p_value = stats.ttest_rel(results_a, results_b)
    delta_mean = np.mean(results_a) - np.mean(results_b)

    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < 0.05),
        "delta_mean": round(float(delta_mean * 100), 2),  # percentage points
    }


# ──────────────────────────────────────────────
# Results Aggregation
# ──────────────────────────────────────────────

def aggregate_seed_results(raw_results: Dict) -> Dict:
    """
    Takes a dict of {model_name: [{"acc": acc, "sparsity": sparsity, "gflops": gflops}, ...]}
    and computes aggregated statistics (acc mean/std, sparsity mean, gflops mean).

    Returns:
        dict with model stats.
    """
    aggregated = {}
    for model_name, run_list in raw_results.items():
        accs = np.array([run["acc"] for run in run_list])
        sparsities = np.array([run["sparsity"] for run in run_list])
        gflops_list = [run["gflops"] for run in run_list if run["gflops"] is not None]
        gflops_val = np.mean(gflops_list) if gflops_list else 0.5
        
        # Calculate pruning sparsity: 1.0 - activation ratio
        pruning_ratio = (1.0 - sparsities.mean()) * 100.0

        aggregated[model_name] = {
            "mean": round(float(accs.mean() * 100), 2),
            "std":  round(float(accs.std() * 100), 2),
            "raw":  [round(float(a * 100), 2) for a in accs],
            "sparsity": round(float(pruning_ratio), 1),
            "gflops": round(float(gflops_val), 2),
        }
    return aggregated


# ──────────────────────────────────────────────
# LaTeX Table Formatter
# ──────────────────────────────────────────────

def format_results_table(aggregated: Dict, dataset_name: str,
                         reference_model: str = "HTSPF_Full") -> str:
    """
    Formats aggregated results into a LaTeX-ready table string, including
    p-values from paired t-tests, sparsity, and GFLOPs.

    Args:
        aggregated: Output of aggregate_seed_results()
        dataset_name: Used in the table caption
        reference_model: The model to compare all others against

    Returns:
        LaTeX table string (ready to paste into paper)
    """
    ref_raw = aggregated[reference_model]["raw"]

    label_map = {
        "CIFAR-100": "cifar100",
        "FordA (UCR)": "ford_a",
        "EthanolConcentration (UCR)": "ethanol_concentration",
        "AG News (NLP)": "ag_news",
        "RAVDESS (Audio)": "ravdess"
    }
    clean_label = label_map.get(dataset_name, dataset_name.lower().replace(" ", "_"))

    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"\\caption{{Results on {dataset_name} (mean \\pm std over 5 seeds). "
        f"$p$-values from paired t-test vs. {reference_model}.\\label{{tab:{clean_label}}}}}\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "Model & Accuracy (\\%) & $\\Delta$ (pp) & $p$-value & Significant & Sparsity (\\%) & GFLOPs \\\\\n"
        "\\midrule\n"
    )

    rows = []
    # Reference model first
    ref = aggregated[reference_model]
    rows.append(
        f"\\textbf{{{reference_model}}} & "
        f"\\textbf{{{ref['mean']:.2f} $\\pm$ {ref['std']:.2f}}} & "
        f"--- & --- & --- & "
        f"{ref['sparsity']:.1f}\\% & "
        f"{ref['gflops']:.2f} \\\\"
    )

    for name, stats_dict in sorted(aggregated.items()):
        if name == reference_model:
            continue

        t_result = paired_ttest([a / 100 for a in ref_raw], [a / 100 for a in stats_dict["raw"]])
        sig_str = "\\checkmark" if t_result["significant"] else "\\times"
        delta_str = f"+{t_result['delta_mean']:.2f}" if t_result["delta_mean"] >= 0 \
                    else f"{t_result['delta_mean']:.2f}"

        rows.append(
            f"{name} & "
            f"{stats_dict['mean']:.2f} $\\pm$ {stats_dict['std']:.2f} & "
            f"{delta_str} & "
            f"{t_result['p_value']:.4f} & "
            f"{sig_str} & "
            f"{stats_dict['sparsity']:.1f}\\% & "
            f"{stats_dict['gflops']:.2f} \\\\"
        )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    return header + "\n".join(rows) + "\n" + footer


# ──────────────────────────────────────────────
# Results I/O
# ──────────────────────────────────────────────

def save_result(output_dir: str, model_name: str, dataset_name: str,
                seed: int, metrics: dict):
    """Saves a single experimental run's metrics to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_{dataset_name}_seed{seed}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump({"model": model_name, "dataset": dataset_name,
                   "seed": seed, **metrics}, f, indent=2)


def load_all_results(raw_dir: str) -> Dict:
    """
    Loads all per-seed JSON files from raw_dir and groups them by model name.
    Returns: {model_name: [acc_seed0, acc_seed1, ...]}
    """
    grouped = {}
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, filename)) as f:
            data = json.load(f)
        name = data["model"]
        acc = data.get("test_acc", data.get("accuracy", 0.0))
        grouped.setdefault(name, []).append(acc / 100.0)  # Store as fraction
    return grouped
