"""
src/train.py — Training Engine for HTSPF Phase 3 Benchmarks.

Provides:
  - set_seed: Full reproducibility
  - train_epoch: Single epoch training loop
  - evaluate: Accuracy + sparsity metric computation
  - EarlyStopping: Prevent overfitting
  - run_experiment: Full training loop for a (model, dataset, seed) tuple
"""

import os
import sys
import time
import json
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(__file__))
from data import get_dataloaders, set_seed
from htspf import htspf_loss, HTSPF
from baselines import get_baseline
from ablations import get_ablation, ABLATION_REGISTRY
from metrics import save_result, compute_flops


# ──────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────

class EarlyStopping:
    """Stops training when val loss hasn't improved for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, lambda_sparsity: float,
                modality: str, log_interval: int = 50):
    """
    Runs one full training epoch.

    Returns:
        avg_loss: Average total loss (task + sparsity)
        avg_acc:  Average top-1 accuracy
        avg_sparsity: Average pathway activation E[m_k]
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    total_sparsity = []

    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        logits, u_k = model(X, modality=modality)
        loss = htspf_loss(logits, y, u_k if u_k is not None else torch.zeros(1),
                          lambda_sparsity=lambda_sparsity)

        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)
        total_loss += loss.item()

        if u_k is not None:
            total_sparsity.append(u_k.mean().item())

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            running_acc = total_correct / total_samples * 100
            print(f"  [Batch {batch_idx+1}/{len(loader)}] "
                  f"loss={total_loss / (batch_idx+1):.4f}  acc={running_acc:.2f}%")

    avg_sparsity = float(np.mean(total_sparsity)) if total_sparsity else 1.0
    return (total_loss / len(loader),
            total_correct / total_samples,
            avg_sparsity)


# ──────────────────────────────────────────────
# Evaluation Loop
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, modality: str, lambda_sparsity: float = 0.0):
    """
    Evaluates model on a dataloader.

    Returns:
        dict with keys: loss, accuracy, sparsity
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    total_sparsity = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits, u_k = model(X, modality=modality)
        loss = htspf_loss(logits, y, u_k if u_k is not None else torch.zeros(1),
                          lambda_sparsity=lambda_sparsity)

        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)
        total_loss += loss.item()

        if u_k is not None:
            total_sparsity.append(u_k.mean().item())

    return {
        "loss":     total_loss / len(loader),
        "accuracy": total_correct / total_samples,
        "sparsity": float(np.mean(total_sparsity)) if total_sparsity else 1.0,
    }


# ──────────────────────────────────────────────
# Full Experiment Runner
# ──────────────────────────────────────────────

def run_experiment(model_name: str, dataset_name: str, seed: int,
                   cfg: dict, device: torch.device,
                   fast: bool = False, checkpoint_dir: str = None) -> dict:
    """
    Runs a complete training + evaluation pipeline for one (model, dataset, seed) triple.

    Args:
        model_name: 'HTSPF_Full', 'HTSPF_noHCAA', 'resnet18', 'inception_time', etc.
        dataset_name: 'cifar100', 'ford_a', 'ethanol_concentration'
        seed: Random seed for this run
        cfg: Loaded experiment.yaml config dict
        device: torch.device
        fast: If True, runs only 2 epochs (for dry-run testing)
        checkpoint_dir: If set, saves best model checkpoint here

    Returns:
        results dict with test_acc, test_loss, sparsity, gflops, time_s
    """
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name} | Dataset: {dataset_name} | Seed: {seed}")
    print(f"{'='*60}")

    # ── Data ──
    train_loader, val_loader, test_loader, modality = \
        get_dataloaders(dataset_name, cfg, seed=seed)

    # ── Model ──
    if model_name in ABLATION_REGISTRY:
        model = get_ablation(model_name, dataset_name, cfg)
    else:
        model = get_baseline(model_name, dataset_name, cfg)
    model = model.to(device)

    # ── Optimizer & Scheduler ──
    tc = cfg["training"]
    if tc["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=tc["lr"],
                                weight_decay=tc["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=tc["lr"],
                              momentum=0.9, weight_decay=tc["weight_decay"])

    n_epochs = 2 if fast else tc["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    early_stopping = EarlyStopping(patience=tc["early_stopping_patience"])
    lambda_sparsity = tc["lambda_sparsity"]
    log_interval = cfg["experiment"]["log_interval"]

    best_val_acc = 0.0
    best_model_state = None
    start_time = time.time()

    # ── Training Loop ──
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc, train_sparsity = train_epoch(
            model, train_loader, optimizer, device,
            lambda_sparsity, modality, log_interval if not fast else -1
        )
        val_metrics = evaluate(model, val_loader, device, modality, lambda_sparsity)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{n_epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
              f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']*100:.2f}%  "
              f"sparsity={val_metrics['sparsity']:.3f}")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        early_stopping.step(val_metrics["loss"])
        if early_stopping.should_stop and not fast:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    elapsed = time.time() - start_time

    # ── Restore best model and evaluate on test set ──
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(device)
    test_metrics = evaluate(model, test_loader, device, modality, lambda_sparsity=0.0)

    print(f"\n>>> TEST RESULTS: acc={test_metrics['accuracy']*100:.2f}%  "
          f"sparsity={test_metrics['sparsity']:.3f}  time={elapsed:.1f}s")

    # ── FLOPs Estimation ──
    try:
        sample_X = next(iter(test_loader))[0][:1].to(device)
        gflops = compute_flops(model, tuple(sample_X.shape[1:]), modality=modality)
    except Exception:
        gflops = None

    # ── Save Checkpoint ──
    if checkpoint_dir and best_model_state is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir,
                                  f"{model_name}_{dataset_name}_seed{seed}.pt")
        torch.save(best_model_state, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    return {
        "test_acc":  round(test_metrics["accuracy"] * 100, 4),
        "test_loss": round(test_metrics["loss"], 6),
        "sparsity":  round(test_metrics["sparsity"], 4),
        "gflops":    round(gflops, 4) if gflops is not None else None,
        "time_s":    round(elapsed, 1),
        "epochs_run": epoch,
    }


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTSPF Training Script")
    parser.add_argument("--model",   required=True,
                        help="Model name (e.g., HTSPF_Full, resnet18, inception_time)")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g., cifar100, ford_a)")
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--config",  default="configs/experiment.yaml")
    parser.add_argument("--fast",    action="store_true",
                        help="Dry-run mode: 2 epochs only")
    parser.add_argument("--output",  default=None,
                        help="Override output directory for results JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    results = run_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        seed=args.seed,
        cfg=cfg,
        device=device,
        fast=args.fast,
        checkpoint_dir=cfg["experiment"].get("checkpoint_dir"),
    )

    out_dir = args.output or os.path.join(cfg["experiment"]["output_dir"], "raw")
    save_result(out_dir, args.model, args.dataset, args.seed, results)
    print(f"\nResults saved to: {out_dir}/{args.model}_{args.dataset}_seed{args.seed}.json")
