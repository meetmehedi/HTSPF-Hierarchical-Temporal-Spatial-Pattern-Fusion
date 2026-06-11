#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Force matplotlib to not use any Xwindows backend
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.dirname(__file__))
from htspf import HTSPF
from data import get_cifar100_dataloaders

def run_interpretability():
    print("Loading configuration...")
    with open("configs/experiment.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    os.makedirs("results/interpretability", exist_ok=True)

    print("Loading dataset...")
    # Load test set for explanation
    _, _, test_loader = get_cifar100_dataloaders(
        root=cfg["datasets"]["cifar100"]["root"],
        batch_size=8,
        val_split=0.1,
        num_workers=0
    )
    
    # Get a batch
    images, labels = next(iter(test_loader))
    images = images.to(device)

    print("Loading model checkpoint...")
    model = HTSPF(
        num_classes=cfg["model"]["num_classes"]["cifar100"],
        d_model=cfg["model"]["d_model"],
        ldwt_scales=cfg["model"]["ldwt_scales"],
        beta=cfg["model"]["beta"],
        gamma=cfg["model"]["gamma"],
        vision_shape=(3, 32, 32),
        patch_size=4,
        ts_channels=1
    ).to(device)

    checkpoint_path = "checkpoints/HTSPF_Full_cifar100_seed0.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Ensure training is finished.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # ---------------------------------------------------------
    # 1. Saliency (Input x Gradient)
    # ---------------------------------------------------------
    print("Computing Saliency Maps...")
    test_images = images[:4].clone().detach().requires_grad_(True)
    
    logits, _ = model(test_images, modality="vision")
    preds = logits.argmax(dim=1)
    
    # Backward pass for the predicted class
    score = logits.gather(1, preds.view(-1, 1)).sum()
    score.backward()
    
    saliency_maps = test_images.grad.abs().max(dim=1)[0].cpu().numpy() # [B, H, W]
    
    # Undo normalization for display
    mean = np.array([0.5071, 0.4867, 0.4408]).reshape(1, 3, 1, 1)
    std = np.array([0.2675, 0.2565, 0.2761]).reshape(1, 3, 1, 1)
    disp_images = test_images.detach().cpu().numpy() * std + mean
    disp_images = np.transpose(disp_images, (0, 2, 3, 1)) # to NHWC
    disp_images = np.clip(disp_images, 0, 1)
    
    fig, axes = plt.subplots(4, 2, figsize=(6, 12))
    for i in range(4):
        axes[i, 0].imshow(disp_images[i])
        axes[i, 0].set_title(f"Original (Pred: {preds[i].item()})")
        axes[i, 0].axis("off")
        
        # Plot saliency heatmaps
        sns.heatmap(saliency_maps[i], cmap="jet", ax=axes[i, 1], cbar=False)
        axes[i, 1].set_title("Saliency (Pixels)")
        axes[i, 1].axis("off")
        
    plt.tight_layout()
    plt.savefig("results/interpretability/saliency_attributions.png")
    plt.close()
    print(" -> Saved Saliency plot to results/interpretability/saliency_attributions.png")

    # ---------------------------------------------------------
    # 2. Extract ASG Gating Probabilities
    # ---------------------------------------------------------
    print("Extracting ASG sparsity activations...")
    with torch.no_grad():
        _, u_k = model(images, modality="vision")
    
    avg_activation = u_k.mean(dim=0).cpu().numpy()
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=[f"Pathway {i}" for i in range(len(avg_activation))], y=avg_activation)
    plt.title("Adaptive Sparsity Gate (ASG) Average Activation")
    plt.ylabel("Probability of Activation")
    plt.ylim(0, 1)
    plt.savefig("results/interpretability/asg_pathway_usage.png")
    plt.close()
    print(" -> Saved ASG activations to results/interpretability/asg_pathway_usage.png")

    print("Interpretability pipeline complete.")

if __name__ == "__main__":
    run_interpretability()
