"""
src/baselines.py — Baseline Model Wrappers for HTSPF Phase 3 Benchmarks.

Each baseline is tied to a specific research claim:
  - ResNet-18:      Lower bound vs. classic CNN (vision)
  - ViT-Small:      HTSPF vs. purpose-built vision transformer (vision)
  - InceptionTime:  HTSPF vs. best deep TSC model (time-series)
  - PerceiverIO:    HTSPF's explicit conflict resolution vs. naive unified arch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1. ResNet-18 for CIFAR (vision)
# ──────────────────────────────────────────────

class ResNet18Baseline(nn.Module):
    """
    Standard ResNet-18 adapted for CIFAR-100 (32x32 inputs).
    Replaces the initial 7x7 conv + maxpool with a smaller 3x3 conv
    which is standard practice for CIFAR.
    """
    def __init__(self, num_classes=100):
        super().__init__()
        import torchvision.models as models
        self.model = models.resnet18(weights=None)
        # CIFAR adaptation: smaller initial conv, no maxpool
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x, modality=None):
        return self.model(x), None   # Return None for u_k (no ASG)


# ──────────────────────────────────────────────
# 2. ViT-Small for CIFAR (vision)
# ──────────────────────────────────────────────

class ViTSmallBaseline(nn.Module):
    """
    ViT-Small trained from scratch on CIFAR-100.
    Uses the torchvision ViT-B/16 architecture scaled down.
    """
    def __init__(self, num_classes=100, image_size=32, patch_size=4):
        super().__init__()
        from torchvision.models import VisionTransformer
        self.model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=6,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            num_classes=num_classes,
        )

    def forward(self, x, modality=None):
        return self.model(x), None


# ──────────────────────────────────────────────
# 3. InceptionTime for Time-Series
# ──────────────────────────────────────────────

class InceptionModule(nn.Module):
    """
    Single InceptionTime module with 3 parallel conv branches + bottleneck.
    From: Fawaz et al. (2020) "InceptionTime: Finding AlexNet for Time Series."
    """
    def __init__(self, in_channels, n_filters=32, kernel_sizes=(9, 19, 39)):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv1d(n_filters, n_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
        )
        self.bn = nn.BatchNorm1d((len(kernel_sizes) + 1) * n_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        branch_outs = [conv(bottleneck) for conv in self.convs]
        branch_outs.append(self.max_pool_conv(x))
        out = torch.cat(branch_outs, dim=1)
        return self.relu(self.bn(out))


class InceptionTimeBaseline(nn.Module):
    """
    InceptionTime: 3 stacked Inception modules with residual connections.
    Gold-standard deep learning baseline for time-series classification.
    """
    def __init__(self, in_channels=1, num_classes=2, n_filters=32):
        super().__init__()
        out_channels = (3 + 1) * n_filters  # 4 branches x n_filters

        self.block1 = InceptionModule(in_channels, n_filters)
        self.res1   = nn.Conv1d(in_channels,  out_channels, kernel_size=1, bias=False)
        self.bn1    = nn.BatchNorm1d(out_channels)

        self.block2 = InceptionModule(out_channels, n_filters)
        self.block3 = InceptionModule(out_channels, n_filters)
        self.res2   = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2    = nn.BatchNorm1d(out_channels)

        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(out_channels, num_classes)
        self.relu   = nn.ReLU()

    def forward(self, x, modality=None):
        # Block 1 + residual
        out = self.block1(x)
        res = self.relu(self.bn1(self.res1(x)))
        out = self.relu(out + res)

        # Blocks 2-3 + residual
        residual = out
        out = self.block2(out)
        out = self.block3(out)
        res = self.relu(self.bn2(self.res2(residual)))
        out = self.relu(out + res)

        out = self.gap(out).squeeze(-1)
        return self.fc(out), None


# ──────────────────────────────────────────────
# 4. Perceiver IO (lightweight cross-domain)
# ──────────────────────────────────────────────

class PerceiverCrossAttention(nn.Module):
    def __init__(self, latent_dim, input_dim, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.kv_proj = nn.Linear(input_dim, latent_dim * 2)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, latents, inputs):
        q = self.q_proj(latents)
        k, v = self.kv_proj(inputs).chunk(2, dim=-1)
        out, _ = self.attn(q, k, v)
        return self.out_proj(out)


class PerceiverIOBaseline(nn.Module):
    """
    Lightweight Perceiver IO for cross-domain input.
    Based on: Jaegle et al. (2021) "Perceiver: General Perception with Iterative Attention."
    
    Uses a fixed latent array that cross-attends to the input, then applies
    standard self-attention in the latent space. No modality-specific LDWT.
    """
    def __init__(self, input_dim=3, input_len=196, num_classes=100,
                 latent_dim=256, num_latents=64, num_self_attn_layers=4, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))

        self.cross_attn   = PerceiverCrossAttention(latent_dim, latent_dim, num_heads)
        self.self_attn    = nn.Sequential(*[
            nn.TransformerEncoderLayer(latent_dim, num_heads, dim_feedforward=latent_dim*4,
                                       batch_first=True, dropout=0.1)
            for _ in range(num_self_attn_layers)
        ])
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x, modality=None):
        B = x.shape[0]
        # Flatten input: (B, C, H, W) -> (B, N, C) for vision
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)   # (B, N, C)
        elif x.dim() == 3:
            x = x.transpose(1, 2)               # (B, T, C)

        inputs = self.input_proj(x)
        latents = self.latents.expand(B, -1, -1)

        latents = self.cross_attn(latents, inputs)
        latents = self.self_attn(latents)

        logits = self.fc(latents.mean(dim=1))
        return logits, None


# ──────────────────────────────────────────────
# Baseline Registry
# ──────────────────────────────────────────────

def get_baseline(name: str, dataset_name: str, cfg: dict) -> nn.Module:
    """
    Factory function that returns the correct baseline model for the
    given name and dataset. Uses cfg to read num_classes.
    """
    nc = cfg["model"]["num_classes"][dataset_name]

    if name == "resnet18":
        return ResNet18Baseline(num_classes=nc)

    elif name == "vit_small":
        return ViTSmallBaseline(
            num_classes=nc,
            image_size=cfg["datasets"]["cifar100"]["image_size"],
            patch_size=cfg["datasets"]["cifar100"]["patch_size"],
        )

    elif name == "inception_time":
        dcfg_key = dataset_name.replace("-", "_")
        in_channels = cfg["datasets"].get(dcfg_key, {}).get("num_channels", 1)
        return InceptionTimeBaseline(in_channels=in_channels, num_classes=nc)

    elif name == "perceiver_io":
        return PerceiverIOBaseline(
            input_dim=cfg["datasets"]["cifar100"]["num_channels"],
            num_classes=nc,
        )

    else:
        raise ValueError(f"Unknown baseline: {name}")
