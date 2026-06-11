"""
src/ablations.py — Ablation Model Variants for HTSPF Phase 3.

Each variant isolates one component's contribution by removing or replacing it.
This proves that every novel component (USE/LDWT, HCAA/CDM, ASG) adds measurable value.

Ablation table:
  HTSPF_noHCAA      — Standard MHA, no conflict detection or Fisher resolution
  HTSPF_noASG       — All pathways always active, no sparsity gating
  HTSPF_noLDWT      — Linear projector replaces LDWT in USE
  HTSPF_noConflict  — HCAA active but CDM/hard suppression disabled (β=∞, soft blend only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(__file__))
from htspf import USE, HCAA, ASG, HTSPF, htspf_loss


# ──────────────────────────────────────────────
# 1. HTSPF_noHCAA: Replace HCAA with standard MHA
# ──────────────────────────────────────────────

class StandardMHA(nn.Module):
    """
    Standard multi-head attention — no Conflict Detection Module,
    no Fisher-weighted resolution. All heads equally weighted.
    """
    def __init__(self, d_tot, num_heads=4, **kwargs):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_tot, num_heads, batch_first=True)
        self.out_proj = nn.Linear(d_tot, d_tot)

    def forward(self, E):
        out, _ = self.attn(E, E, E)
        return self.out_proj(out)


class HTSPF_noHCAA(HTSPF):
    """Ablation: Replaces all HCAA pathway modules with vanilla MHA."""
    def __init__(self, num_classes=10, d_model=128, ldwt_scales=3, **kwargs):
        super().__init__(num_classes=num_classes, d_model=d_model,
                         ldwt_scales=ldwt_scales, **kwargs)
        d_tot = d_model * (ldwt_scales + 1)
        self.hcaa_pathways = nn.ModuleList([
            StandardMHA(d_tot=d_tot) for _ in range(self.num_pathways)
        ])


# ──────────────────────────────────────────────
# 2. HTSPF_noASG: All pathways always active
# ──────────────────────────────────────────────

class AlwaysOnASG(nn.Module):
    """
    Replaces ASG with always-on passthrough — all pathways contribute equally.
    This ablation measures inference cost penalty of removing the sparsity gate.
    """
    def __init__(self, d_tot, num_pathways=1, **kwargs):
        super().__init__()
        self.num_pathways = num_pathways

    def forward(self, E, pathways):
        # Uniform average of all pathways — no gating
        out = sum(pathways) / len(pathways)
        # Return dummy u_k of all ones for compatibility with htspf_loss
        B = E.shape[0]
        u_k = torch.ones(B, self.num_pathways, device=E.device)
        return out, u_k


class HTSPF_noASG(HTSPF):
    """Ablation: Replaces ASG with always-on uniform passthrough."""
    def __init__(self, num_classes=10, d_model=128, ldwt_scales=3, **kwargs):
        super().__init__(num_classes=num_classes, d_model=d_model,
                         ldwt_scales=ldwt_scales, **kwargs)
        d_tot = d_model * (ldwt_scales + 1)
        self.asg = AlwaysOnASG(d_tot=d_tot, num_pathways=self.num_pathways)


# ──────────────────────────────────────────────
# 3. HTSPF_noLDWT: Replace LDWT with linear projector in USE
# ──────────────────────────────────────────────

class USE_LinearOnly(USE):
    """
    USE variant that replaces the LDWT decomposition with a single
    learnable linear transformation. This ablation tests whether
    the wavelet-style hierarchical decomposition adds value over
    a simple projection.
    """
    def __init__(self, d_model=128, patch_size=16, ldwt_scales=3,
                 vision_shape=(3, 224, 224), ts_channels=1):
        super().__init__(d_model=d_model, patch_size=patch_size,
                         ldwt_scales=ldwt_scales,
                         vision_shape=vision_shape, ts_channels=ts_channels)
        # Output must match the normal USE: N x D*(J+1)
        d_tot = d_model * (ldwt_scales + 1)
        # Simple linear projector from D to D_tot
        self.linear_expand = nn.Linear(d_model, d_tot)

    def forward(self, x, modality="vision"):
        # Inherit modality projection from parent USE
        if modality == "vision":
            z = self.vis_proj(x).flatten(2) + self.vis_pos  # (B, D, N)
        elif modality == "timeseries":
            z = self.ts_proj(x) + self.ts_pos               # (B, D, N)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Skip LDWT entirely — single linear expansion
        E = self.linear_expand(z.transpose(1, 2))  # (B, N, D_tot)
        return E


class HTSPF_noLDWT(HTSPF):
    """Ablation: Replaces LDWT in USE with a single linear projector."""
    def __init__(self, num_classes=10, d_model=128, ldwt_scales=3, **kwargs):
        super().__init__(num_classes=num_classes, d_model=d_model,
                         ldwt_scales=ldwt_scales, **kwargs)
        vision_shape = kwargs.get("vision_shape", (3, 32, 32))
        patch_size = kwargs.get("patch_size", 4)
        ts_channels = kwargs.get("ts_channels", 1)
        self.use = USE_LinearOnly(
            d_model=d_model,
            ldwt_scales=ldwt_scales,
            vision_shape=vision_shape,
            patch_size=patch_size,
            ts_channels=ts_channels,
        )


# ──────────────────────────────────────────────
# 4. HTSPF_noConflict: HCAA with CDM disabled (soft blend only)
# ──────────────────────────────────────────────

class HCAA_noConflict(HCAA):
    """
    HCAA variant where the hard suppression threshold β is effectively
    disabled (set to infinity). Only the soft Fisher blend is applied.
    This tests whether the hard conflict suppression in CDM is necessary,
    or if a soft weighted average is sufficient.
    """
    def __init__(self, d_tot, num_heads=4, **kwargs):
        # beta=float('inf') disables hard suppression: F(h1)/F(h2) > inf is never true
        super().__init__(d_tot=d_tot, num_heads=num_heads, beta=float('inf'))


class HTSPF_noConflict(HTSPF):
    """Ablation: HCAA active but hard suppression disabled (β=∞, soft blend only)."""
    def __init__(self, num_classes=10, d_model=128, ldwt_scales=3, **kwargs):
        super().__init__(num_classes=num_classes, d_model=d_model,
                         ldwt_scales=ldwt_scales, **kwargs)
        d_tot = d_model * (ldwt_scales + 1)
        self.hcaa_pathways = nn.ModuleList([
            HCAA_noConflict(d_tot=d_tot) for _ in range(self.num_pathways)
        ])


# ──────────────────────────────────────────────
# Ablation Registry
# ──────────────────────────────────────────────

ABLATION_REGISTRY = {
    "HTSPF_noHCAA":     HTSPF_noHCAA,
    "HTSPF_noASG":      HTSPF_noASG,
    "HTSPF_noLDWT":     HTSPF_noLDWT,
    "HTSPF_noConflict": HTSPF_noConflict,
    "HTSPF_Full":       HTSPF,    # The full model is also in the registry for unified runner
}


def get_ablation(name: str, dataset_name: str, cfg: dict) -> nn.Module:
    """
    Factory function returning the correct ablation model.
    """
    if name not in ABLATION_REGISTRY:
        raise ValueError(f"Unknown ablation: '{name}'. "
                         f"Choose from: {list(ABLATION_REGISTRY.keys())}")

    mc = cfg["model"]
    nc = mc["num_classes"][dataset_name]
    ModelClass = ABLATION_REGISTRY[name]

    # Determine vision shape and ts_channels from dataset config
    if dataset_name == "cifar100":
        dcfg = cfg["datasets"]["cifar100"]
        vision_shape = (dcfg["num_channels"], dcfg["image_size"], dcfg["image_size"])
        patch_size = dcfg["patch_size"]
        ts_channels = 1
    else:
        vision_shape = (3, 32, 32)
        patch_size = 4
        dcfg_key = dataset_name.replace("-", "_")
        ts_channels = cfg["datasets"].get(dcfg_key, {}).get("num_channels", 1)

    return ModelClass(
        num_classes=nc,
        d_model=mc["d_model"],
        ldwt_scales=mc["ldwt_scales"],
        beta=mc["beta"],
        gamma=mc["gamma"],
        vision_shape=vision_shape,
        patch_size=patch_size,
        ts_channels=ts_channels,
    )
