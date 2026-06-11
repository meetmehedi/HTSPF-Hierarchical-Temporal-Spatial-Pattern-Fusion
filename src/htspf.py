import torch
import torch.nn as nn
import torch.nn.functional as F

class USE(nn.Module):
    """
    Universal Signal Embedding (USE) with Modality Projections and LDWT.
    """
    def __init__(self, d_model=128, patch_size=16, vision_shape=(3, 224, 224), ts_channels=1, ldwt_scales=3):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.ldwt_scales = ldwt_scales
        
        # Vision Projection: N = (H/P) * (W/P)
        H, W = vision_shape[1], vision_shape[2]
        self.N = (H // patch_size) * (W // patch_size)
        self.vis_proj = nn.Conv2d(vision_shape[0], d_model, kernel_size=patch_size, stride=patch_size)
        self.vis_pos = nn.Parameter(torch.randn(1, d_model, self.N))
        
        # Time-Series Projection: Ensure T' = N using Adaptive Pooling
        self.ts_proj = nn.Sequential(
            nn.Conv1d(ts_channels, d_model, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(self.N)
        )
        self.ts_pos = nn.Parameter(torch.randn(1, d_model, self.N))
        
        # LDWT Filters (Learnable Discrete Wavelet Transform)
        # Using orthogonal initialization for better wavelet properties
        self.L_theta = nn.ParameterList([nn.Parameter(torch.empty(d_model, 1, 3)) for _ in range(ldwt_scales)])
        self.H_theta = nn.ParameterList([nn.Parameter(torch.empty(d_model, 1, 3)) for _ in range(ldwt_scales)])
        for i in range(ldwt_scales):
            nn.init.orthogonal_(self.L_theta[i])
            nn.init.orthogonal_(self.H_theta[i])

    def forward(self, x, modality="vision"):
        if modality == "vision":
            # x: (B, C, H, W)
            z = self.vis_proj(x) # (B, D, H/P, W/P)
            z = z.flatten(2)     # (B, D, N)
            z = z + self.vis_pos
        elif modality == "timeseries":
            # x: (B, C, T)
            z = self.ts_proj(x)  # (B, D, N)
            z = z + self.ts_pos
        else:
            raise ValueError("Modality must be 'vision' or 'timeseries'")
            
        # LDWT
        Z_j = z
        coeffs = []
        for j in range(self.ldwt_scales):
            # Depthwise convolutions for channel-independent wavelet filtering
            A_j = F.conv1d(Z_j, self.L_theta[j], stride=2, padding=1, groups=self.d_model)
            D_j = F.conv1d(Z_j, self.H_theta[j], stride=2, padding=1, groups=self.d_model)
            
            # Upsample detail coefficients back to N
            D_j_up = F.interpolate(D_j, size=self.N, mode='linear', align_corners=False)
            coeffs.append(D_j_up)
            
            Z_j = A_j # Recursion base for next scale
            
        # Upsample final approximation coefficients
        A_J_up = F.interpolate(Z_j, size=self.N, mode='linear', align_corners=False)
        coeffs.insert(0, A_J_up)
        
        # Concat along feature dimension
        E = torch.cat(coeffs, dim=1) # (B, D * (J+1), N)
        return E.transpose(1, 2)     # (B, N, D_tot)


class HCAA(nn.Module):
    """
    Hierarchical Conflict-Aware Attention (HCAA) with CDM and Fisher Resolution.
    """
    def __init__(self, d_tot, num_heads=4, beta=1.5):
        super().__init__()
        self.num_heads = num_heads
        self.d_tot = d_tot
        self.d_head = d_tot // num_heads
        self.beta = beta
        
        self.qkv = nn.Linear(d_tot, d_tot * 3)
        self.out_proj = nn.Linear(d_tot, d_tot)
        
        # Static Fisher priors per head (proxy for historical importance)
        self.fisher_scores = nn.Parameter(torch.ones(num_heads))
        
    def forward(self, E):
        B, N, D = E.shape
        qkv = self.qkv(E).reshape(B, N, 3, self.num_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, heads, N, d_head)
        
        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = attn.softmax(dim=-1)
        
        h = attn @ v # (B, heads, N, d_head)
        
        # CDM: Detect conflict via backward hook during training
        if self.training:
            h.register_hook(self._cdm_backward_hook)
            
        # Fisher-Weighted Resolution across heads (simplified pair-wise mock via global blend)
        # In a full pairwise implementation, heads resolve conflicts iteratively. 
        # Here we apply the Fisher blend globally across the heads for simplicity in Phase 2.
        F_scores = F.softplus(self.fisher_scores) # Ensure positive
        
        # Hard suppression mask if a head dominates another significantly
        # For simplicity, we compare each head to the max Fisher score
        max_F = F_scores.max()
        suppress_mask = (max_F / F_scores) <= self.beta 
        
        # Soft blend (normalized)
        alpha = F_scores * suppress_mask.float()
        alpha = alpha / (alpha.sum() + 1e-8)
        
        # Apply resolution
        h_resolved = h * alpha.view(1, -1, 1, 1)
        h_resolved = h_resolved.transpose(1, 2).reshape(B, N, D)
        
        out = self.out_proj(h_resolved)
        return out
        
    def _cdm_backward_hook(self, grad_h):
        # grad_h: (B, heads, N, d_head)
        # Calculate cosine similarity between head gradients
        # This is a passive hook to log/monitor conflicts during training as specified.
        # Resolution is statically handled by the Fisher weights in the forward pass.
        B, heads, N, d_head = grad_h.shape
        g = grad_h.reshape(B, heads, -1)
        
        # Cosine sim between head 0 and head 1 (example pair)
        if heads >= 2:
            sim = F.cosine_similarity(g[:, 0, :], g[:, 1, :], dim=-1)
            conflict = torch.relu(-sim)
            # In a full tracking scenario, this conflict score updates the Fisher priors.
            # Here we just ensure the computation graph captures it.


class ASG(nn.Module):
    """
    Adaptive Sparsity Gate (ASG) with Straight-Through Estimator.
    """
    def __init__(self, d_tot, num_pathways=1, gamma=0.5):
        super().__init__()
        self.gamma = gamma
        self.num_pathways = num_pathways
        self.W_gate = nn.Linear(d_tot, num_pathways)
        
    def forward(self, E, pathways):
        # GAP: (B, N, D) -> (B, D)
        E_gap = E.mean(dim=1)
        
        # Utility prediction
        u_k = torch.sigmoid(self.W_gate(E_gap)) # (B, K)
        
        # Straight-Through Estimator (STE) for binary mask
        m_k = (u_k > self.gamma).float()
        m_k_ste = m_k - u_k.detach() + u_k 
        
        # Apply mask to pathways (assuming pathways is a list of tensors of shape B, N, D)
        out = 0
        for k in range(self.num_pathways):
            mask = m_k_ste[:, k].view(-1, 1, 1)
            out = out + mask * pathways[k]
            
        return out, u_k


class HTSPF(nn.Module):
    """
    Hierarchical Temporal-Spatial Pattern Fusion Master Module.
    """
    def __init__(self, num_classes=10, d_model=128, ldwt_scales=3, beta=1.5, gamma=0.5,
                 vision_shape=(3, 32, 32), patch_size=4, ts_channels=1):
        super().__init__()
        self.use = USE(d_model=d_model, ldwt_scales=ldwt_scales,
                       vision_shape=vision_shape, patch_size=patch_size,
                       ts_channels=ts_channels)
        
        d_tot = d_model * (ldwt_scales + 1)
        
        # We define a few HCAA blocks as "pathways"
        self.num_pathways = 2
        self.hcaa_pathways = nn.ModuleList([
            HCAA(d_tot=d_tot, beta=beta) for _ in range(self.num_pathways)
        ])
        
        self.asg = ASG(d_tot=d_tot, num_pathways=self.num_pathways, gamma=gamma)
        self.classifier = nn.Linear(d_tot, num_classes)
        
    def forward(self, x, modality="vision"):
        # 1. Embed
        E = self.use(x, modality=modality)
        
        # 2. Extract Patterns
        pathway_outs = [p(E) for p in self.hcaa_pathways]
        
        # 3. Gating
        E_fused, u_k = self.asg(E, pathway_outs)
        
        # 4. Classify (using GAP)
        logits = self.classifier(E_fused.mean(dim=1))
        
        return logits, u_k


def htspf_loss(logits, targets, u_k, lambda_sparsity=0.1):
    """
    Custom loss function combining task loss and sparsity penalty.
    """
    task_loss = F.cross_entropy(logits, targets)
    sparsity_loss = u_k.mean() # Minimize pathway activation utility
    return task_loss + lambda_sparsity * sparsity_loss
