# HTSPF: Architecture Specification Document
**Phase 1: Theoretical Formalization**

This document provides the mathematical formalization of the Hierarchical Temporal-Spatial Pattern Fusion (HTSPF) framework. Since the framework must operate across diverse modalities (specifically Vision and Time-Series in this phase), the architecture is defined by three core components.

---

## 1. Universal Signal Embedding (USE)

The objective of USE is to project heterogeneous input modalities into a shared 2D frequency-context manifold $M \in \mathbb{R}^{F \times C}$ where $F$ represents frequency/scale dimensions and $C$ represents context/spatial dimensions.

### 1.1 Modality-Specific Projections
To resolve dimensional mismatches, raw inputs are first mapped to a uniform $D$-dimensional sequence space.

**For Vision (Images):**
Given an image $X_v \in \mathbb{R}^{H \times W \times 3}$, we extract non-overlapping patches $x_{v,i} \in \mathbb{R}^{P^2 \cdot 3}$. A linear projection aligns these patches:
$$ Z_v = [x_{v,1}W_v; x_{v,2}W_v; \dots; x_{v,N}W_v] + E_{pos,v} $$
where $N = HW/P^2$ is the sequence length, $W_v \in \mathbb{R}^{(P^2 \cdot 3) \times D}$ is the projection weight, and $E_{pos,v}$ is a learned 2D positional embedding.

**For Time-Series:**
Given a multivariate time-series $X_t \in \mathbb{R}^{T \times C_{in}}$, we apply a 1D convolution to extract local temporal features:
$$ Z_t = \text{Conv1D}(X_t; W_t) + E_{pos,t} $$
where $W_t \in \mathbb{R}^{K \times C_{in} \times D}$ and $E_{pos,t}$ is a 1D positional embedding. Stride and padding are configured such that the output sequence length $T'$ equals the vision sequence length $N$ (i.e., $T' = N$), ensuring dimensional consistency across modalities.

### 1.2 Learnable Discrete Wavelet Transform (LDWT)
Both modalities now exist as sequences $Z \in \mathbb{R}^{N \times D}$. The LDWT parameterizes the high-pass $H_{\theta}$ and low-pass $L_{\theta}$ filters as learnable weights, replacing fixed wavelet bases (like Haar or Daubechies) with adaptive ones. We define the base case as $Z_0 := Z$.

At scale $j$:
$$ A_j = Z_{j-1} \circledast L_{\theta}^{(j)} \downarrow 2 \quad \text{(Approximation Coefficients)} $$
$$ D_j = Z_{j-1} \circledast H_{\theta}^{(j)} \downarrow 2 \quad \text{(Detail Coefficients)} $$
To ensure unambiguous recursion for subsequent scales, we explicitly define the input to the next scale as the current approximation coefficients: $Z_j := A_j$.

Since consecutive downsampling reduces the sequence length (e.g., $A_J \in \mathbb{R}^{N/2^J \times D}$), we apply 1D interpolation to upsample all coefficients back to length $N$ before concatenation along the feature dimension. The shared latent manifold $E$ is thus:
$$ E = [\text{Up}(A_J); \text{Up}(D_J); \text{Up}(D_{J-1}); \dots; \text{Up}(D_1)] $$
where $E \in \mathbb{R}^{N \times (J+1)D}$.

---

## 2. Hierarchical Conflict-Aware Attention (HCAA)

HCAA resolves contradictory pattern hypotheses extracted from the manifold $E$. 

### 2.1 Multi-Scale Attention
Standard self-attention is applied at multiple hierarchical levels:
$$ \text{Attn}^{(j)}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V $$
Let $h_i^{(j)}$ represent the $i$-th pattern hypothesis (attention head output) at scale $j$.

### 2.2 Conflict Detection Module (CDM)
**Note on Inference:** The CDM operates *only during training*. A conflict occurs when two hypotheses $h_1$ and $h_2$ produce contradictory gradients with respect to a target representation or downstream objective. We define the conflict score $C(h_1, h_2)$ as the negative cosine similarity of their projected gradients:
$$ C(h_1, h_2) = \max(0, -\cos(\nabla h_1, \nabla h_2)) $$
During inference, this gradient computation is bypassed; conflicts are pre-resolved via the learned Fisher priors (detailed below), ensuring zero dynamic overhead.

### 2.3 Fisher-Weighted Resolution
To resolve high-conflict pairs ($C > \tau$) during training, and to statically resolve them at inference time, we use the offline-computed diagonal Fisher Information Matrix $\mathcal{F}$ associated with the weights generating each hypothesis. The Fisher score $F(h)$ serves as a proxy for the hypothesis's historical importance.

The resolved output $\tilde{h}$ uses a Fisher-weighted strategy:
$$ \alpha_1 = \frac{F(h_1)}{F(h_1) + F(h_2)}, \quad \alpha_2 = \frac{F(h_2)}{F(h_1) + F(h_2)} $$
To enforce hard suppression of the lower-confidence pattern, we apply a threshold $\beta > 1$:
$$ \text{if } \frac{F(h_1)}{F(h_2)} > \beta, \text{ then } \alpha_2 = 0 \text{ and } \alpha_1 = 1 $$
When no single hypothesis dominates (i.e., $F(h_1)/F(h_2) \leq \beta$ and $F(h_2)/F(h_1) \leq \beta$), the standard soft blend derived from the Fisher scores applies.
$$ \tilde{h} = \alpha_1 h_1 + \alpha_2 h_2 $$
This explicitly suppresses the lower-confidence (lower Fisher score) pattern, directly adapting gradient conflict resolution from continual learning (e.g., MEWC-LLM) into the attention space.

---

## 3. Adaptive Sparsity Gate (ASG)

To enable deployment on resource-constrained hardware, the ASG prunes redundant pattern pathways during inference.

### 3.1 Input-Conditioned Gating
For each pathway $p_k$, a lightweight gating network $g_k(\cdot)$ predicts its utility based on the manifold $E$. To collapse the sequence dimension, we first apply Global Average Pooling (GAP) to $E$:
$$ u_k = \sigma(W_{gate}^{(k)} \cdot \text{GAP}(E) + b_{gate}^{(k)}) $$
where $\text{GAP}(E) \in \mathbb{R}^{(J+1)D}$ and $W_{gate}^{(k)} \in \mathbb{R}^{1 \times (J+1)D}$ (indicating each pathway has its own independent gate weights), yielding a scalar utility $u_k$.

A hard concrete distribution or straight-through estimator generates a binary mask $m_k \in \{0, 1\}$ during the forward pass:
$$ m_k = \mathbb{I}(u_k > \gamma) $$
The pathway output is masked: $\hat{p}_k = m_k \cdot p_k$. If $m_k = 0$, the pathway is skipped entirely.

### 3.2 Sparsity Regularization
To train the ASG, a sparsity penalty is added to the total loss:
$$ \mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \frac{1}{K} \sum_{k=1}^K u_k $$
where $\lambda$ controls the trade-off between accuracy and computational cost. By adjusting $\gamma$ at inference time, HTSPF can dynamically scale its FLOPs without retraining.

---

## 4. Theoretical Properties

**Property 1 (Conflict Bound):**
Given the Fisher resolution $\tilde{h}$, the variance of the resulting gradients is bounded by the dominant hypothesis, ensuring that optimization does not diverge during cross-modal transfer.
*Proof Sketch:* Let $g_1 = \nabla h_1$ and $g_2 = \nabla h_2$. If $\cos(g_1, g_2) < 0$ (a conflict) and $F(h_1) \gg F(h_2)$, the thresholding mechanism sets $\alpha_2 = 0$. The combined gradient $\nabla \tilde{h} = \alpha_1 g_1 + \alpha_2 g_2$ collapses to exactly $g_1$. Since $g_1$ corresponds to the historically stable Fisher prior, the variance of $\nabla \tilde{h}$ is strictly bounded by the variance of $g_1$, preventing the catastrophic interference that would otherwise occur from summing conflicting gradients. (Note: The variance bounds for the soft blend case, where neither hypothesis fully dominates, will be formally bounded in terms of $\text{Var}(g_1)$ and $\text{Var}(g_2)$ in the full paper. Full proof deferred to the full paper).

**Property 2 (Sparsity Cost Reduction):**
Assuming $K$ pathways each requiring $O(FLOPs)$ computation, the expected cost at inference time is $O(FLOPs) \cdot \mathbb{E}[m_k]$, strictly bounded by the hyperparameter $\lambda$ chosen during training.

