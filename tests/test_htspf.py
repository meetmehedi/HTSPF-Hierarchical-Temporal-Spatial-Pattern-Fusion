import torch
import sys
import os

# Add src to path to import htspf
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from htspf import USE, HCAA, ASG, HTSPF, htspf_loss

def test_use_vision():
    model = USE(d_model=64, patch_size=16, vision_shape=(3, 224, 224), ldwt_scales=2)
    x = torch.randn(2, 3, 224, 224)
    E = model(x, modality="vision")
    
    # N = 224/16 * 224/16 = 14 * 14 = 196
    # D_tot = 64 * (2 + 1) = 192
    assert E.shape == (2, 196, 192), f"Expected (2, 196, 192), got {E.shape}"
    print("USE Vision Test: PASS")

def test_use_timeseries():
    model = USE(d_model=64, patch_size=16, ts_channels=12, ldwt_scales=2)
    x = torch.randn(2, 12, 500) # Sequence length 500
    E = model(x, modality="timeseries")
    
    # N should match vision N=196
    assert E.shape == (2, 196, 192), f"Expected (2, 196, 192), got {E.shape}"
    print("USE Time-Series Test: PASS")

def test_hcaa():
    E = torch.randn(2, 196, 192)
    model = HCAA(d_tot=192, num_heads=4)
    out = model(E)
    
    assert out.shape == (2, 196, 192)
    print("HCAA Test: PASS")

def test_asg():
    E = torch.randn(2, 196, 192)
    pathways = [torch.randn(2, 196, 192) for _ in range(3)]
    
    model = ASG(d_tot=192, num_pathways=3, gamma=0.5)
    out, u_k = model(E, pathways)
    
    assert out.shape == (2, 196, 192)
    assert u_k.shape == (2, 3)
    print("ASG Test: PASS")

def test_htspf_forward_backward():
    model = HTSPF(num_classes=10, d_model=64, ldwt_scales=2)
    model.train() # Enable training to trigger backward hooks
    
    x = torch.randn(2, 3, 224, 224)
    targets = torch.tensor([3, 7])
    
    logits, u_k = model(x, modality="vision")
    assert logits.shape == (2, 10)
    
    loss = htspf_loss(logits, targets, u_k, lambda_sparsity=0.1)
    loss.backward()
    
    print("HTSPF Full Forward/Backward Test: PASS")

if __name__ == "__main__":
    test_use_vision()
    test_use_timeseries()
    test_hcaa()
    test_asg()
    test_htspf_forward_backward()
    print("All tests passed successfully.")
