"""
src/data.py — Unified Dataloader Factory for HTSPF Phase 3 Benchmarks.

Supports:
  - Vision: CIFAR-100
  - Time-Series: UCR Archive datasets (FordA, EthanolConcentration)

All dataloaders return (X, y) with normalized inputs and consistent N sequence length.
"""

import os
import random
import numpy as np
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torchvision
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from datasets import load_dataset


# ──────────────────────────────────────────────
# Reproducibility Utilities
# ──────────────────────────────────────────────

def set_seed(seed: int):
    """Fixes all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Vision: CIFAR-100
# ──────────────────────────────────────────────

class HFDatasetWrapper(Dataset):
    def __init__(self, hf_ds, transform=None):
        self.hf_ds = hf_ds
        self.transform = transform
    def __len__(self):
        return len(self.hf_ds)
    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img = item['img']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        label = item['fine_label']
        if self.transform:
            img = self.transform(img)
        return img, label

def get_cifar100_dataloaders(root="uoft-cs/cifar100", batch_size=64,
                              val_split=0.1, num_workers=4, seed=0):
    """
    Returns (train_loader, val_loader, test_loader) for CIFAR-100 using HuggingFace datasets.
    Images are 32x32. With patch_size=4, N = 8*8 = 64 patches.
    """
    normalize = T.Normalize(mean=[0.5071, 0.4867, 0.4408],
                            std=[0.2675, 0.2565, 0.2761])

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    try:
        hf_dataset = load_dataset(root)
        train_full = HFDatasetWrapper(hf_dataset['train'], transform=train_transform)
        test_set = HFDatasetWrapper(hf_dataset['test'], transform=test_transform)

    except Exception as e:
        print(f"  [WARNING] CIFAR-100 HF download failed: {e}")
        print(f"  [WARNING] Using synthetic data for pipeline validation.")
        n_train, n_test = 5000, 1000
        X_train = torch.randn(n_train, 3, 32, 32)
        y_train = torch.randint(0, 100, (n_train,))
        X_test  = torch.randn(n_test, 3, 32, 32)
        y_test  = torch.randint(0, 100, (n_test,))
        train_full = TensorDataset(X_train, y_train)
        test_set   = TensorDataset(X_test, y_test)

    n_val = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

# ──────────────────────────────────────────────
# Vision: CIFAR-10 (Kaggle)
# ──────────────────────────────────────────────

class KaggleCIFAR10Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_dir = os.path.join(root_dir, split)
        
        if split == "train":
            df = pd.read_csv(os.path.join(root_dir, "trainLabels.csv"))
            self.classes = sorted(df['label'].unique())
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.data = [(os.path.join(self.img_dir, f"{row['id']}.png"), self.class_to_idx[row['label']]) 
                         for _, row in df.iterrows()]
        else:
            # test directory might just have numbered images
            img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]
            self.data = [(os.path.join(self.img_dir, f), -1) for f in img_files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_cifar10_dataloaders(root="data/cifar10", batch_size=64,
                            val_split=0.1, num_workers=4, seed=0):
    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2470, 0.2435, 0.2616])
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_transform = T.Compose([T.ToTensor(), normalize])

    train_full = KaggleCIFAR10Dataset(root, split="train", transform=train_transform)
    test_set = KaggleCIFAR10Dataset(root, split="test", transform=test_transform)

    n_val = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# Time-Series: UCR Archive
# ──────────────────────────────────────────────

def _download_ucr_dataset(name: str, root: str):
    """Downloads and caches a UCR dataset using sktime or tslearn."""
    try:
        from sktime.datasets import load_UCR_UEA_dataset
        X_train, y_train = load_UCR_UEA_dataset(name, split="train", return_X_y=True)
        X_test, y_test   = load_UCR_UEA_dataset(name, split="test",  return_X_y=True)
        return X_train, y_train, X_test, y_test
    except ImportError:
        raise ImportError(
            "Please install sktime: pip install sktime\n"
            "Required for UCR dataset loading."
        )


def _sktime_to_tensor(X_sktime, y, n_seq_len: int):
    """
    Convert sktime DataFrame (n_samples, n_channels) of Series to
    a tensor of shape (n_samples, n_channels, n_seq_len).
    Normalizes each sample to zero mean, unit variance.
    """
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = torch.tensor(le.fit_transform(y), dtype=torch.long)

    n_samples = len(X_sktime)
    n_channels = X_sktime.shape[1]

    all_series = []
    for i in range(n_samples):
        channels = []
        for c in range(n_channels):
            series = X_sktime.iloc[i, c].to_numpy().astype(np.float32)
            # Interpolate or truncate to fixed n_seq_len
            if len(series) != n_seq_len:
                series = np.interp(
                    np.linspace(0, len(series)-1, n_seq_len),
                    np.arange(len(series)),
                    series
                )
            channels.append(series)
        all_series.append(np.stack(channels, axis=0))

    X_tensor = torch.tensor(np.stack(all_series, axis=0), dtype=torch.float32)

    # Normalize: z-score per sample per channel
    mean = X_tensor.mean(dim=-1, keepdim=True)
    std  = X_tensor.std(dim=-1, keepdim=True) + 1e-8
    X_tensor = (X_tensor - mean) / std

    return X_tensor, y_enc


def get_ucr_dataloaders(name: str, root: str = "data/ucr",
                        batch_size: int = 64, val_split: float = 0.1,
                        n_seq_len: int = 196, num_workers: int = 0, seed: int = 0):
    """
    Returns (train_loader, val_loader, test_loader) for a UCR dataset.

    n_seq_len is set to match the Vision USE sequence length N (default=196 for 32x32/patch4).
    """
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = _download_ucr_dataset(name, root)

    X_train, y_train = _sktime_to_tensor(X_train_raw, y_train_raw, n_seq_len)
    X_test,  y_test  = _sktime_to_tensor(X_test_raw,  y_test_raw,  n_seq_len)

    train_full = TensorDataset(X_train, y_train)
    test_set   = TensorDataset(X_test, y_test)

    n_val   = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, n_val],
                                      generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# Unified Factory
# ──────────────────────────────────────────────

def get_dataloaders(dataset_name: str, cfg: dict, seed: int = 0):
    """
    Unified factory function called by the training script.

    Args:
        dataset_name: one of 'cifar100', 'ford_a', 'ethanol_concentration'
        cfg: the loaded experiment.yaml config dict
        seed: random seed for reproducibility

    Returns:
        (train_loader, val_loader, test_loader, modality)
    """
    if dataset_name == "cifar100":
        dcfg = cfg["datasets"]["cifar100"]
        loaders = get_cifar100_dataloaders(
            root=dcfg["root"],
            batch_size=cfg["training"]["batch_size"],
            val_split=dcfg["val_split"],
            num_workers=cfg["training"]["num_workers"],
            seed=seed,
        )
        return *loaders, "vision"

    elif dataset_name == "cifar10":
        dcfg = cfg["datasets"]["cifar10"]
        loaders = get_cifar10_dataloaders(
            root=dcfg["root"],
            batch_size=cfg["training"]["batch_size"],
            val_split=dcfg["val_split"],
            num_workers=cfg["training"]["num_workers"],
            seed=seed,
        )
        return *loaders, "vision"

    elif dataset_name == "ford_a":
        dcfg = cfg["datasets"]["ford_a"]
        loaders = get_ucr_dataloaders(
            name=dcfg["name"],
            root=dcfg["root"],
            batch_size=cfg["training"]["batch_size"],
            seed=seed,
        )
        return *loaders, "timeseries"

    elif dataset_name == "ethanol_concentration":
        dcfg = cfg["datasets"]["ethanol_concentration"]
        loaders = get_ucr_dataloaders(
            name=dcfg["name"],
            root=dcfg["root"],
            batch_size=cfg["training"]["batch_size"],
            seed=seed,
        )
        return *loaders, "timeseries"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
