import torch
import torch.nn.functional as F
from torchgeo.datasets import EuroSAT100

WATER_CLASSES = {"river", "sealake"}

def load_split(split, root_dir="./data", water_classes=None):
    """Load a EuroSAT100 split fully into memory.
    
    Args:
        split (str): One of 'train', 'val', or 'test'.
        root_dir (str): Root directory where dataset is downloaded/stored.
        water_classes (set): Class names that count as water.
        
    Returns:
        tuple: (images_tensor, labels_tensor)
    """
    if water_classes is None:
        water_classes = WATER_CLASSES
        
    ds = EuroSAT100(root=root_dir, split=split, download=True)
    x = torch.stack([ds[i]["image"].float() for i in range(len(ds))])
    
    water_ids = {i for i, c in enumerate(ds.classes) if c.lower() in water_classes}
    y = torch.tensor([float(int(ds[i]["label"]) in water_ids) for i in range(len(ds))])
    
    return x, y


def compute_normalization_stats(train_x):
    """Compute per-channel mean and standard deviation from training set.
    
    Args:
        train_x (torch.Tensor): Training image tensor of shape (N, C, H, W).
        
    Returns:
        tuple: (mean, std) formatted as (C, 1, 1) tensors.
    """
    mean = train_x.mean(dim=(0, 2, 3)).view(-1, 1, 1)
    std = train_x.std(dim=(0, 2, 3)).clamp_min(1e-6).view(-1, 1, 1)
    return mean, std


def prepare_tensor(x, bands, mean, std, size=128):
    """Normalize, keep only the chosen bands, and resize for ResNet.
    
    Args:
        x (torch.Tensor): Input image tensor of shape (N, C, H, W).
        bands (list[int]): Band indices to select.
        mean (torch.Tensor): Normalization mean of shape (C, 1, 1).
        std (torch.Tensor): Normalization standard deviation of shape (C, 1, 1).
        size (int): Target spatial size to resize to.
        
    Returns:
        torch.Tensor: Processed image tensor.
    """
    xn = (x - mean) / std
    xn = xn[:, bands]
    return F.interpolate(xn, size=size, mode="bilinear", align_corners=False)
