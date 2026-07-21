import torch
import torch.nn as nn
from torchgeo.models import ResNet18_Weights, resnet18

def build_model(bands, device=None):
    """Create a ResNet-18 model pretrained on Sentinel-2 data.
    
    If the selected band subset is not the full 13-channel Sentinel-2 stack,
    we extract only the corresponding pretrained filters from the first conv layer
    and rescale them so the input signal magnitude remains consistent.
    
    Args:
        bands (list[int]): Indices of selected bands.
        device (torch.device | str): Device to load the model onto.
        
    Returns:
        nn.Module: Configured PyTorch model.
    """
    model = resnet18(weights=ResNet18_Weights.SENTINEL2_ALL_MOCO)
    n = len(bands)
    
    if n != 13:
        # Extract matching pretrained filters and scale weights
        w = model.conv1.weight.data[:, bands] * (13.0 / n)
        model.conv1 = nn.Conv2d(n, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = w
        
    # Replace last layer for binary classification (outputs 1 logit)
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    if device is not None:
        model = model.to(device)
        
    return model
