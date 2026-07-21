import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.dataset import prepare_tensor
from src.model import build_model

def train_and_eval(train_x, train_y, test_x, test_y, bands, mean, std, name, device,
                   epochs=12, batch_size=16, lr=1e-3, size=128):
    """Train the customized ResNet18 model and generate test predictions.
    
    Args:
        train_x (torch.Tensor): Raw train images.
        train_y (torch.Tensor): Train labels.
        test_x (torch.Tensor): Raw test images.
        test_y (torch.Tensor): Test labels.
        bands (list[int]): Band indices to keep.
        mean (torch.Tensor): Normalization mean.
        std (torch.Tensor): Normalization standard deviation.
        name (str): Band set name.
        device (torch.device | str): Device to train on.
        epochs (int): Number of training epochs.
        batch_size (int): Dataloader batch size.
        lr (float): Learning rate.
        size (int): Target spatial resolution.
        
    Returns:
        np.ndarray: Predicted probabilities on test set.
    """
    print(f"training {name} ...")
    
    # Prepare datasets
    train_in = prepare_tensor(train_x, bands, mean, std, size=size)
    test_in = prepare_tensor(test_x, bands, mean, std, size=size)
    
    train_dl = DataLoader(TensorDataset(train_in, train_y), batch_size=batch_size, shuffle=True)
    
    model = build_model(bands, device=device)
    
    # Setup loss weighting for class imbalance
    n_pos = train_y.sum()
    pos_weight = (len(train_y) - n_pos) / max(n_pos, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        for m in model.modules():            # freeze BatchNorm running stats
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            
            # Apply spatial augmentations (random horizontal flip, random rot90)
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[3])
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=[2, 3])
            
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradients
            optimizer.step()
            total += loss.item() * len(y)
            
    print(f"  final loss {total / len(train_y):.4f}")

    # Inference
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(test_in.to(device)).squeeze(1)).cpu().numpy()
        
    return probs
