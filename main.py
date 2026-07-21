"""
EuroSAT100: is there water in this image? (Water = River or SeaLake)

Band ablation study
-------------------
Water is defined physically by how it reflects light: it is visible in the
green band, but absorbs almost all near-infrared (NIR) and shortwave-infrared
(SWIR) light. The visible (RGB) bands alone therefore carry only part of the
signal. This script tests that idea directly by training the same ResNet18 on
three growing sets of Sentinel-2 bands:

    1. RGB only (3 bands) -- B04, B03, B02
    2. Water bands (4 bands) -- green + NIR + SWIR1 + SWIR2
                                 (the bands where water is most distinct)
    3. All bands (13 bands) -- the full Sentinel-2 stack

Hypothesis: RGB should do worst, and adding the NIR/SWIR bands should help,
because that is where water separates most cleanly from land.

Everything else (model, training, augmentation) is held fixed so the only
thing changing is the input bands. Results are written to one image:
band_comparison.png

Run: python main.py
"""

import numpy as np
import torch
from src.dataset import load_split, compute_normalization_stats
from src.train import train_and_eval
from src.utils import score_predictions, make_comparison_image

EPOCHS = 12 # loss converges by ~12; more just overfits 60 images
BATCH = 16
LR = 1e-3
SIZE = 128

# EuroSAT's 13-band order (index -> band):
# 0 B01 1 B02(blue) 2 B03(green) 3 B04(red) 4 B05 5 B06 6 B07
# 7 B08(NIR) 8 B08A 9 B09 10 B10 11 B11(SWIR1) 12 B12(SWIR2)
BAND_SETS = {
    "RGB (3)": [3, 2, 1], # red, green, blue
    "Water bands (4)": [2, 7, 11, 12], # green, NIR, SWIR1, SWIR2
    "All bands (13)": list(range(13)),
}

def main():
    torch.manual_seed(42)
    np.random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 1. Load splits
    train_x, train_y = load_split("train")
    test_x, test_y = load_split("test")
    print(f"train {len(train_y)} ({int(train_y.sum())} water) | "
          f"test {len(test_y)} ({int(test_y.sum())} water)\n")

    # 2. Compute training split statistics for normalization
    mean, std = compute_normalization_stats(train_x)

    # 3. Train models and generate predictions
    results = []
    for name, bands in BAND_SETS.items():
        probs = train_and_eval(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            bands=bands,
            mean=mean,
            std=std,
            name=name,
            device=device,
            epochs=EPOCHS,
            batch_size=BATCH,
            lr=LR,
            size=SIZE
        )
        # Evaluate predictions
        res = score_predictions(probs, test_y, name)
        results.append(res)

    # 4. Display comparison table
    print("\nband set             accuracy  precision  recall   f1     AUC")
    for r in results:
        print(f"{r['name']:<20} {r['accuracy']:.3f}     {r['precision']:.3f}"
              f"      {r['recall']:.3f}   {r['f1']:.3f}  {r['roc_auc']:.3f}")

    # 5. Generate and save summary figure
    make_comparison_image(results, test_y, "band_comparison.png")

if __name__ == "__main__":
    main()