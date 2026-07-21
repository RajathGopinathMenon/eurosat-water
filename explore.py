#!/usr/bin/env python3
"""Show 2 water and 2 non-water images from EuroSAT100."""

import ssl
import numpy as np
import matplotlib.pyplot as plt
from torchgeo.datasets import EuroSAT100

ssl._create_default_https_context = ssl._create_unverified_context  # macOS cert fix

WATER = {"river", "sealake"}
R, G, B = 3, 2, 1          # B04, B03, B02 in EuroSAT's band order

ds = EuroSAT100(root="./data", split="train", download=True)
water_ids = {i for i, c in enumerate(ds.classes) if c.lower() in WATER}


def rgb(sample):
    """True-colour image with a 2-98% contrast stretch."""
    img = sample["image"].numpy()[[R, G, B]].transpose(1, 2, 0)
    lo, hi = np.percentile(img, [2, 98])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


# Find the first 2 water tiles and first 2 non-water tiles.
water, other = [], []
for i in range(len(ds)):
    s = ds[i]
    (water if int(s["label"]) in water_ids else other).append(s)
    if len(water) >= 2 and len(other) >= 2:
        break

picks = [(water[0], "WATER"), (water[1], "WATER"),
         (other[0], "not water"), (other[1], "not water")]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, (s, tag) in zip(axes, picks):
    ax.imshow(rgb(s))
    ax.set_title(f"{tag}\n{ds.classes[int(s['label'])]}", fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.savefig("water_examples.png", dpi=150, bbox_inches="tight")
print("saved water_examples.png")