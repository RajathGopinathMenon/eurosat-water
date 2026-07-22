# EuroSAT100: Sentinel-2 Water Classification & Band Ablation Study

This repository implements a binary land-cover classification model (Water vs. No Water) using the EuroSAT100 dataset. It contains a band ablation study comparing performance across different Sentinel-2 band sets to verify the importance of spectral bands beyond visible RGB.

## Project Structure

The project has been structured into modular components:

```
EUROSET/
│
├── README.md             # Project documentation
├── requirements.txt      # Python package dependencies
├── .gitignore            # Git exclusion rules
├── main.py               # Main pipeline execution script
├── explore.py            # Dataset exploration & visualization utility
│
└── src/                  # Core modules
    ├── __init__.py       # Package initializer
    ├── dataset.py        # EuroSAT100 loading and preparation utilities
    ├── model.py          # Pretrained ResNet-18 stem customization
    ├── train.py          # Training and validation pipelines
    └── utils.py          # Performance metrics and visualization helpers
```

## The Ablation Study

Water reflects green light but absorbs almost all near-infrared (NIR) and shortwave-infrared (SWIR) radiation. This study evaluates the effect of input channels by training three ResNet18 models under identical training conditions but with different spectral band inputs:

1. **RGB Only (3 channels)**: Red, Green, and Blue bands (B04, B03, B02).
2. **Water Bands Only (4 channels)**: Green, NIR, SWIR1, and SWIR2 bands (B03, B08, B11, B12).
3. **All Bands (13 channels)**: Complete Sentinel-2 spectral stack (B01-B12).

### Hypothesis
Models trained with NIR and SWIR bands (Water Bands and All Bands) will outperform the RGB-only model, as water separates most cleanly from land in these spectral ranges.

---

## Installation

Ensure you have Python 3.8+ installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Ablation Study

Run the main pipeline to load the EuroSAT100 dataset, train the models on each band set, print metrics, and save the comparative results to `band_comparison.png`:

```bash
python main.py
```

### Dataset Exploration

Run the exploration script to print dataset image with RGB and Infrared bands.

```bash
python explore.py
```
