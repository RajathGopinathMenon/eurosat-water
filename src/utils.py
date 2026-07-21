import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

def score_predictions(probs, test_y, name):
    """Compute standard classification metrics from predicted probabilities.
    
    Args:
        probs (np.ndarray): Predicted probabilities.
        test_y (torch.Tensor): True test labels.
        name (str): The name of this model/band set.
        
    Returns:
        dict: Containing accuracy, precision, recall, f1, roc_auc, confusion matrix, etc.
    """
    true = test_y.numpy().astype(int)
    pred = (probs >= 0.5).astype(int)
    return {
        "name": name,
        "probs": probs,
        "accuracy": accuracy_score(true, pred),
        "precision": precision_score(true, pred, zero_division=0),
        "recall": recall_score(true, pred, zero_division=0),
        "f1": f1_score(true, pred, zero_division=0),
        "roc_auc": roc_auc_score(true, probs),
        "cm": confusion_matrix(true, pred, labels=[0, 1]),
    }


def make_comparison_image(results, test_y, path="band_comparison.png"):
    """Create a high-quality visualization comparing model performance.
    
    Creates a plot with:
    - Bar charts for key metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    - ROC curves comparison
    - Confusion matrices for each evaluated band set
    
    Args:
        results (list[dict]): List of result dictionaries from score_predictions.
        test_y (torch.Tensor): True test labels.
        path (str): File path to save the generated image.
    """
    true = test_y.numpy().astype(int)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors = ["#C44E52", "#DD8452", "#4C72B0"]      # Colors corresponding to RGB, Water bands, All bands
    
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    
    # 1. Bar chart comparing metrics
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(len(keys))
    w = 0.26
    for i, r in enumerate(results):
        vals = [r[k] for k in keys]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=r["name"], color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.set_title("EuroSAT100 water classification — effect of input bands", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    # 2. ROC curves
    ax = fig.add_subplot(gs[0, 2])
    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(true, r["probs"])
        ax.plot(fpr, tpr, color=colors[i], label=f"{r['name']} ({r['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    
    # 3. Confusion matrices
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[1, i])
        cm = r["cm"]
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for a in range(2):
            for b in range(2):
                ax.text(b, a, cm[a][b], ha="center", va="center", fontsize=14,
                        color="white" if cm[a][b] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["no water", "water"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["no water", "water"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{r['name']}\nacc {r['accuracy']:.2f}", fontsize=10)
        
    fig.suptitle(f"Test set: {len(test_y)} images ({int(test_y.sum())} water)", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nsaved {path}")
