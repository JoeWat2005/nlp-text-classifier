from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def _ensure_parent_dir(save_path: str) -> None:
    """
    Ensure the directory for save_path exists (e.g. 'models/' in 'models/plot.png').
    If save_path is just a filename with no directory, do nothing.
    """
    parent = Path(save_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, save_path="models/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    _ensure_parent_dir(save_path)
    plt.savefig(save_path)
    plt.close()

    print(f"Confusion matrix saved to {save_path}")


def plot_roc(y_true, y_prob, save_path="models/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    _ensure_parent_dir(save_path)
    plt.savefig(save_path)
    plt.close()

    print(f"ROC curve saved to {save_path}")
