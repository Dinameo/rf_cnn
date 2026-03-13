from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter


def plot_loss(train_losses, val_losses, ax):
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend()
    ax.grid(True)
def plot_accuracy(train_accs, val_accs, ax):
    ax.plot(train_accs, label="Train Accuracy")
    ax.plot(val_accs, label="Val Accuracy")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend()
    ax.grid(True)
def plot_confusion_matrix(y_true, y_pred, ax):
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

from collections import Counter
import matplotlib.pyplot as plt

def plot_distribution_of_dataset(dataset, ax, title="Dataset Distribution"):

    if hasattr(dataset, "indices"):  # nếu là Subset
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = dataset.targets

    counts = Counter(targets)

    x = list(counts.keys())
    y = list(counts.values())

    bars = ax.bar(x, y)

    # hiện số trên mỗi bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom"
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)

def create_visualization_figure(row, col, figsize=(18, 5)):
    fig, axes = plt.subplots(row, col, figsize=figsize)
    return fig, axes