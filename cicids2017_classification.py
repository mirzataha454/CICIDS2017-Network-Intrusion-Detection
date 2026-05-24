"""
=============================================================
 Network Intrusion Detection using CICIDS2017 Dataset
 Classification Model: Random Forest + XGBoost Comparison
 Author: [Your Name] | Roll No: [Your Roll Number]
=============================================================

Dataset: CICIDS2017 (~2.2 GB)
Kaggle URL: https://www.kaggle.com/datasets/dhoogla/cicids2017
Description: Network traffic classification into BENIGN vs Attack types.

Steps:
  1. Download Dataset
  2. Load & Preprocess
  3. Train Classification Model
  4. Evaluate & Plot Results
  5. Save Graphs
"""

# ─── 1. IMPORTS ────────────────────────────────────────────────────────────────
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ─── GLOBAL STYLE ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})
PALETTE = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD",
           "#BA7517", "#E24B4A", "#639922", "#D4537E"]

os.makedirs("results", exist_ok=True)

# ─── 2. DOWNLOAD DATASET ───────────────────────────────────────────────────────
def download_dataset():
    """
    Download CICIDS2017 from Kaggle using the Kaggle API.
    Run this ONCE before running the rest of the script.

    Prerequisites:
        pip install kaggle
        Place kaggle.json in ~/.kaggle/  (from Kaggle → Account → API token)
    """
    print("Downloading CICIDS2017 dataset from Kaggle (~2.2 GB)...")
    os.system(
        "kaggle datasets download -d dhoogla/cicids2017 "
        "--unzip -p ./data/"
    )
    print("Download complete. Files saved to ./data/")


# ─── 3. LOAD DATASET ───────────────────────────────────────────────────────────
def load_dataset(data_dir="./data/"):
    """
    Load all CSV files from the CICIDS2017 directory using chunked reading
    to handle GB-sized files without running out of memory.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Please run download_dataset() first."
        )

    print(f"Found {len(csv_files)} CSV file(s). Loading with chunked reader...")
    chunks = []
    CHUNK_SIZE = 100_000

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        print(f"  Reading: {filename}")
        for chunk in pd.read_csv(
            filepath,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
        ):
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal records loaded: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    return df


# ─── 4. PREPROCESS ─────────────────────────────────────────────────────────────
def preprocess(df):
    """
    Clean column names, handle missing/infinite values, encode labels,
    and scale features.
    """
    # --- Clean column names ---
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # --- Identify label column ---
    label_col = None
    for candidate in ["Label", "label", " Label", "Class"]:
        if candidate.strip() in df.columns:
            label_col = candidate.strip()
            break
    if label_col is None:
        raise ValueError("Could not find label column. "
                         f"Columns: {list(df.columns)}")

    print(f"\nLabel column: '{label_col}'")
    print("Class distribution (before encoding):")
    print(df[label_col].value_counts())

    # --- Drop non-numeric / irrelevant columns ---
    df.drop(
        columns=[c for c in ["Flow_ID", "Source_IP", "Destination_IP",
                              "Timestamp", "Source_Port", "Destination_Port"]
                 if c in df.columns],
        inplace=True,
        errors="ignore",
    )

    # --- Replace inf and NaN ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"\nRecords after cleaning: {len(df):,}")

    # --- Encode labels ---
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    class_names = le.classes_

    # --- Feature matrix ---
    X = df.drop(columns=[label_col])

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # --- Optimise dtypes to save RAM ---
    for col in X.select_dtypes("float64").columns:
        X[col] = X[col].astype("float32")
    for col in X.select_dtypes("int64").columns:
        X[col] = X[col].astype("int32")

    # --- Scale ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nFeature matrix shape: {X_scaled.shape}")
    print(f"Classes: {list(class_names)}")
    return X_scaled, y, class_names, X.columns.tolist()


# ─── 5. TRAIN MODEL ────────────────────────────────────────────────────────────
def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,          # use all CPU cores
        random_state=42,
        verbose=1,
    )
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


# ─── 6. EVALUATE ───────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, class_names):
    """Return predictions and print metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy : {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    return y_pred


# ─── 7. PLOTS ──────────────────────────────────────────────────────────────────

def plot_class_distribution(y, class_names):
    """Bar chart of class distribution in the full dataset."""
    counts = pd.Series(y).value_counts().sort_index()
    labels = [class_names[i] for i in counts.index]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, counts.values, color=PALETTE[:len(labels)],
                  edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{val:,}", ha="center", va="bottom",
                fontsize=9, fontweight="500")

    ax.set_title("Class Distribution in CICIDS2017 Dataset")
    ax.set_xlabel("Traffic Class")
    ax.set_ylabel("Number of Records")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/1_class_distribution.png")
    plt.show()
    print("  Saved: results/1_class_distribution.png")


def plot_confusion_matrix(y_test, y_pred, class_names):
    """Normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)),
                                    max(6, len(class_names) - 1)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center",
                    va="center", fontsize=8, color=color)

    ax.set_title("Normalised Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig("results/2_confusion_matrix.png")
    plt.show()
    print("  Saved: results/2_confusion_matrix.png")


def plot_feature_importance(model, feature_names, top_n=20):
    """Horizontal bar chart of top feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [PALETTE[i % len(PALETTE)] for i in range(top_n)]
    ax.barh(range(top_n), top_vals[::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/3_feature_importance.png")
    plt.show()
    print("  Saved: results/3_feature_importance.png")


def plot_per_class_metrics(y_test, y_pred, class_names):
    """Grouped bar chart: Precision, Recall, F1 per class."""
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred)

    x = np.arange(len(class_names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.4), 6))
    ax.bar(x - width, p,  width, label="Precision", color="#378ADD")
    ax.bar(x,         r,  width, label="Recall",    color="#1D9E75")
    ax.bar(x + width, f1, width, label="F1-Score",  color="#D85A30")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall & F1-Score")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/4_per_class_metrics.png")
    plt.show()
    print("  Saved: results/4_per_class_metrics.png")


def plot_accuracy_summary(y_test, y_pred):
    """Simple accuracy summary card chart."""
    acc = accuracy_score(y_test, y_pred)
    err = 1 - acc

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Correct", "Incorrect"], [acc, err],
           color=["#1D9E75", "#E24B4A"], width=0.4, edgecolor="white")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Proportion")
    ax.set_title("Model Accuracy Summary")

    for i, val in enumerate([acc, err]):
        ax.text(i, val + 0.02, f"{val*100:.2f}%",
                ha="center", fontweight="500", fontsize=12)

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/5_accuracy_summary.png")
    plt.show()
    print("  Saved: results/5_accuracy_summary.png")


# ─── 8. README GENERATOR ───────────────────────────────────────────────────────
def save_readme(acc, class_names):
    readme = f"""# Network Intrusion Detection — CICIDS2017

## 📋 Assignment
Machine Learning Classification on a GB-sized dataset.

## 📦 Dataset
- **Name**: CICIDS2017 (Canadian Institute for Cybersecurity)
- **Size**: ~2.2 GB
- **Source**: [Kaggle](https://www.kaggle.com/datasets/dhoogla/cicids2017)
- **Task**: Multi-class classification — BENIGN vs various network attacks

## 🧠 Model
- **Algorithm**: Random Forest Classifier
- **Library**: scikit-learn
- **Train/Test Split**: 80% / 20%

## 📊 Results
| Metric | Score |
|--------|-------|
| Accuracy | {acc*100:.2f}% |
| Classes | {len(class_names)} |

## 📈 Graphs
| Graph | Description |
|-------|-------------|
| `results/1_class_distribution.png` | Distribution of traffic classes |
| `results/2_confusion_matrix.png` | Normalised confusion matrix |
| `results/3_feature_importance.png` | Top 20 most important features |
| `results/4_per_class_metrics.png` | Precision, Recall, F1 per class |
| `results/5_accuracy_summary.png` | Overall accuracy bar |

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (requires Kaggle API key)
python cicids2017_classification.py --download

# 3. Run full pipeline
python cicids2017_classification.py
```

## 📁 Repo Structure
```
├── data/                     ← dataset CSVs (gitignored)
├── results/                  ← saved plots
├── cicids2017_classification.py
├── requirements.txt
└── README.md
```

## ⚙️ Requirements
See `requirements.txt`
"""
    with open("README.md", "w") as f:
        f.write(readme)
    print("  Saved: README.md")


def save_requirements():
    reqs = """numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
kaggle>=1.6
"""
    with open("requirements.txt", "w") as f:
        f.write(reqs)
    print("  Saved: requirements.txt")


# ─── 9. GITIGNORE ──────────────────────────────────────────────────────────────
def save_gitignore():
    content = """# Data (too large for GitHub)
data/
*.csv

# Python
__pycache__/
*.pyc
.env

# Jupyter
.ipynb_checkpoints/
"""
    with open(".gitignore", "w") as f:
        f.write(content)
    print("  Saved: .gitignore")


# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def main(download=False):
    print("=" * 60)
    print(" CICIDS2017 Network Intrusion Detection — Full Pipeline")
    print("=" * 60)

    if download:
        download_dataset()

    # Step 3 — Load
    df = load_dataset("./data/")

    # Step 2 — Preprocess
    X, y, class_names, feature_names = preprocess(df)

    # Plot class distribution (before split)
    print("\n[Plot 1] Class distribution...")
    plot_class_distribution(y, class_names)

    # Step 3 — Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # Step 4 — Train
    model = train_model(X_train, y_train)

    # Step 5 — Evaluate
    y_pred = evaluate(model, X_test, y_test, class_names)
    acc = accuracy_score(y_test, y_pred)

    # Step 5 — Graphs
    print("\n[Plot 2] Confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, class_names)

    print("[Plot 3] Feature importance...")
    plot_feature_importance(model, feature_names)

    print("[Plot 4] Per-class metrics...")
    plot_per_class_metrics(y_test, y_pred, class_names)

    print("[Plot 5] Accuracy summary...")
    plot_accuracy_summary(y_test, y_pred)

    # Step 6 — Repo files
    print("\nGenerating repo files...")
    save_readme(acc, class_names)
    save_requirements()
    save_gitignore()

    print("\n" + "=" * 60)
    print(f" DONE — Final Accuracy: {acc*100:.2f}%")
    print(" All graphs saved to: results/")
    print(" Next: git init → git add . → git commit → git push")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    download_flag = "--download" in sys.argv
    main(download=download_flag)
