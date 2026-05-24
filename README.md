# Network Intrusion Detection — CICIDS2017

> Multi-class classification on the Canadian Institute for Cybersecurity Intrusion Detection dataset (~2.2 GB)

---

## 📋 Assignment
Machine Learning Classification on a GB-sized dataset using Python and scikit-learn.

## 📦 Dataset
| Field | Details |
|-------|---------|
| Name | CICIDS2017 |
| Size | ~2.2 GB |
| Source | [Kaggle — dhoogla/cicids2017](https://www.kaggle.com/datasets/dhoogla/cicids2017) |
| Task | Multi-class: BENIGN vs DoS, DDoS, PortScan, Brute Force, etc. |
| Features | 80 network flow features |

## 🧠 Model
- **Algorithm**: Random Forest Classifier (`n_estimators=100`, `max_depth=20`)
- **Library**: scikit-learn 1.3+
- **Train/Test Split**: 80% / 20% stratified

## 📊 Results

![Class Distribution](results/1_class_distribution.png)
![Confusion Matrix](results/2_confusion_matrix.png)
![Feature Importance](results/3_feature_importance.png)
![Per-Class Metrics](results/4_per_class_metrics.png)
![Accuracy Summary](results/5_accuracy_summary.png)

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

You also need a Kaggle API key:
1. Go to [kaggle.com](https://kaggle.com) → Account → API → Create New Token
2. Place the downloaded `kaggle.json` in `~/.kaggle/`

### Run

```bash
# Step 1 — Download dataset (~2.2 GB)
python cicids2017_classification.py --download

# Step 2 — Run full pipeline (load → preprocess → train → evaluate → plot)
python cicids2017_classification.py
```

Results (5 graphs) are saved to the `results/` folder.

## 📁 Repository Structure
```
├── cicids2017_classification.py   ← main script
├── requirements.txt
├── README.md
├── .gitignore
└── results/
    ├── 1_class_distribution.png
    ├── 2_confusion_matrix.png
    ├── 3_feature_importance.png
    ├── 4_per_class_metrics.png
    └── 5_accuracy_summary.png
```

> **Note**: The `data/` folder is gitignored (too large for GitHub). Use the `--download` flag or download manually from Kaggle.

## 📜 License
MIT — open source, free to use and modify.
