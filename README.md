# Biomarker Prediction for Cancer Immunotherapy using Classical Neural Network

## 🔬 Project Overview
This project uses a classical neural network to predict gene mutations involved in CTLA-4 pathway regulation from high-dimensional RNA-sequencing data. It aims to identify the most relevant genes for cancer immunotherapy through gradient-based feature attribution.

## 🧬 Problem Statement

Given expression levels of **2048 genes** from RNA sequencing (`RNA.csv`), the model predicts the presence of mutations across **13 target genes** (`MUTATION.csv`) associated with the **CTLA-4 signaling pathway**, a key regulator in cancer immunotherapy.

In addition to mutation prediction, the model identifies the **top 20 most influential input genes** for each mutation target, helping to highlight potential biomarker candidates and improve interpretability.

## 🧠 Model Architecture
A feedforward neural network is implemented using PyTorch:
- Input Layer: 2048 features
- Two hidden layers (hyperparameters tuned with Optuna)
- Output Layer: 13 mutation targets
- Activation: ReLU
- Output: Sigmoid (for probability prediction)

## ⚙️ Features
- 10-fold cross-validation for robust evaluation
- Hyperparameter optimization with Optuna
- Gradient-based gene importance extraction
- Held-out test set evaluation (metrics: F1 Score, Precision, Recall, AUC-ROC)
- Custom sample prediction via `run.py`
- Gene importance visualization and Excel-style export

## 📁 File Structure
```
├── data/
│   ├── RNA.csv                # Gene expression data
│   └── MUTATION.csv          # Mutation status for 13 genes + folds
├── results/                  # Evaluation output and plots
├── main.py                   # Main entry point
├── evaluator.py              # Cross-validation, training, and evaluation
├── model.py                  # GeneSelectorNN architecture
├── utils.py                  # Visualization, feature attribution, helpers
├── run.py                    # Load trained model and predict on a custom sample
```
## 📂 Dataset

Due to GitHub's file size restrictions, the dataset is hosted externally.

🔗 **[Click here to download `data.zip` (73.7 MB)](https://drive.google.com/uc?id=1h5qnC2BKpLfNmezEiG_nhr6ckvtmOlgO&export=download)**

### Contents:
- `RNA.csv`: Gene expression matrix (2048 features)
- `MUTATION.csv`: Mutation labels for 13 genes + cross-validation folds

### 📦 Setup:
After downloading `data.zip`, extract it into the root of the project.

It should create:
project-root/
├── data/
│   ├── RNA.csv
│   └── MUTATION.csv

> 🔒 Add `data/` to `.gitignore` if you wish to avoid accidentally committing large files.

---

## 📊 Outputs
- Top 20 genes by importance for each sample
- Gene importance scores (CSV and bar chart)
- Per-class and macro/micro metrics
- Neural network visualization for each trial
- Excel-style image table for top genes of custom input

## 🔍 Getting Started
Run the training pipeline:
```bash
python main.py
```
After training, make a prediction on a custom sample:
```bash
python run.py
```

## 🧪 Requirements
- Python 3.8+
- PyTorch
- pandas, numpy, seaborn, matplotlib
- scikit-learn
- optuna

## 🛡️ Safety
To suppress PyTorch's future warnings when loading models:
```python
best_params = torch.load(hyperparams_path, weights_only=True)
model.load_state_dict(torch.load(model_path, weights_only=True))
```

## 📚 Reference
This implementation is inspired by the research work:
Biomarker discovery with quantum neural networks: a case-study in CTLA4-activation pathways by Phuong‑Nam Nguyen
🔗 **[Paper Link](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-024-05755-0)**
🔗 **[Repository](https://github.com/namnguyen0510/Biomarker-Discovery-with-Quantum-Neural-Networks)**

This repo presents a classical neural network version of the approach described in the paper.

> No code from the original implementation was used — this is an independent reimplementation based on the concept presented in the paper.

## 👤 Authors
**[Ananth B](https://github.com/1n1nth)**
**[Alen K Aji](https://github.com/AlenKAji)**
**[George Jacob](https://github.com/george6007)**
---
Feel free to raise issues or contribute improvements!

