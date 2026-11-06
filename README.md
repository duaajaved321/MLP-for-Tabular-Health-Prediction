# MLP for Tabular Health Prediction

An educational binary-classification experiment using the **UCI Early Stage Diabetes Risk Prediction** dataset (UCI dataset ID: 529). The notebook trains a PyTorch Multi-Layer Perceptron (MLP) and compares it with a Random Forest baseline.

## Project Goals

- Explore a tabular health-risk dataset.
- Prepare numerical and categorical features.
- Train a PyTorch MLP for binary classification.
- Monitor training and validation loss.
- Evaluate predictions with a confusion matrix, precision, recall, and F1-score.
- Compare the MLP with a Random Forest baseline.

## Dataset

**Early Stage Diabetes Risk Prediction** — UCI Machine Learning Repository, dataset ID **529**. The notebook retrieves the data with `ucimlrepo`. Features include age, gender, questionnaire-based symptom indicators, and the binary target `class` (Positive/Negative).

Dataset: https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction

## Project Structure

```text
MLP-for-Tabular-Health-Prediction/
├── Milestone_MLP_for_Tabular_Health_Prediction.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Workflow

1. Load the UCI dataset and inspect its shape, types, missing values, and duplicate rows.
2. Remove exact duplicate records and create a stratified 90/10 train/test split.
3. Explore class counts, age distributions, symptom prevalence, and gender/class proportions.
4. Split the training partition into model-fitting and validation subsets.
5. Fit preprocessing only on the model-fitting subset: standardize age, ordinal-encode categorical inputs, and encode the target separately.
6. Train a PyTorch MLP with two hidden layers (32 and 16 units), ReLU activations, Adam, and `BCEWithLogitsLoss`.
7. Monitor training and validation loss.
8. Evaluate on the held-out test set and compare against a Random Forest baseline.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/duaajaved321/MLP-for-Tabular-Health-Prediction.git
cd MLP-for-Tabular-Health-Prediction
```

### 2. Create a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open `Milestone_MLP_for_Tabular_Health_Prediction.ipynb` and run all cells from top to bottom. Dataset retrieval requires an internet connection.

## Evaluation

The notebook reports confusion matrices, precision, recall, F1-score, a side-by-side model comparison, MLP learning curves, and a results summary generated from actual metrics. No performance scores are hardcoded in this README.

## Implementation Notes

- The target column is excluded from model inputs and encoded separately.
- The symptom-prevalence plot excludes the target label.
- The validation subset is created before preprocessing.
- The scaler and categorical encoder are fitted only on the model-fitting subset.
- The test set is reserved for final evaluation.
- MLP and Random Forest use the same fitting and test partitions.

## Limitations

This is a small, questionnaire-based dataset and a single-split experiment. No hyperparameter search, probability calibration, external validation, or clinical safety assessment is performed. Results are educational only and must not be used for diagnosis or clinical decision-making. This project is not a medical device or a clinically validated prediction system.

## Author

**Dua Javed**
