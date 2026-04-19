---

## Pipeline Overview

### Step 1: Data Preparation
- Merge cytokine MFI values with cell type proportions and demographics
- One row per subject, one column per feature
- Log1p transform on cytokines to reduce right skew
- Median imputation for missing cell type values

### Step 2: EDA + Global PCA
- Scree plot, variance explained, PC loadings
- PC1 = Global Inflammation (57.8%), PC2 = Th17/Regulatory (13.9%)
- Cytokine expression by sex with Welch's t-test

### Step 3: PCA + KNN Classification
- Separate PCAs per modality (cytokines and cell types)
- Grid search over PCA components (2–10) and K (3–7)
- Stratified 5-fold cross-validation
- Soft clustering (Fuzzy C-Means) comparison

### Step 4–7: XGBoost + Multiple Models (In Progress)
- XGBoost, Random Forest, Logistic Regression, SVM, Ensemble
- Feature importance analysis
- Logistic Regression coefficients for interpretability
- Sex-stratified analysis

---

## Key Results

| Target | Features | Method | Accuracy | F1 |
|--------|----------|--------|----------|----|
| Sex | Cytokines only | KNN | 56.8% | 0.591 |
| Sex | Cytokines only | XGBoost (CV) | ~58% | ~0.58 |
| Sex | Cytokines + Cell Types | KNN | 66.4% | 0.661 |
| Age (binary) | Cytokines only | Logistic Regression | 55.5% | 0.536 |
| Age (binary) | Cytokines only | XGBoost | 49.6% | 0.432 |
| Age (binary) | Cytokines only | KNN | 46.8% | 0.419 |
| Age (4-class) | Cytokines only | XGBoost | ~27% | ~0.24 |

**Random baseline**: 50% (binary), 25% (4-class)

**Key finding**: Cell type composition adds ~10% accuracy over cytokines 
alone for sex prediction. Fuzzy clustering confirms no sex-linked cluster 
structure — KNN is better for sex prediction.

### Age Prediction Notes

Binary age classification (Young 18-39 vs Older 40-66) from resting cytokine 
profiles is significantly harder than sex prediction:

- **Logistic Regression** is the only model beating random baseline (55.5% vs 50%)
- **XGBoost** performs at chance (49.6%) — overfits badly at n=125
- **KNN** falls below chance (46.8%) — distance-based approach fails for age signal
- **ROC-AUC = 0.597** (Logistic Regression) confirms a weak but real age signal

Interpretation: Resting cytokine levels do not strongly differentiate age 
in healthy adults without immune challenge. The logistic regression 
coefficients in `results_lr_coefficients_age.csv` identify which cytokines 
push toward older age — the more biologically meaningful finding.

4-class age prediction (~27% vs 25% random) confirms the signal is too weak 
to reliably distinguish between individual decade-based age groups.

---

## Running the Pipelines

### 1. Data Preparation
```bash
cd Data-Emma
python data-prep-fix-subject.py   # generates analysis_merged_subject_level_new.csv
```

### 2. PCA + KNN (Cytokines Only — Sex)
```bash
cd PCA+KNN-Emma
python pca_knn_cytokines.py
```

### 3. PCA + KNN (Cytokines + Cell Types — Sex)
```bash
python pca_knn_cytokine+celltype.py
```

### 4. PCA + KNN (Cytokines Only — Binary Age)
```bash
python AGE-pca_knn_cytokines.py
```

### 5. XGBoost Pipelines
```bash
cd XGBoost/cytokine-only/sex
python xgboost_pipeline_sex.py

cd XGBoost/sex/sex_enhanced
python xgboost_enhanced.py

cd XGBoost/celltype+cytokine/sex
python binary-classification-cytokine+celltypes.py

cd XGBoost/age
python complete_pipeline_age_prediction.py        # 4-class
python xgboost-binary-age-pipeline.py            # binary
```

---

## Exported Data Files

All pipelines export to their respective `Outputs/` folders.

### Core ML Inputs

| File | Shape | Description |
|------|-------|-------------|
| `X_processed.npy` | (125, 18) | Standardized cytokines |
| `X_pca_best.npy` | (125, n_pca) | PCA-reduced features |
| `y_labels.npy` | (125,) | Binary labels (1=Male/Older, 0=Female/Young) |
| `pca_components.npy` | (n_pca, n_features) | PCA projection matrix |
| `X_combined_pca.npy` | (125, 14) | Combined cytokine + cell type PCs |
| `X_cytokines_processed.npy` | (125, 18) | Standardized cytokines (full pipeline) |
| `X_cell_processed.npy` | (125, 59) | Standardized cell types |

### Results Files

| File | Description |
|------|-------------|
| `results_summary.csv` | CV accuracy across all hyperparameter combos |
| `feature_importance.csv` | Variance-weighted PCA loadings per feature |
| `cytokine_sex_comparison.csv` | Welch's t-test results, cytokines by sex |
| `celltype_sex_comparison.csv` | Welch's t-test results, cell types by sex |
| `fcm_vs_knn_comparison.csv` | Fuzzy C-Means vs KNN accuracy comparison |
| `pca_loadings_global.csv` | Global PCA loadings for PC1–PC3 |

---

## Loading Data in Downstream Models

```python
import numpy as np
from pathlib import Path

BASE_DIR = Path("PCA+KNN-Emma/Outputs/cytokines_only")

X = np.load(BASE_DIR / "X_processed.npy")   # standardized cytokines
y = np.load(BASE_DIR / "y_labels.npy")       # binary labels
```

**Important**: `X_processed.npy` is globally standardized.
Re-standardize within each cross-validation fold for any new model.

---

## Critical Notes

- **Fuzzy clustering** confirms no sex-linked cluster structure in cytokine 
  space — the KNN signal requires a supervised approach to detect
- **57% on cytokines alone** is informative, not a failure — it means resting 
  cytokine secretion doesn't strongly differentiate sex at baseline
- **66% with cell types** is the key finding — immune cell composition carries 
  sex signal that cytokine secretion alone misses
- **Age prediction** is harder than sex — 4-class accuracy near random baseline 
  suggests cytokines alone do not strongly predict age in healthy donors
- **ROC-AUC = 1.0** in some XGBoost outputs is a training-set artifact — 
  always use cross-validated metrics for honest evaluation

---

## Design Philosophy

- No black-box ML dependencies in core PCA+KNN pipeline
- Fully reproducible feature engineering
- Strict separation between preprocessing and modeling
- Explicit export layer for cross-model compatibility
- Biological interpretability prioritized via cytokine importance ranking
- Separate PCA per modality prevents cell type count from dominating variance