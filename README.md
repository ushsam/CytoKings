## CytoKings: Predicting Biological Sex from Cytokine Expression Profiles

### Project Overview

Machine learning pipeline to predict biological sex from resting cytokine 
expression profiles in 125 healthy adult donors (63M/62F). 
18 cytokine features measured via Luminex multiplex bead assay (MFI values).
---

## Repository Structure

```
CytoKings/
├── Data/
│   ├── dataprep.py                          ← data merging + preprocessing
│   ├── EDA+Batch_Effects.py                 ← EDA + batch effect analysis
│   ├── analysis_merged_subject_level.csv    ← merged dataset
│   ├── grifols_data_final.csv               ← raw metadata
│   ├── grifols_cytokine_data.csv            ← raw cytokine MFI values
│   ├── eda_subject_analysis/                ← EDA outputs
│   └── batch_effect_analysis/               ← batch validation outputs
├── PCA+KNN/
│   ├── PCA_KNN.ipynb                        ← Method 1: PCA + KNN (from scratch)
│   └── Outputs/
├── XGBOOST/
│   ├── Xgboost.py                           ← Method 2: XGBoost + grid search
│   ├── permutation_test.py                  ← validates accuracy above chance
│   ├── model_distinguisher.py               ← McNemar, calibration, LR coefficients
│   └── Outputs/
├── Extra-analyses/                          ← archived scripts, not part of pipeline
├── .gitignore
├── requirements.txt
└── README.md
```

## Recommended Run Order

```
1. Data/dataprep.py
2. Data/EDA+Batch_Effects.py
3. PCA+KNN/PCA_KNN.ipynb
4. XGBOOST/Xgboost.py
5. XGBOOST/permutation_test.py
6. XGBOOST/model_distinguisher.py
```

## Step-by-Step Instructions

### 1. Data Preparation
Run `dataprep.py` to merge raw cytokine and metadata files.

Input files (place in `Data/`):
- `grifols_data_final.csv` — subject metadata
- `grifols_cytokine_data.csv` — raw cytokine MFI values

Output:
- `Data/analysis_merged_subject_level.csv`

### 2. Exploratory Data Analysis + Batch Effect Analysis
Run `EDA.py` to generate all EDA figures and validate batch effects.

EDA outputs → `Data/eda_subject_analysis/`
- Cytokine distribution plots (raw + log1p)
- Correlation heatmap
- Sex-stratified boxplots
- PCA scree plot and PC1 vs PC2 scatter
- `figure1_data_overview_panel.png` ← paper Figure 1

Batch outputs → `Data/batch_effect_analysis/`
- Chi-square test: batch vs sex (p=0.165 — not significant)
- AUC comparison with/without batch
- SHAP stability analysis
- Clustermap and outlier analysis
- **Conclusion: batch excluded from all modeling**

### 3. PCA + KNN (Method 1 — from scratch)
Open and run `PCA+KNN/PCA_KNN.ipynb`

What it does:
- SVD-based PCA implemented from scratch
- Euclidean KNN with majority vote from scratch
- Skewness and outlier checks
- Grid search: PCA components ∈ {2..10} × K ∈ {3..7} = 45 combinations
- Stratified 5-fold CV
- Variance-weighted feature importance
- Exports preprocessed data for downstream models

Outputs → `PCA+KNN/Outputs/`
- `results_summary.csv` — all 45 grid search results
- `figures/` — scree plot, grid search heatmap, confusion matrix,
  PCA scatter, feature importance, per-fold accuracy

Best config: PCA=4, K=7, CV Accuracy=56.8%

### 4. XGBoost + Model Comparison (Method 2 — from scratch CV loop)
Run `XGBOOST/Xgboost.py`

What it does:
- Outlier clipping (99th percentile) + RobustScaler + ANOVA feature selection
- Grid search for XGBoost hyperparameters (manual CV loop — not GridSearchCV)
- Grid search for Logistic Regression C tuning
- 10×5-fold Repeated Stratified CV for all models:
  XGBoost, Logistic Regression, Random Forest, Gradient Boosting, SVM
- SHAP beeswarm + bar plots (feature directionality)
- CV-based ROC curve (not training set)

Outputs → `XGBOOST/Outputs/`
- `sex_model_comparison.csv` ← Table 1 in paper
- `results_table2_hyperparameters.csv` ← Table 2 in paper
- `sex_xgboost_feature_importance.csv`
- `results_shap_values.csv`
- `figures/` — comprehensive results panel, SHAP plots, ROC curve

### 5. Permutation Test
Run `XGBOOST/permutation_test.py`

What it does:
- Shuffles sex labels 1000 times and refits XGBoost each time
- Confirms that real accuracy exceeds 95% of permuted accuracies
- Provides p-value for paper: *"accuracy was significantly above chance"*

Outputs → `XGBOOST/Outputs/`
- `permutation_test_results.csv`
- `figures/permutation_test_plot.png`

### 6. Model Distinguisher
Run `XGBOOST/model_distinguisher.py` (requires step 4 to be run first)

What it does:
- **McNemar's test** — whether models make statistically different errors
- **Calibration curves** — whether predicted probabilities are trustworthy
- **Logistic regression coefficients** — which cytokines are higher in males vs females
- **Permutation feature importance** — model-agnostic feature ranking

Outputs → `XGBOOST/Outputs/`
- `model_distinguisher_results.csv` — McNemar p-values
- `logreg_coefficients.csv` — directionality per cytokine
- `permutation_importance.csv`
- `figures/` — calibration curves, coefficient plot, permutation importance

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Expected Runtime

| Script | Runtime |
|---|---|
| `dataprep.py` | < 1 min |
| `EDA+Batch_Effects.py` | < 1 min (SHAP bootstrap in batch analysis) |
| `PCA_KNN.ipynb` | < 1 min |
| `Xgboost.py` | < 1 min (grid search + 10×5-fold CV) |
| `permutation_test.py` | < 2 min (1000 permutations) |
| `model_distinguisher.py` | <1 min |

## Notes
- All scripts use relative paths — no hardcoded directories
- Random seed fixed at 42 throughout for reproducibility
