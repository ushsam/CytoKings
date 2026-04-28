## Pipeline

### 1. Data Preparation
Run `dataprep.py` to merge the raw cytokine and metadata files into a
single subject-level CSV.

Input files (place in `Data/`):
- `grifols_data_final.csv` — subject metadata
- `grifols_cytokine_data.csv` — raw cytokine MFI values

Output:
- `Data/analysis_merged_subject_level.csv`

---

### 2. Exploratory Data Analysis + Batch Effect Analysis
Run `EDA.py` to generate all EDA figures, summary statistics, and
batch effect validation. Note: EDA explores all demographic variables
for completeness. Downstream modeling and the paper focus exclusively
on sex prediction.

Input:
- `Data/analysis_merged_subject_level.csv`

#### EDA Outputs → `Data/eda_subject_analysis/`
- `cytokine_summary_stats.csv`
- `cytokine_corr_matrix.csv`
- `missingness_summary.csv`
- Per-cytokine distribution plots (raw + log1p)
- `cytokine_corr_heatmap.png`
- `SEXC_counts.png`
- Sex-stratified boxplots per cytokine
- Age regression plots per cytokine
- `PCA_scree.png`
- `PCA_PC1_PC2_by_SEXC.png`
- `PCA_loadings.csv`
- `figure1_data_overview_panel.png` ← paper Figure 1

#### Batch Effect Outputs → `Data/batch_effect_analysis/`
Tests performed:
- Chi-square test of independence between batch and sex
- Cross-validated ROC-AUC comparison with and without batch feature
- SHAP feature stability analysis across 10 bootstrap resamples
- IQR and z-score outlier detection per cytokine
- Cytokine correlation heatmap and clustermap

Result: chi-square p=0.165 — no significant batch-sex association.
Batch excluded from all downstream modeling.

- `batch_sex_chisquare.csv`
- `batch_mixing_table.csv`
- `batch_auc_comparison.csv`
- `shap_stability_no_batch.csv`
- `shap_stability_with_batch.csv`
- `outlier_iqr_summary.csv`
- `outlier_zscore_summary.csv`
- `top_outlier_samples.csv`
- `batch_corr_heatmap.png`
- `batch_clustermap.png`
- `IL21_vs_IL17A_scatter.png`

---

### 3. Models
[Add model scripts here as completed]

#### Step 3: PCA + KNN Classification
- SVD-based PCA implemented from scratch
- Grid search over PCA components (2–10) and K (3–7)
- Stratified 5-fold cross-validation
- Fuzzy C-Means soft clustering comparison

#### Steps 4–5: Supervised Classification (In Progress)
- XGBoost with ANOVA-based feature selection (from scratch CV loop)
- Logistic Regression L2 (sklearn)
- SHAP feature importance analysis

---

## Recommended Run Order

1. dataprep.py
2. EDA+Bath_Effects.py
3. Model scripts

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```