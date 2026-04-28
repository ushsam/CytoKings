
## Pipeline

### 1. Data Preparation
Run `dataprep.py` to merge the raw cytokine and metadata files into a
single subject-level CSV.

Input files (place in `Data/`):
- `grifols_data_final.csv` — subject metadata
- `grifols_cytokine_data.csv` — raw cytokine MFI values

Output:
- `Data/analysis_merged_subject_level.csv`

### 2. Batch Effect Analysis
Run `batch_test.ipynb` to verify that collection date (batch) does not
confound sex classification before any modeling is performed.

Input:
- `Data/analysis_merged_subject_level.csv`

Tests performed:
- Chi-square test of independence between batch and sex
- Cross-validated ROC-AUC comparison with and without batch as a feature
- SHAP feature stability analysis across bootstrap resamples

Result: chi-square p=0.165 — no significant batch-sex association detected.
Batch is excluded from all downstream modeling.

Output:
- `Data/eda_subject_analysis/batch_sex_chisquare.csv`
- `Data/eda_subject_analysis/batch_auc_comparison.csv`

### 3. Exploratory Data Analysis
Run `EDA.py` to generate all EDA figures and summary statistics.
Note: EDA explores all demographic variables for completeness.
The downstream modeling and paper focus exclusively on sex prediction.

Input:
- `Data/analysis_merged_subject_level.csv`

Output folder: `Data/eda_subject_analysis/`
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
- `figure1_data_overview_panel.png`

### 4. Models
[Add model scripts here as completed]

### Step 4: PCA + KNN Classification
- SVD-based PCA implemented from scratch
- Grid search over PCA components (2-10) and K (3-7)
- Stratified 5-fold cross-validation
- Fuzzy C-Means soft clustering comparison

### Steps 5-7: XGBoost + Multiple Models (In Progress)
- XGBoost, Logistic Regression
- ANOVA-based feature selection
- SHAP feature importance analysis

## Recommended Run Order
1. `dataprep.py`
2. `batch_test.ipynb`
3. `EDA.py`
4. Model scripts (in progress)

## Requirements
pip install numpy==1.26.4 pandas==2.2.1 matplotlib==3.8.3 seaborn==0.13.2 scikit-learn==1.4.1 xgboost==2.0.3 shap==0.44.1 scipy==1.12.0