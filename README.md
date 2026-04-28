
## Pipeline

### 1. Data Preparation
Run `dataprep.py` to merge the raw cytokine and metadata files into a single subject-level CSV.

Input files (place in `Data/`):
- `grifols_data_final.csv` — subject metadata
- `grifols_cytokine_data.csv` — raw cytokine MFI values

Output:
- `Data/analysis_merged_subject_level.csv`

### 2. Exploratory Data Analysis
Run `EDA.py` to generate all EDA figures and summary statistics.

Output folder: `Data/eda_subject_analysis/`
- `cytokine_summary_stats.csv`
- `cytokine_corr_matrix.csv`
- `missingness_summary.csv`
- Per-cytokine distribution plots (raw + log1p)
- Cytokine correlation heatmap
- Demographic distribution plots
- Sex-stratified boxplots per cytokine
- Age regression plots per cytokine
- PCA scree plot and PC1 vs PC2 scatter

### 3. Models
[Add model scripts here as completed]

### Step 3: PCA + KNN Classification
- Separate PCAs per modality (cytokines and cell types)
- Grid search over PCA components (2–10) and K (3–7)
- Stratified 5-fold cross-validation
- Soft clustering (Fuzzy C-Means) comparison

### Steps 4–7: XGBoost + Multiple Models (In Progress)
- XGBoost, Random Forest, Logistic Regression, SVM, Ensemble
- Feature importance analysis