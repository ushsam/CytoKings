# Cytokine Expression Classification Pipeline (PCA + KNN)

## Objective

This pipeline evaluates whether cytokine expression patterns can predict biological sex and identifies key immune signals driving class separation.

It also produces clean, reusable datasets for benchmarking advanced machine learning models.

## Full Pipeline Overview

- Data loading and validation
- Z-score standardization
- PCA (dimensionality reduction)
- KNN classification
- Stratified 5-fold cross-validation
- Grid search over PCA components and K
- Feature importance estimation
- Visualization panel generation
- Downstream data export (core deliverable)

## Downstream Data Export (Important)

This section defines exact files used by other models.

After running the pipeline, the following files are generated:

### 1. Core ML Inputs (Used by All Downstream Models)

These are the primary files other models should load.

#### X_processed.npy

- Shape: (n_samples, n_features)
- Description: Standardized cytokine expression matrix

Used for:
- XGBoost
- SVM
- Random Forest
- Logistic Regression
- Neural Networks

```python
X = np.load("X_processed.npy")
```

#### y_labels.npy

- Shape: (n_samples,)
- Description: Binary classification labels
  - 1 = Male
  - 0 = Female

Used for all supervised learning tasks:

```python
y = np.load("y_labels.npy")
```

### 2. Dimensionality-Reduced Input (Optional Feature Set)

#### X_pca_best.npy

- Shape: (n_samples, best_n_pca)
- Description: PCA-transformed feature space using optimal configuration

Used for:
- Faster training models
- Linear classifiers
- Visualization-based ML
- Low-dimensional embeddings

```python
X_pca = np.load("X_pca_best.npy")
```

### 3. PCA Model Parameters (For Reproducibility)

#### pca_components.npy

- Shape: (best_n_pca, n_features)
- Description: PCA projection matrix learned from full dataset

Used for:
- Reconstructing PCA transformation
- Applying identical feature mapping to new data

```python
components = np.load("pca_components.npy")
```

### 4. Benchmarking and Interpretability Files

#### results_summary.csv

Contains:
- PCA components tested
- K values tested
- Cross-validation accuracy (mean and standard deviation)

Used for:
- Model benchmarking
- Hyperparameter comparison
- Reproducibility audits

#### feature_importance.csv

Columns:
- Cytokine name
- Importance score (variance-weighted PCA loadings)

Used for:
- Biological interpretation
- Feature selection
- Biomarker discovery

## Downstream Usage Protocol

### Step 1: Load core data

```python
X = np.load("X_processed.npy")
y = np.load("y_labels.npy")
```

### Step 2: Train any model

Examples: XGBoost, SVM, Random Forest

- Perform your own train/test split or cross-validation
- IMPORTANT: re-standardize within each fold

### Step 3: Optional PCA usage

```python
X_pca = np.load("X_pca_best.npy")
```

Use when:
- Reducing dimensionality
- Improving training speed
- Avoiding high-dimensional noise

## Critical Notes

- X_processed.npy is already globally standardized  
  -> Must be re-standardized inside cross-validation for any new model  

- PCA transformation is provided for consistency  
  -> Downstream models may recompute PCA independently  

- Cross-validation strategy in this pipeline is a reference-only baseline  

## Baseline Performance Reference

Use this pipeline as a benchmark:

- Accuracy
- Balanced Accuracy
- Sensitivity
- Specificity
- F1 Score

All downstream models should report performance relative to this baseline.

## Design Philosophy

- No black-box ML dependencies in core pipeline
- Fully reproducible feature engineering
- Strict separation between preprocessing and modeling
- Explicit export layer for cross-model compatibility
- Biological interpretability prioritized via cytokine importance ranking
