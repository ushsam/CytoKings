"""
XGBoost Pipeline for Sex Classification from Cytokine Expression
Target: SEXC (Biological Sex - Binary Classification: 1=Male, 0=Female)
Features: 18 cytokines (standardized expression levels)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from xgboost import XGBClassifier
import warnings
import joblib
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data"
OUTPUT_DIR  = BASE_DIR / "XGBoost-cytokine-sex" / "old-binary-sex-outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*70)
print("Loading processed data...")
print("="*70)

X = np.load(BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokines_only" / "X_processed.npy")
y = np.load(BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokines_only" / "y_labels.npy")

cytokines = [
    'IFN-gamma', 'IL-12p70', 'IL-13', 'IL-1beta', 'IL-2', 'IL-4', 'IL-5',
    'IL-6', 'TNF-alpha', 'GM-CSF', 'IL-18', 'IL-10', 'IL-17A', 'IL-21',
    'IL-22', 'IL-23', 'IL-27', 'IL-9'
]

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Features: {len(cytokines)} cytokines")

# ============================================================================
# 2. HYPERPARAMETER TUNING WITH GRID SEARCH
# ============================================================================
print("\n" + "="*70)
print("Hyperparameter Tuning with Grid Search...")
print("="*70)

param_grid = {
    'n_estimators':    [100, 200, 300],
    'max_depth':       [3, 5, 7, 9],
    'learning_rate':   [0.01, 0.05, 0.1],
    'subsample':       [0.8, 0.9],
    'colsample_bytree':[0.8, 0.9],
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False,
                    eval_metric='logloss')
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy',
                           n_jobs=-1, verbose=1)
grid_search.fit(X, y)

best_params = grid_search.best_params_
print(f"\nBest parameters: {best_params}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# ============================================================================
# 3. TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================
print("\n" + "="*70)
print("Training final XGBoost model...")
print("="*70)

model = XGBClassifier(**best_params, random_state=42,
                      use_label_encoder=False, eval_metric='logloss')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

model.fit(X, y)

# ============================================================================
# 4. EVALUATION METRICS
# ============================================================================
print("\n" + "="*70)
print("Evaluation Metrics")
print("="*70)

y_pred       = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

accuracy  = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall    = recall_score(y, y_pred)
f1        = f1_score(y, y_pred)
roc_auc   = roc_auc_score(y, y_pred_proba)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\n" + "-"*70)
print("Classification Report")
print("-"*70)
print(classification_report(y, y_pred, target_names=['Female', 'Male']))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("Feature Importance (Top 10 Cytokines)")
print("="*70)

feature_importance = pd.DataFrame({
    'Cytokine':   cytokines,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("Saving results...")
print("="*70)

joblib.dump(model, OUTPUT_DIR / "xgboost_sex_classifier.pkl")
print("✓ Model saved")

feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
print("✓ Feature importance saved")

results_df = pd.DataFrame([{
    'Best Parameters':  str(best_params),
    'CV Mean Accuracy': f"{cv_scores.mean():.4f}",
    'CV Std':           f"{cv_scores.std():.4f}",
    'Train Accuracy':   f"{accuracy:.4f}",
    'Train Precision':  f"{precision:.4f}",
    'Train Recall':     f"{recall:.4f}",
    'Train F1-Score':   f"{f1:.4f}",
    'Train ROC-AUC':    f"{roc_auc:.4f}",
}])
results_df.to_csv(OUTPUT_DIR / "xgboost_results_summary.csv", index=False)
print("✓ Results summary saved")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("Creating visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top 10 Feature Importance
top_features = feature_importance.head(10)
axes[0, 0].barh(top_features['Cytokine'], top_features['Importance'])
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Top 10 Most Important Cytokines')
axes[0, 0].invert_yaxis()

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
axes[0, 1].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')
axes[0, 1].set_xticks([0, 1]);  axes[0, 1].set_xticklabels(['Female', 'Male'])
axes[0, 1].set_yticks([0, 1]);  axes[0, 1].set_yticklabels(['Female', 'Male'])
for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')

# ROC Curve
fpr, tpr, _ = roc_curve(y, y_pred_proba)
axes[1, 0].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}')
axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()

# CV scores
axes[1, 1].bar(range(1, 6), cv_scores)
axes[1, 1].axhline(cv_scores.mean(), color='r', linestyle='--',
                   label=f'Mean: {cv_scores.mean():.3f}')
axes[1, 1].set_xlabel('Fold')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('5-Fold Cross-Validation Scores')
axes[1, 1].set_xticks(range(1, 6))
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / "xgboost_results_visualization.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Visualization saved")

print("\n" + "="*70)
print("PIPELINE COMPLETE — SEX CLASSIFICATION (XGBoost)")
print("="*70)
print(f"  Accuracy     : {accuracy:.4f}")
print(f"  CV Accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  ROC-AUC      : {roc_auc:.4f}")
print(f"  Top cytokine : {feature_importance.iloc[0]['Cytokine']}")
print(f"  Outputs      : {OUTPUT_DIR}")
print(f"  Figures      : {FIGURES_DIR}")
print("="*70)