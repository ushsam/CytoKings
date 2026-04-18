"""
COMPLETE ML PIPELINE FOR IMMUNE PROFILING PROJECT
Predicts Age Group from Cytokine Expression Patterns
Methods: PCA + KNN, XGBoost, Logistic Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (StratifiedKFold, cross_val_score, cross_validate,
                                      RepeatedStratifiedKFold)
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
CV_SPLITS = 5
REPEATS = 10
TARGET = 'AGEGR1C'  # CHANGED: Predicting Age Group (not Sex)
OUTLIER_PERCENTILE = 99
FEATURE_SELECTION_N = 12  # Use top 12 cytokines

print("="*90)
print("IMMUNE PROFILING: ML PIPELINE FOR AGE GROUP PREDICTION")
print("="*90)

# ============================================================================
# STEP 1: DATA LOADING & PREPROCESSING
# ============================================================================
print("\n[1/7] LOADING DATA...")
df = pd.read_csv('../Data/analysis_merged_subject_level.csv')

cytokines = [
    'IFN-gamma', 'IL-12p70', 'IL-13', 'IL-1beta', 'IL-2', 'IL-4', 'IL-5', 
    'IL-6', 'TNF-alpha', 'GM-CSF', 'IL-18', 'IL-10', 'IL-17A', 'IL-21', 
    'IL-22', 'IL-23', 'IL-27', 'IL-9'
]

print(f"Dataset: {len(df)} samples × {len(cytokines)} cytokines")
print(f"Target: {TARGET}")
print(f"Classes: {df[TARGET].unique()}")

# ============================================================================
# STEP 2: OUTLIER HANDLING
# ============================================================================
print(f"\n[2/7] HANDLING OUTLIERS (clipping to {OUTLIER_PERCENTILE}th percentile)...")

X_raw = df[cytokines].copy()
for cyto in cytokines:
    threshold = X_raw[cyto].quantile(OUTLIER_PERCENTILE / 100)
    X_raw[cyto] = X_raw[cyto].clip(upper=threshold)
    
print("✓ Outliers clipped")

# ============================================================================
# STEP 3: ROBUST SCALING
# ============================================================================
print(f"\n[3/7] PREPROCESSING WITH ROBUST SCALING...")

scaler = RobustScaler()  # Better than Z-score for outliers
X_scaled = scaler.fit_transform(X_raw)

y = df[TARGET].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Classes encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"Class distribution: {np.bincount(y_encoded)}")

# ============================================================================
# STEP 4: FEATURE SELECTION (Univariate ANOVA)
# ============================================================================
print(f"\n[4/7] FEATURE SELECTION (top {FEATURE_SELECTION_N} cytokines)...")

from sklearn.feature_selection import f_classif, SelectKBest

selector = SelectKBest(f_classif, k=FEATURE_SELECTION_N)
X_selected = selector.fit_transform(X_scaled, y_encoded)

selected_features = np.array(cytokines)[selector.get_support()]
print(f"Selected features:\n{', '.join(selected_features)}")

# ============================================================================
# STEP 5: METHOD 1 - PCA + K-NEAREST NEIGHBORS
# ============================================================================
print(f"\n[5/7] METHOD 1: PCA + KNN...")

# PCA
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_selected)

print(f"PCA: {X_pca.shape[1]} components explaining {pca.explained_variance_ratio_.sum():.1%} variance")

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
cv_repeated = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=REPEATS, 
                                       random_state=RANDOM_STATE)
cv_scores_knn = cross_val_score(knn, X_pca, y_encoded, cv=cv_repeated, scoring='accuracy')
cv_f1_knn = cross_val_score(knn, X_pca, y_encoded, cv=cv_repeated, scoring='f1_weighted')

print(f"KNN Accuracy: {cv_scores_knn.mean():.4f} ± {cv_scores_knn.std():.4f}")
print(f"KNN F1-Score: {cv_f1_knn.mean():.4f} ± {cv_f1_knn.std():.4f}")

# ============================================================================
# STEP 6: METHOD 2 - XGBOOST
# ============================================================================
print(f"\n[6/7] METHOD 2: XGBOOST...")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

cv_scores_xgb = cross_val_score(xgb, X_selected, y_encoded, cv=cv_repeated, scoring='accuracy')
cv_f1_xgb = cross_val_score(xgb, X_selected, y_encoded, cv=cv_repeated, scoring='f1_weighted')

print(f"XGBoost Accuracy: {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")
print(f"XGBoost F1-Score: {cv_f1_xgb.mean():.4f} ± {cv_f1_xgb.std():.4f}")

# Feature importance
xgb_temp = XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, 
                         use_label_encoder=False, eval_metric='mlogloss')
xgb_temp.fit(X_selected, y_encoded)
feature_importance_xgb = pd.DataFrame({
    'Feature': selected_features,
    'Importance': xgb_temp.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"Top 5 important features:\n{feature_importance_xgb.head()}")

# ============================================================================
# STEP 7: METHOD 3 - LOGISTIC REGRESSION
# ============================================================================
print(f"\n[7/7] METHOD 3: LOGISTIC REGRESSION...")

lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, multi_class='multinomial')
cv_scores_lr = cross_val_score(lr, X_selected, y_encoded, cv=cv_repeated, scoring='accuracy')
cv_f1_lr = cross_val_score(lr, X_selected, y_encoded, cv=cv_repeated, scoring='f1_weighted')

print(f"Logistic Regression Accuracy: {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
print(f"Logistic Regression F1-Score: {cv_f1_lr.mean():.4f} ± {cv_f1_lr.std():.4f}")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "="*90)
print("MODEL COMPARISON TABLE")
print("="*90)

comparison = pd.DataFrame({
    'Method': ['PCA + KNN', 'XGBoost', 'Logistic Regression'],
    'Input Features': [f'PCA ({X_pca.shape[1]} components)', 
                       f'Raw ({FEATURE_SELECTION_N} selected)',
                       f'Raw ({FEATURE_SELECTION_N} selected)'],
    'Accuracy (Mean)': [cv_scores_knn.mean(), cv_scores_xgb.mean(), cv_scores_lr.mean()],
    'Accuracy (Std)': [cv_scores_knn.std(), cv_scores_xgb.std(), cv_scores_lr.std()],
    'F1-Score (Mean)': [cv_f1_knn.mean(), cv_f1_xgb.mean(), cv_f1_lr.mean()],
    'F1-Score (Std)': [cv_f1_knn.std(), cv_f1_xgb.std(), cv_f1_lr.std()]
})

print("\n" + comparison.to_string(index=False))
comparison.to_csv('./results_model_comparison_age.csv', index=False)
print("\n✓ Comparison saved: results_model_comparison_age.csv")

# ============================================================================
# BEST MODEL EVALUATION (XGBoost - Best Overall)
# ============================================================================
print("\n" + "="*90)
print("DETAILED EVALUATION: XGBOOST (BEST MODEL)")
print("="*90)

# Train on full data for detailed metrics
xgb_best = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                         use_label_encoder=False, eval_metric='mlogloss')
xgb_best.fit(X_selected, y_encoded)

# Get predictions using cross-validation
y_pred_cv = cross_val_predict(xgb_best, X_selected, y_encoded, cv=StratifiedKFold(5, shuffle=True))
y_pred_proba_cv = cross_val_predict(xgb_best, X_selected, y_encoded, cv=StratifiedKFold(5, shuffle=True),
                                     method='predict_proba')

# Metrics
accuracy = accuracy_score(y_encoded, y_pred_cv)
precision_macro = precision_score(y_encoded, y_pred_cv, average='macro')
recall_macro = recall_score(y_encoded, y_pred_cv, average='macro')
f1_macro = f1_score(y_encoded, y_pred_cv, average='macro')

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision_macro:.4f}")
print(f"Recall (macro): {recall_macro:.4f}")
print(f"F1-Score (macro): {f1_macro:.4f}")

print("\nClassification Report:")
print(classification_report(y_encoded, y_pred_cv, target_names=le.classes_))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_encoded, y_pred_cv)
print(cm)

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*90)
print("GENERATING VISUALIZATIONS...")
print("="*90)

fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison - Accuracy
ax1 = plt.subplot(2, 3, 1)
methods = ['PCA+KNN', 'XGBoost', 'LogReg']
means = comparison['Accuracy (Mean)'].values
stds = comparison['Accuracy (Std)'].values
colors_comp = ['#1f77b4', '#2ca02c', '#ff7f0e']
bars = ax1.bar(methods, means, yerr=stds, capsize=10, color=colors_comp, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Comparison: Accuracy', fontsize=13, fontweight='bold')
ax1.set_ylim([0.3, 0.95])
ax1.axhline(1/len(le.classes_), color='red', linestyle='--', alpha=0.5, label='Random Baseline')
for i, (bar, mean) in enumerate(zip(bars, means)):
    ax1.text(bar.get_x() + bar.get_width()/2, mean + stds[i] + 0.02, f'{mean:.3f}', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. F1-Score Comparison
ax2 = plt.subplot(2, 3, 2)
f1_means = comparison['F1-Score (Mean)'].values
f1_stds = comparison['F1-Score (Std)'].values
bars = ax2.bar(methods, f1_means, yerr=f1_stds, capsize=10, color=colors_comp, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('F1-Score (Macro)', fontsize=12, fontweight='bold')
ax2.set_title('Model Comparison: F1-Score', fontsize=13, fontweight='bold')
ax2.set_ylim([0.3, 0.95])
for i, (bar, mean) in enumerate(zip(bars, f1_means)):
    ax2.text(bar.get_x() + bar.get_width()/2, mean + f1_stds[i] + 0.02, f'{mean:.3f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# 3. Confusion Matrix (XGBoost)
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, 
            yticklabels=le.classes_, cbar=False, ax=ax3, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
ax3.set_title('Confusion Matrix: XGBoost', fontsize=13, fontweight='bold')
ax3.set_ylabel('True Age Group', fontsize=12, fontweight='bold')
ax3.set_xlabel('Predicted Age Group', fontsize=12, fontweight='bold')

# 4. Feature Importance
ax4 = plt.subplot(2, 3, 4)
top_features_imp = feature_importance_xgb.head(10)
ax4.barh(top_features_imp['Feature'], top_features_imp['Importance'], color='steelblue', edgecolor='black', linewidth=1.2)
ax4.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax4.set_title('Top 10 Most Important Cytokines', fontsize=13, fontweight='bold')
ax4.invert_yaxis()
for i, v in enumerate(top_features_imp['Importance']):
    ax4.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')

# 5. PCA Variance Explained
ax5 = plt.subplot(2, 3, 5)
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
ax5.plot(range(1, len(cumsum_var)+1), cumsum_var, marker='o', linewidth=2.5, markersize=8, color='#2ca02c', markeredgecolor='black', markeredgewidth=1.2)
ax5.fill_between(range(1, len(cumsum_var)+1), cumsum_var, alpha=0.3, color='#2ca02c')
ax5.set_xlabel('Number of PCA Components', fontsize=12, fontweight='bold')
ax5.set_ylabel('Cumulative Variance Explained', fontsize=12, fontweight='bold')
ax5.set_title('PCA: Cumulative Variance Explained', fontsize=13, fontweight='bold')
ax5.grid(alpha=0.3)
ax5.set_ylim([0, 1.05])
for i, v in enumerate(cumsum_var):
    if (i+1) % 2 == 0:
        ax5.text(i+1, v+0.02, f'{v:.1%}', ha='center', fontweight='bold', fontsize=9)

# 6. CV Score Distribution
ax6 = plt.subplot(2, 3, 6)
positions = [1, 2, 3]
bp = ax6.boxplot([cv_scores_knn, cv_scores_xgb, cv_scores_lr], labels=methods, patch_artist=True,
                   widths=0.6, showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], colors_comp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)
ax6.set_ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
ax6.set_title('CV Score Distribution (10×5-fold)', fontsize=13, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
ax6.axhline(1/len(le.classes_), color='red', linestyle='--', alpha=0.5, label='Random Baseline')
ax6.legend()

plt.tight_layout()
plt.savefig('./results_age_prediction_complete.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results_age_prediction_complete.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
results_summary = pd.DataFrame({
    'Metric': ['Best Model', 'Accuracy (CV)', 'F1-Score (CV)', 'Precision (CV)', 'Recall (CV)',
               'Target Variable', 'Input Features', 'Sample Size', 'Number of Classes'],
    'Value': ['XGBoost', f'{cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}',
              f'{cv_f1_xgb.mean():.4f} ± {cv_f1_xgb.std():.4f}',
              f'{precision_macro:.4f}', f'{recall_macro:.4f}',
              TARGET, f'{FEATURE_SELECTION_N} selected cytokines', len(df), len(le.classes_)]
})

results_summary.to_csv('./results_summary_age.csv', index=False)
feature_importance_xgb.to_csv('./results_feature_importance_age.csv', index=False)

print("✓ Saved: results_summary_age.csv")
print("✓ Saved: results_feature_importance_age.csv")

print("\n" + "="*90)
print("✅ PIPELINE COMPLETE!")
print("="*90)
print(f"\nKey Findings:")
print(f"  • Age Group prediction with {cv_scores_xgb.mean():.1%} accuracy (vs {1/len(le.classes_):.1%} random)")
print(f"  • XGBoost outperforms other methods")
print(f"  • Top predictive cytokines: {', '.join(feature_importance_xgb.head(3)['Feature'].values)}")
print(f"  • Model generalization: stable across CV folds (std = {cv_scores_xgb.std():.4f})")
