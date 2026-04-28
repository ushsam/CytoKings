"""
XGBoost Sex Classification Pipeline (SEXC Prediction)
========================================================
Target: SEXC (Biological Sex - Binary Classification)
Features: 18 cytokines from healthy adults
Method: XGBoost with model comparison (LogReg, RF, SVM, GB)
Cross-validation: 10×5-fold Repeated Stratified K-Fold
Preprocessing: Outlier clipping (99th percentile) + Robust Scaling + ANOVA Feature Selection
"""

import os
import pathlib
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

# Models
from XGBOOST.xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from pathlib import Path

# ============================================================================
# CONFIGURATION — edit these values to adapt the pipeline to your dataset
# ============================================================================

BASE_DIR       = Path(__file__).resolve().parent.parent

DATA_PATH   = BASE_DIR / "Data" / "analysis_merged_subject_level.csv"
OUTPUT_DIR     = BASE_DIR / "XGBoost-cytokine-sex" / "final-sex-outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Columns to use as features. Set to None to auto-detect all numeric columns.
FEATURE_COLUMNS = [
    'IFN-gamma', 'IL-12p70', 'IL-13', 'IL-1beta', 'IL-2', 'IL-4', 'IL-5',
    'IL-6', 'TNF-alpha', 'GM-CSF', 'IL-18', 'IL-10', 'IL-17A', 'IL-21',
    'IL-22', 'IL-23', 'IL-27', 'IL-9'
]

TARGET_COLUMN     = 'SEXC'  # Column name of the binary target
POSITIVE_CLASS    = 'Male'  # Value in TARGET_COLUMN treated as class 1
N_FEATURES_SELECT = 12      # How many top features ANOVA should keep (None = keep all)
 
# ============================================================================
# OUTPUT LOG SETUP — all print output is mirrored to a text file
# ============================================================================
 
import sys
 
class Tee:
    """Mirrors stdout to both the console and a log file."""
    def __init__(self, filepath):
        self._file = open(filepath, 'w', encoding='utf-8')
        self._stdout = sys.stdout
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()
 
_log_path = OUTPUT_DIR / "pipeline_output.txt"
sys.stdout = Tee(_log_path)
 
# ============================================================================
# 1. LOAD & PREPROCESS DATA
# ============================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)
 
df = pd.read_csv(DATA_PATH)
print(f"✓ Data loaded: {df.shape[0]} samples × {df.shape[1]} columns")
print(f"  Source: {DATA_PATH}")
 
# Resolve feature columns
if FEATURE_COLUMNS is None:
    FEATURE_COLUMNS = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COLUMN in FEATURE_COLUMNS:
        FEATURE_COLUMNS.remove(TARGET_COLUMN)
    print(f"  (Auto-detected {len(FEATURE_COLUMNS)} numeric feature columns)")
 
# Resolve number of features to select
n_select = N_FEATURES_SELECT if N_FEATURES_SELECT is not None else len(FEATURE_COLUMNS)
n_select = min(n_select, len(FEATURE_COLUMNS))
 
# Build X and y
X = df[FEATURE_COLUMNS].copy()
y = (df[TARGET_COLUMN] == POSITIVE_CLASS).astype(int).values
 
neg_label = f"not {POSITIVE_CLASS}"
n_pos = (y == 1).sum()
n_neg = (y == 0).sum()
 
print(f"✓ Features: {len(FEATURE_COLUMNS)} columns")
print(f"✓ Target: {TARGET_COLUMN}  (positive class = '{POSITIVE_CLASS}')")
print(f"  - Negative (0): {n_neg}")
print(f"  - Positive (1): {n_pos}")
print(f"  - Balance: {n_neg}/{n_pos} ratio = {n_neg / n_pos:.3f}")
 
# ============================================================================
# 2. OUTLIER HANDLING (99th Percentile Clipping)
# ============================================================================
print("\n" + "=" * 80)
print("OUTLIER HANDLING (99th Percentile Clipping)")
print("=" * 80)
 
X_clipped = X.copy()
for col in X_clipped.columns:
    p99 = X_clipped[col].quantile(0.99)
    X_clipped[col] = np.minimum(X_clipped[col], p99)
 
print(f"✓ Clipped all features to 99th percentile")
 
# ============================================================================
# 3. ROBUST SCALING
# ============================================================================
print("\n" + "=" * 80)
print("ROBUST SCALING")
print("=" * 80)
 
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_clipped)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS)
 
print(f"✓ Applied RobustScaler (handles outliers better than Z-score)")
 
# ============================================================================
# 4. FEATURE SELECTION (ANOVA)
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE SELECTION (ANOVA)")
print("=" * 80)
 
selector = SelectKBest(f_classif, k=n_select)
X_selected = selector.fit_transform(X_scaled, y)
 
mask              = selector.get_support()
selected_features = [FEATURE_COLUMNS[i] for i in range(len(FEATURE_COLUMNS)) if mask[i]]
f_scores          = selector.scores_
 
feature_selection_df = pd.DataFrame({
    'Feature':       FEATURE_COLUMNS,
    'ANOVA_F_Score': f_scores,
    'Selected':      mask
}).sort_values('ANOVA_F_Score', ascending=False)
 
print(f"✓ Selected top {n_select} features (ANOVA)")
print(f"  Feature/Sample ratio: {n_select}/{X.shape[0]} = {n_select/X.shape[0]:.3f} (good: <0.05)")
print("\nTop 5 selected features:")
print(feature_selection_df[feature_selection_df['Selected']].head(5).to_string(index=False))
 
feature_selection_df.to_csv(OUTPUT_DIR / 'sex_feature_selection_anova.csv', index=False)
print("\n✓ Saved: Outputs/sex_feature_selection_anova.csv")
 
# ============================================================================
# 5. MODEL SETUP & CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL SETUP & CROSS-VALIDATION")
print("=" * 80)
 
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
print(f"✓ Using RepeatedStratifiedKFold (5 folds × 10 repeats = 50 total folds)")
 
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                         random_state=42, use_label_encoder=False,
                                         eval_metric='logloss', verbosity=0),
}
 
# ============================================================================
# 6. CROSS-VALIDATION FOR ALL MODELS
# ============================================================================
print("\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS (10×5-fold)")
print("=" * 80)
 
cv_results = {}
 
for model_name, model in models.items():
    print(f"\n{model_name}...")
 
    accuracies, precisions, recalls, f1s, roc_aucs = [], [], [], [], []
 
    for train_idx, test_idx in cv.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
 
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
 
        try:
            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            roc_aucs.append(roc_auc_score(y_test, y_proba))
        except Exception:
            accuracies.append(0.0); precisions.append(0.0)
            recalls.append(0.0);    f1s.append(0.0); roc_aucs.append(0.0)
 
    cv_results[model_name] = {
        'Accuracy':      f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        'Precision':     f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
        'Recall':        f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
        'F1-Score':      f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
        'ROC-AUC':       f"{np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}",
        'Accuracy_Mean': np.mean(accuracies),
    }
 
    print(f"  ✓ Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  ✓ F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  ✓ ROC-AUC:  {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
 
# ============================================================================
# 7. SAVE MODEL COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE")
print("=" * 80)
 
cv_results_df = pd.DataFrame(cv_results).T
print("\n" + cv_results_df.to_string())
 
cv_results_df.to_csv(OUTPUT_DIR / 'sex_model_comparison.csv')
print("\n✓ Saved: Outputs/sex_model_comparison.csv")
 
# ============================================================================
# 8. TRAIN FINAL XGBoost & GET FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("FINAL XGBOOST MODEL & FEATURE IMPORTANCE")
print("=" * 80)
 
xgb_final = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                           random_state=42, use_label_encoder=False,
                           eval_metric='logloss', verbosity=0)
xgb_final.fit(X_selected, y)
 
feature_importance_df = pd.DataFrame({
    'Feature':    selected_features,
    'Importance': xgb_final.feature_importances_
}).sort_values('Importance', ascending=False)
 
print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))
 
feature_importance_df.to_csv(OUTPUT_DIR / 'sex_xgboost_feature_importance.csv', index=False)
print("\n✓ Saved: Outputs/sex_xgboost_feature_importance.csv")
 
# ============================================================================
# 9. FINAL MODEL EVALUATION (On Full Data)
# ============================================================================
print("\n" + "=" * 80)
print("FINAL XGBOOST EVALUATION")
print("=" * 80)
 
y_pred_final  = xgb_final.predict(X_selected)
y_proba_final = xgb_final.predict_proba(X_selected)[:, 1]
 
accuracy_final  = accuracy_score(y, y_pred_final)
precision_final = precision_score(y, y_pred_final, zero_division=0)
recall_final    = recall_score(y, y_pred_final, zero_division=0)
f1_final        = f1_score(y, y_pred_final, zero_division=0)
roc_auc_final   = roc_auc_score(y, y_proba_final)
cm              = confusion_matrix(y, y_pred_final)
 
print(f"Accuracy:  {accuracy_final:.4f}")
print(f"Precision: {precision_final:.4f}")
print(f"Recall:    {recall_final:.4f}")
print(f"F1-Score:  {f1_final:.4f}")
print(f"ROC-AUC:   {roc_auc_final:.4f}")
 
print("\nConfusion Matrix:")
print(f"  True Negatives  (Correctly classified {neg_label}):  {cm[0, 0]}")
print(f"  False Positives ({neg_label} misclassified):         {cm[0, 1]}")
print(f"  False Negatives ({POSITIVE_CLASS} misclassified):    {cm[1, 0]}")
print(f"  True Positives  (Correctly classified {POSITIVE_CLASS}): {cm[1, 1]}")
 
print("\nClassification Report:")
print(classification_report(y, y_pred_final,
      target_names=[neg_label, POSITIVE_CLASS], zero_division=0))
 
# ============================================================================
# 10. COMPREHENSIVE VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)
 
models_sorted = sorted(cv_results.items(), key=lambda x: x[1]['Accuracy_Mean'], reverse=True)
model_names   = [m[0] for m in models_sorted]
acc_means     = [m[1]['Accuracy_Mean'] for m in models_sorted]
 
fig = plt.figure(figsize=(18, 12))
gs  = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
 
# 1. Model Comparison - Accuracy
ax1 = fig.add_subplot(gs[0, 0])
colors_bar = ['#FF6B6B' if m == 'XGBoost' else '#4ECDC4' for m in model_names]
ax1.barh(model_names, acc_means, color=colors_bar)
ax1.set_xlabel('Accuracy')
ax1.set_title(f'Model Comparison: CV Accuracy ({TARGET_COLUMN})',
              fontsize=11, fontweight='bold')
ax1.set_xlim(0.4, 0.8)
for i, v in enumerate(acc_means):
    ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
 
# 2. Feature Importance (Top 10)
ax2 = fig.add_subplot(gs[0, 1])
top_features = feature_importance_df.head(10)
colors_feat  = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(top_features)))
ax2.barh(top_features['Feature'], top_features['Importance'], color=colors_feat)
ax2.set_xlabel('Importance Score')
ax2.set_title('Top 10 Most Important Features (XGBoost)', fontsize=11, fontweight='bold')
ax2.invert_yaxis()
 
# 3. Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3,
            xticklabels=[neg_label, POSITIVE_CLASS],
            yticklabels=[neg_label, POSITIVE_CLASS])
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')
ax3.set_title('XGBoost Confusion Matrix', fontsize=11, fontweight='bold')
 
# 4. ROC Curve
ax4 = fig.add_subplot(gs[1, 0])
fpr, tpr, _ = roc_curve(y, y_proba_final)
ax4.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC = {roc_auc_final:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curve (XGBoost)', fontsize=11, fontweight='bold')
ax4.legend(loc='lower right')
ax4.grid(alpha=0.3)
 
# 5. Class Distribution
ax5 = fig.add_subplot(gs[1, 1])
class_counts = [n_neg, n_pos]
ax5.bar([neg_label, POSITIVE_CLASS], class_counts,
        color=['#FF9999', '#66B2FF'], edgecolor='black', linewidth=1.5)
ax5.set_ylabel('Count')
ax5.set_title(f'Class Distribution ({TARGET_COLUMN})', fontsize=11, fontweight='bold')
for i, v in enumerate(class_counts):
    ax5.text(i, v + 1, str(v), ha='center', fontsize=10, fontweight='bold')
 
# 6. Top 3 Models Accuracy Heatmap
ax6 = fig.add_subplot(gs[1, 2])
top3_names  = model_names[:3]
top3_accs   = np.array([[cv_results[m]['Accuracy_Mean'] for m in top3_names]])
sns.heatmap(top3_accs, annot=True, fmt='.3f', cmap='YlOrRd',
            cbar_kws={'label': 'Accuracy'}, ax=ax6,
            xticklabels=top3_names, yticklabels=['CV Accuracy'])
ax6.set_title('Top 3 Models Performance', fontsize=11, fontweight='bold')
 
# 7. Feature Selection Pie
ax7 = fig.add_subplot(gs[2, 0])
n_sel     = int(mask.sum())
n_not_sel = len(FEATURE_COLUMNS) - n_sel
ax7.pie([n_sel, n_not_sel],
        labels=[f'Selected\n({n_sel})', f'Not Selected\n({n_not_sel})'],
        autopct='%1.1f%%', colors=['#90EE90', '#FFB6C6'], startangle=90)
ax7.set_title(f'ANOVA Feature Selection ({n_sel}/{len(FEATURE_COLUMNS)} features)',
              fontsize=11, fontweight='bold')
 
# 8. All Models Accuracy Bar Chart
ax8 = fig.add_subplot(gs[2, 1:])
bar_colors = ['#FF6B6B' if m == 'XGBoost' else '#4ECDC4' for m in model_names]
ax8.bar(model_names, acc_means, color=bar_colors)
ax8.set_ylabel('Score')
ax8.set_xlabel('Model')
ax8.set_title('Cross-Validated Accuracy by Model', fontsize=11, fontweight='bold')
ax8.set_ylim(0.4, 0.8)
ax8.grid(axis='y', alpha=0.3)
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')
 
plt.savefig(FIGURES_DIR / 'sex_classification_comprehensive_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Outputs/sex_classification_comprehensive_results.png")
plt.close()
 
# --- Individual panels ---
SAVE_INDIVIDUAL_PANELS = True  # Set to False to skip individual panel saving
 
if SAVE_INDIVIDUAL_PANELS:
    print("\nSaving individual panels...")
 
    # 1. Model Comparison - Accuracy
    fig1, ax = plt.subplots(figsize=(8, 5))
    ax.barh(model_names, acc_means, color=['#FF6B6B' if m == 'XGBoost' else '#4ECDC4' for m in model_names])
    ax.set_xlabel('Accuracy')
    ax.set_title(f'Model Comparison: CV Accuracy ({TARGET_COLUMN})', fontsize=12, fontweight='bold')
    ax.set_xlim(0.4, 0.8)
    for i, v in enumerate(acc_means):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panel_1_model_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: Outputs/panel_1_model_accuracy.png")
 
    # 2. Feature Importance
    fig2, ax = plt.subplots(figsize=(8, 5))
    top_feat    = feature_importance_df.head(10)
    colors_feat = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(top_feat)))
    ax.barh(top_feat['Feature'], top_feat['Importance'], color=colors_feat)
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 10 Most Important Features (XGBoost)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panel_2_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: Outputs/panel_2_feature_importance.png")
 
    # 3. Confusion Matrix
    fig3, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=[neg_label, POSITIVE_CLASS],
                yticklabels=[neg_label, POSITIVE_CLASS])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('XGBoost Confusion Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panel_3_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: Outputs/panel_3_confusion_matrix.png")
 
    # 4. ROC Curve
    fig4, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y, y_proba_final)
    ax.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC = {roc_auc_final:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (XGBoost)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panel_4_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: Outputs/panel_4_roc_curve.png")
 
    # 5. Class Distribution
    fig5, ax = plt.subplots(figsize=(5, 4))
    ax.bar([neg_label, POSITIVE_CLASS], [n_neg, n_pos],
           color=['#FF9999', '#66B2FF'], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Count')
    ax.set_title(f'Class Distribution ({TARGET_COLUMN})', fontsize=12, fontweight='bold')
    for i, v in enumerate([n_neg, n_pos]):
        ax.text(i, v + 1, str(v), ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panel_5_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: Outputs/panel_5_class_distribution.png")
 
    # 6. Top 3 Models Heatmap
    fig6, ax = plt.subplots(figsize=(6, 3))
    top3_names = model_names[:3]
    top3_accs  = np.array([[cv_results[m]['Accuracy_Mean'] for m in top3_names]])
    sns.heatmap(top3_accs, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Accuracy'}, ax=ax,
                xticklabels=top3_names, yticklabels=['CV Accuracy'])
    ax.set_title('Top 3 Models Performance', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panel_6_top3_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: Outputs/panel_6_top3_heatmap.png")
 
    # 7. Feature Selection Pie
    fig7, ax = plt.subplots(figsize=(5, 5))
    n_sel     = int(mask.sum())
    n_not_sel = len(FEATURE_COLUMNS) - n_sel
    ax.pie([n_sel, n_not_sel],
           labels=[f'Selected\n({n_sel})', f'Not Selected\n({n_not_sel})'],
           autopct='%1.1f%%', colors=['#90EE90', '#FFB6C6'], startangle=90)
    ax.set_title(f'ANOVA Feature Selection ({n_sel}/{len(FEATURE_COLUMNS)} features)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panel_7_feature_selection_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: Outputs/panel_7_feature_selection_pie.png")
 
    # 8. All Models Accuracy Bar
    fig8, ax = plt.subplots(figsize=(8, 5))
    ax.bar(model_names, acc_means,
           color=['#FF6B6B' if m == 'XGBoost' else '#4ECDC4' for m in model_names])
    ax.set_ylabel('Score')
    ax.set_xlabel('Model')
    ax.set_title('Cross-Validated Accuracy by Model', fontsize=12, fontweight='bold')
    ax.set_ylim(0.4, 0.8)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'panel_8_accuracy_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: Outputs/panel_8_accuracy_by_model.png")
 
# ============================================================================
# 11. SAVE RESULTS SUMMARY CSV
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS SUMMARY")
print("=" * 80)
 
results_summary = {
    'Target':                  TARGET_COLUMN,
    'Positive_Class':          POSITIVE_CLASS,
    'Samples':                 X.shape[0],
    'Features_Original':       len(FEATURE_COLUMNS),
    'Features_Selected':       n_select,
    'Feature_Sample_Ratio':    n_select / X.shape[0],
    'Negative_Count':          n_neg,
    'Positive_Count':          n_pos,
    'Class_Balance_Ratio':     n_neg / n_pos,
    'XGBoost_CV_Accuracy':     f"{cv_results['XGBoost']['Accuracy_Mean']:.4f}",
    'XGBoost_CV_F1':           cv_results['XGBoost']['F1-Score'],
    'XGBoost_CV_ROC_AUC':      cv_results['XGBoost']['ROC-AUC'],
    'XGBoost_Train_Accuracy':  f"{accuracy_final:.4f}",
    'XGBoost_Train_F1':        f"{f1_final:.4f}",
    'XGBoost_Train_ROC_AUC':   f"{roc_auc_final:.4f}",
    'Best_Model':              model_names[0],
    'Best_Model_Accuracy':     f"{acc_means[0]:.4f}",
}
 
pd.DataFrame([results_summary]).T.rename(columns={0: 'Value'}).to_csv(
    OUTPUT_DIR / 'sex_classification_results_summary.csv'
)
print("✓ Saved: Outputs/sex_classification_results_summary.csv")
 
# ============================================================================
# 12. PRINT FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print(f"SUMMARY — {TARGET_COLUMN} CLASSIFICATION RESULTS")
print("=" * 80)
print(f"\n📊 Dataset: {X.shape[0]} samples  |  Source: {DATA_PATH.name}")
print(f"   - {neg_label}: {n_neg}")
print(f"   - {POSITIVE_CLASS}: {n_pos}")
print(f"\n🔧 Preprocessing:")
print(f"   - Outlier clipping : 99th percentile")
print(f"   - Scaling          : RobustScaler")
print(f"   - Feature selection: ANOVA ({n_select} of {len(FEATURE_COLUMNS)} features)")
print(f"\n🤖 Model Performance (10×5-fold CV):")
print(f"   - Best Model : {model_names[0]} ({acc_means[0]:.1%} accuracy)")
print(f"   - XGBoost    : {cv_results['XGBoost']['Accuracy_Mean']:.1%} accuracy")
print(f"\n⭐ Top 3 Important Features:")
for i, (_, row) in enumerate(feature_importance_df.head(3).iterrows(), 1):
    print(f"   {i}. {row['Feature']} ({row['Importance']:.3f})")
print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
print("\n" + "=" * 80)
print("✓ Pipeline completed successfully!")
print("=" * 80)
 
# Close the log file tee
sys.stdout.close()
sys.stdout = sys.stdout._stdout
print(f"✓ Log saved: {_log_path}")