"""
COMPLETE ML PIPELINE FOR IMMUNE PROFILING PROJECT
Predicts Age Group (Binary: Young vs Older) from Cytokine Expression Patterns
Methods: PCA + KNN, XGBoost, Logistic Regression
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      RepeatedStratifiedKFold)
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.feature_selection import f_classif, SelectKBest
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data"
OUTPUT_DIR  = BASE_DIR / "XGBoost-cytokine-age" / "Outputs" / "binary_age"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SAVE PANEL HELPER
# ============================================================================
def save_panel(fig, ax, filename, pad=1.15):
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        extent   = ax.get_tightbbox(renderer)
        extent   = extent.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(FIGURES_DIR / filename,
                    bbox_inches=extent.expanded(pad, pad),
                    dpi=150, facecolor="white")
        print(f"  Panel saved: {filename}")
    except Exception as e:
        fig.savefig(FIGURES_DIR / filename,
                    bbox_inches="tight", dpi=150, facecolor="white")
        print(f"  Panel saved (fallback): {filename} — {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE        = 42
CV_SPLITS           = 5
REPEATS             = 10
OUTLIER_PERCENTILE  = 99
FEATURE_SELECTION_N = 12

print("="*90)
print("IMMUNE PROFILING: BINARY AGE GROUP PREDICTION (Young vs Older)")
print("="*90)

# ============================================================================
# STEP 1: DATA LOADING & BINARY AGE LABELING
# ============================================================================
print("\n[1/7] LOADING DATA...")
df = pd.read_csv(DATA_DIR / "analysis_merged_subject_level.csv")

# Extract cytokine columns — exclude demographics and cell type columns
DEMO_COLS = ['SUBJECT_ID', 'AGE', 'AGEGR1N', 'AGEGR1C', 'RACEGRP',
             'SEXN', 'SEXC', 'FASFL', 'AGE_BINARY', 'Batch date']
cytokines = [c for c in df.columns
             if c not in DEMO_COLS
             and not c.startswith('CT_')]
print(f"  Cytokines detected: {cytokines}")

# Binary label: 0 = Young (18-39), 1 = Older (40-66)
df['AGE_BINARY'] = df['AGEGR1C'].apply(
    lambda x: 0 if x in ['18-29', '30-39'] else 1
)

print(f"Dataset       : {len(df)} samples x {len(cytokines)} cytokines")
print(f"Target        : AGE_BINARY (0=Young 18-39, 1=Older 40-66)")
print(f"Young (18-39) : {(df['AGE_BINARY']==0).sum()} donors "
      f"({100*(df['AGE_BINARY']==0).mean():.1f}%)")
print(f"Older (40-66) : {(df['AGE_BINARY']==1).sum()} donors "
      f"({100*(df['AGE_BINARY']==1).mean():.1f}%)")

y = df['AGE_BINARY'].values

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
# STEP 3: ROBUST SCALING + LOG TRANSFORM
# ============================================================================
print(f"\n[3/7] PREPROCESSING...")

X_log    = np.log1p(X_raw.values)
scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X_log)

print(f"✓ Log1p transform applied")
print(f"✓ Robust scaling applied")
print(f"Class distribution: Young={(y==0).sum()} Older={(y==1).sum()}")

# ============================================================================
# STEP 4: FEATURE SELECTION (Univariate ANOVA F-test)
# ============================================================================
print(f"\n[4/7] FEATURE SELECTION (top {FEATURE_SELECTION_N} cytokines by ANOVA)...")

selector          = SelectKBest(f_classif, k=FEATURE_SELECTION_N)
X_selected        = selector.fit_transform(X_scaled, y)
selected_features = np.array(cytokines)[selector.get_support()]

f_scores, p_values = f_classif(X_scaled, y)
anova_df = pd.DataFrame({
    'Cytokine':    cytokines,
    'F_Score':     f_scores.round(4),
    'P_Value':     p_values.round(6),
    'Selected':    selector.get_support(),
    'Significant': p_values < 0.05
}).sort_values('F_Score', ascending=False)

print(f"Selected features: {', '.join(selected_features)}")
print(f"\nTop 5 cytokines by F-score:")
print(anova_df.head(5)[['Cytokine','F_Score','P_Value','Significant']].to_string(index=False))
print(f"\nSignificant cytokines (p<0.05): "
      f"{anova_df[anova_df['Significant']]['Cytokine'].tolist()}")

# ============================================================================
# STEP 5: METHOD 1 — PCA + KNN
# ============================================================================
print(f"\n[5/7] METHOD 1: PCA + KNN...")

pca   = PCA(n_components=6)
X_pca = pca.fit_transform(X_selected)

print(f"  PCA: {X_pca.shape[1]} components explaining "
      f"{pca.explained_variance_ratio_.sum():.1%} variance")

knn    = KNeighborsClassifier(n_neighbors=7)
cv_rep = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=REPEATS,
                                  random_state=RANDOM_STATE)

cv_acc_knn = cross_val_score(knn, X_pca, y, cv=cv_rep, scoring='accuracy')
cv_f1_knn  = cross_val_score(knn, X_pca, y, cv=cv_rep, scoring='f1')
cv_auc_knn = cross_val_score(knn, X_pca, y, cv=cv_rep, scoring='roc_auc')

print(f"  Accuracy : {cv_acc_knn.mean():.4f} ± {cv_acc_knn.std():.4f}")
print(f"  F1 Score : {cv_f1_knn.mean():.4f} ± {cv_f1_knn.std():.4f}")
print(f"  ROC-AUC  : {cv_auc_knn.mean():.4f} ± {cv_auc_knn.std():.4f}")

# ============================================================================
# STEP 6: METHOD 2 — XGBOOST
# ============================================================================
print(f"\n[6/7] METHOD 2: XGBOOST...")

xgb = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_STATE,
    objective='binary:logistic',
    eval_metric='logloss'
)

cv5_xgb = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_acc_xgb = cross_val_score(xgb, X_selected, y, cv=cv5_xgb, scoring='accuracy')
cv_f1_xgb  = cross_val_score(xgb, X_selected, y, cv=cv5_xgb, scoring='f1')
cv_auc_xgb = cross_val_score(xgb, X_selected, y, cv=cv5_xgb, scoring='roc_auc')

print(f"  Accuracy : {cv_acc_xgb.mean():.4f} ± {cv_acc_xgb.std():.4f}")
print(f"  F1 Score : {cv_f1_xgb.mean():.4f} ± {cv_f1_xgb.std():.4f}")
print(f"  ROC-AUC  : {cv_auc_xgb.mean():.4f} ± {cv_auc_xgb.std():.4f}")
print(f"  XGBoost AUC fold scores: {cv_auc_xgb}")

xgb_fit = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=RANDOM_STATE,
    objective='binary:logistic',
    eval_metric='logloss'
)
xgb_fit.fit(X_selected, y)
feature_importance_xgb = pd.DataFrame({
    'Feature':    selected_features,
    'Importance': xgb_fit.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"  Top 5 features: "
      f"{', '.join(feature_importance_xgb.head(5)['Feature'].tolist())}")

# ============================================================================
# STEP 7: METHOD 3 — LOGISTIC REGRESSION
# ============================================================================
print(f"\n[7/7] METHOD 3: LOGISTIC REGRESSION (L2 regularized)...")

lr = LogisticRegression(
    penalty='l2', C=0.1, max_iter=1000,
    random_state=RANDOM_STATE, class_weight='balanced'
)

cv_acc_lr = cross_val_score(lr, X_selected, y, cv=cv_rep, scoring='accuracy')
cv_f1_lr  = cross_val_score(lr, X_selected, y, cv=cv_rep, scoring='f1')
cv_auc_lr = cross_val_score(lr, X_selected, y, cv=cv_rep, scoring='roc_auc')

print(f"  Accuracy : {cv_acc_lr.mean():.4f} ± {cv_acc_lr.std():.4f}")
print(f"  F1 Score : {cv_f1_lr.mean():.4f} ± {cv_f1_lr.std():.4f}")
print(f"  ROC-AUC  : {cv_auc_lr.mean():.4f} ± {cv_auc_lr.std():.4f}")

lr_fit = LogisticRegression(penalty='l2', C=0.1, max_iter=1000,
                             random_state=RANDOM_STATE, class_weight='balanced')
lr_fit.fit(X_selected, y)
lr_coef_df = pd.DataFrame({
    'Feature':     selected_features,
    'Coefficient': lr_fit.coef_[0].round(4),
    'Direction':   ['↑ Older' if c > 0 else '↑ Young'
                    for c in lr_fit.coef_[0]]
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\n  LR Coefficients (top 5):")
print(lr_coef_df.head(5).to_string(index=False))

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "="*90)
print("MODEL COMPARISON — BINARY AGE PREDICTION (Young vs Older)")
print("="*90)

comparison = pd.DataFrame({
    'Method':       ['PCA + KNN', 'XGBoost', 'Logistic Regression'],
    'Features':     [f'PCA ({X_pca.shape[1]} PCs)',
                     f'{FEATURE_SELECTION_N} selected',
                     f'{FEATURE_SELECTION_N} selected'],
    'Accuracy':     [cv_acc_knn.mean(), cv_acc_xgb.mean(), cv_acc_lr.mean()],
    'Accuracy Std': [cv_acc_knn.std(),  cv_acc_xgb.std(),  cv_acc_lr.std()],
    'F1 Score':     [cv_f1_knn.mean(),  cv_f1_xgb.mean(),  cv_f1_lr.mean()],
    'ROC-AUC':      [cv_auc_knn.mean(), cv_auc_xgb.mean(), cv_auc_lr.mean()],
})

print(comparison.to_string(index=False))

best_model_name = comparison.loc[comparison['Accuracy'].idxmax(), 'Method']
best_accuracy   = comparison['Accuracy'].max()
print(f"\n  → Best model: {best_model_name} ({best_accuracy:.4f} accuracy)")
print(f"  → Random baseline: 50.0% (binary classification)")
print()

# ============================================================================
# DETAILED EVALUATION — LOGISTIC REGRESSION
# ============================================================================
print("="*90)
print("DETAILED EVALUATION: LOGISTIC REGRESSION")
print("="*90)

cv5          = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
y_pred_cv    = cross_val_predict(lr_fit, X_selected, y, cv=cv5)
y_pred_proba = cross_val_predict(lr_fit, X_selected, y, cv=cv5,
                                  method='predict_proba')[:, 1]

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y, y_pred_cv)
print(f"                Pred Young   Pred Older")
print(f"  Actual Young :  {cm[0,0]:>4}         {cm[0,1]:>4}")
print(f"  Actual Older :  {cm[1,0]:>4}         {cm[1,1]:>4}")
print(f"\nClassification Report:")
print(classification_report(y, y_pred_cv,
                             target_names=['Young (18-39)', 'Older (40-66)']))

roc_auc_final = roc_auc_score(y, y_pred_proba)
print(f"ROC-AUC: {roc_auc_final:.4f}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
comparison.to_csv(OUTPUT_DIR / "results_binary_age_comparison.csv", index=False)
anova_df.to_csv(OUTPUT_DIR / "results_anova_age.csv", index=False)
feature_importance_xgb.to_csv(OUTPUT_DIR / "results_xgb_importance_age.csv", index=False)
lr_coef_df.to_csv(OUTPUT_DIR / "results_lr_coefficients_age.csv", index=False)

summary = pd.DataFrame({
    'Metric': ['Target', 'Classes', 'Sample Size',
               'Best Model', 'Best Accuracy', 'Best F1', 'Best ROC-AUC',
               'Random Baseline', 'Features Used'],
    'Value':  ['Binary Age (Young vs Older)',
               'Young=18-39 (n=' + str((y==0).sum()) + '), '
               'Older=40-66 (n=' + str((y==1).sum()) + ')',
               str(len(df)),
               best_model_name,
               f'{best_accuracy:.4f}',
               f'{comparison.loc[comparison["Accuracy"].idxmax(),"F1 Score"]:.4f}',
               f'{comparison.loc[comparison["Accuracy"].idxmax(),"ROC-AUC"]:.4f}',
               '50.0%',
               f'{FEATURE_SELECTION_N} cytokines (ANOVA selected)']
})
summary.to_csv(OUTPUT_DIR / "results_summary_binary_age.csv", index=False)
print("\n✓ All results saved")

# ============================================================================
# VISUALIZATION
# ============================================================================
fig = plt.figure(figsize=(20, 12))

# Panel 1: Model comparison accuracy
ax1 = plt.subplot(2, 3, 1)
methods  = ['KNN', 'XGBoost', 'LogReg']
means    = comparison['Accuracy'].values
stds     = comparison['Accuracy Std'].values
colors_c = ['#1f77b4', '#2ca02c', '#ff7f0e']
bars = ax1.bar(methods, means, yerr=stds, capsize=10,
               color=colors_c, alpha=0.7, edgecolor='black')
ax1.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Random (50%)')
ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Model Comparison: Accuracy\n(Young vs Older)',
              fontsize=12, fontweight='bold')
ax1.set_ylim([0.3, 0.85])
for bar, mean, std in zip(bars, means, stds):
    ax1.text(bar.get_x() + bar.get_width()/2, mean + std + 0.01,
             f'{mean:.3f}', ha='center', fontweight='bold', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Panel 2: ROC-AUC comparison
ax2 = plt.subplot(2, 3, 2)
aucs = comparison['ROC-AUC'].values
bars = ax2.bar(methods, aucs, color=colors_c, alpha=0.7, edgecolor='black')
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Random')
ax2.set_ylabel('ROC-AUC', fontsize=11, fontweight='bold')
ax2.set_title('Model Comparison: ROC-AUC\n(Young vs Older)',
              fontsize=12, fontweight='bold')
ax2.set_ylim([0.3, 0.85])
for bar, val in zip(bars, aucs):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
             f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Panel 3: Confusion matrix
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Young', 'Older'],
            yticklabels=['Young', 'Older'],
            cbar=False, ax=ax3,
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax3.set_title('Confusion Matrix\n(Logistic Regression, CV)',
              fontsize=12, fontweight='bold')
ax3.set_ylabel('True Age Group', fontsize=11)
ax3.set_xlabel('Predicted Age Group', fontsize=11)

# Panel 4: ROC curve
ax4 = plt.subplot(2, 3, 4)
fpr, tpr, _ = roc_curve(y, y_pred_proba)
ax4.plot(fpr, tpr, color='#ff7f0e', linewidth=2.5,
         label=f'Logistic Regression (AUC={roc_auc_final:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax4.set_xlabel('False Positive Rate', fontsize=11)
ax4.set_ylabel('True Positive Rate', fontsize=11)
ax4.set_title('ROC Curve — Binary Age Prediction',
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# Panel 5: LR coefficients
ax5 = plt.subplot(2, 3, 5)
coef_plot   = lr_coef_df.sort_values('Coefficient')
colors_coef = ['#E87D7D' if c < 0 else '#6BAED6'
               for c in coef_plot['Coefficient']]
ax5.barh(coef_plot['Feature'], coef_plot['Coefficient'],
         color=colors_coef, alpha=0.85, edgecolor='white')
ax5.axvline(0, color='black', linewidth=0.8)
ax5.set_xlabel('Coefficient (+ = predicts Older)', fontsize=10)
ax5.set_title('LR Coefficients\n(blue=↑Older, red=↑Young)',
              fontsize=12, fontweight='bold')
ax5.tick_params(labelsize=8)

# Panel 6: ANOVA F-scores
ax6 = plt.subplot(2, 3, 6)
anova_plot = anova_df.head(12).sort_values('F_Score')
colors_sig = ['#2ca02c' if s else '#aaaaaa'
              for s in anova_plot['Significant']]
ax6.barh(anova_plot['Cytokine'], anova_plot['F_Score'],
         color=colors_sig, alpha=0.85, edgecolor='white')
ax6.set_xlabel('ANOVA F-Score (higher = more discriminating)', fontsize=10)
ax6.set_title('Cytokine Age Discrimination\n(green = p<0.05)',
              fontsize=12, fontweight='bold')
ax6.tick_params(labelsize=8)

plt.suptitle('Binary Age Prediction: Young (18-39) vs Older (40-66)\n'
             f'Best Model: {best_model_name} | '
             f'Accuracy: {best_accuracy:.3f} | Random Baseline: 50%',
             fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "results_binary_age_prediction.png",
            dpi=150, bbox_inches='tight', facecolor='white')

# Save individual panels
fig.canvas.draw()
save_panel(fig, ax1, "panel_accuracy_comparison.png")
save_panel(fig, ax2, "panel_roc_auc_comparison.png")
save_panel(fig, ax3, "panel_confusion_matrix.png")
save_panel(fig, ax4, "panel_roc_curve.png")
save_panel(fig, ax5, "panel_lr_coefficients.png")
save_panel(fig, ax6, "panel_anova_discrimination.png")

# Save F1 comparison as separate figure
fig_f1, ax_f1 = plt.subplots(figsize=(6, 5))
f1s  = comparison['F1 Score'].values
bars = ax_f1.bar(methods, f1s, color=colors_c, alpha=0.7, edgecolor='black')
ax_f1.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Random')
ax_f1.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
ax_f1.set_title('Model Comparison: F1 Score\n(Young vs Older)',
                fontsize=12, fontweight='bold')
ax_f1.set_ylim([0.3, 0.85])
for bar, val in zip(bars, f1s):
    ax_f1.text(bar.get_x() + bar.get_width()/2, val + 0.01,
               f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)
ax_f1.legend(fontsize=9)
ax_f1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "panel_f1_comparison.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Panel saved: panel_f1_comparison.png")


print("✓ Figure saved: results_binary_age_prediction.png")
print(f"  Panels saved to: {FIGURES_DIR}")

print("\n" + "="*90)
print("PIPELINE COMPLETE — BINARY AGE PREDICTION")
print("="*90)
print(f"\n  Target       : Young (18-39) vs Older (40-66)")
print(f"  Best model   : {best_model_name}")
print(f"  Accuracy     : {best_accuracy:.4f}  (random=0.500)")
print(f"  Top cytokines: {', '.join(feature_importance_xgb.head(3)['Feature'].tolist())}")
print(f"  Outputs      : {OUTPUT_DIR}")
print(f"  Figures      : {FIGURES_DIR}")
print("="*90)