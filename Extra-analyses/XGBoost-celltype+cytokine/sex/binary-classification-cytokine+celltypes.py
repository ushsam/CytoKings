"""
XGBoost + Multiple Models Pipeline — Cytokines + Cell Types
Target: SEXC (Biological Sex - Binary Classification: 1=Male, 0=Female)
Features: Combined cytokine + cell type PCA (6 features)
          and full 77-feature standardized matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      RepeatedStratifiedKFold, cross_val_predict)
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from sklearn.feature_selection import f_classif, SelectKBest
from XGBOOST.xgboost import XGBClassifier
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR    = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR    = BASE_DIR / "Data-Emma"
PCA_DIR = BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokine+celltype"
OUTPUT_DIR  = BASE_DIR / "XGBoost" / "celltype+cytokine" / "sex" / "Outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE        = 42
CV_SPLITS           = 5
REPEATS             = 10
OUTLIER_PERCENTILE  = 99
FEATURE_SELECTION_N = 12

print("="*90)
print("ML PIPELINE — SEX CLASSIFICATION: CYTOKINES + CELL TYPES")
print("="*90)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/8] LOADING DATA...")

df = pd.read_csv(DATA_DIR / "analysis_merged_subject_level.csv")

CYTO_COLS = [
    'IFN-gamma', 'IL-12p70', 'IL-13', 'IL-1beta', 'IL-2', 'IL-4', 'IL-5',
    'IL-6', 'TNF-alpha', 'GM-CSF', 'IL-18', 'IL-10', 'IL-17A', 'IL-21',
    'IL-22', 'IL-23', 'IL-27', 'IL-9'
]
CELL_COLS = [c for c in df.columns if c.startswith("CT_")]

y = (df["SEXC"] == "Male").astype(int).values

# Load pre-computed PCA outputs from full features pipeline
X_combined_pca = np.load(PCA_DIR / "X_combined_pca.npy")    # 6 PCs (3 cyto + 3 cell)
X_cyto_scaled  = np.load(PCA_DIR / "X_cytokines_processed.npy")
X_cell_scaled  = np.load(PCA_DIR / "X_cell_processed.npy")

print(f"  Subjects          : {len(df)}")
print(f"  Cytokine features : {len(CYTO_COLS)}")
print(f"  Cell type features: {len(CELL_COLS)}")
print(f"  Combined PCA shape: {X_combined_pca.shape}  (3 cyto PCs + 3 cell PCs)")
print(f"  Males             : {y.sum()}  Females: {(y==0).sum()}")

# ============================================================================
# STEP 2: OUTLIER HANDLING + RAW FEATURE MATRIX
# ============================================================================
print(f"\n[2/8] PREPARING FULL FEATURE MATRIX...")

# Clip outliers on raw data
X_cyto_raw = df[CYTO_COLS].copy()
X_cell_raw = df[CELL_COLS].copy()

for col in CYTO_COLS:
    threshold = X_cyto_raw[col].quantile(OUTLIER_PERCENTILE / 100)
    X_cyto_raw[col] = X_cyto_raw[col].clip(upper=threshold)

# Log1p + robust scale cytokines
X_cyto_log    = np.log1p(X_cyto_raw.values)
scaler_cyto   = RobustScaler()
X_cyto_sc     = scaler_cyto.fit_transform(X_cyto_log)

# Robust scale cell types (no log — proportions)
scaler_cell   = RobustScaler()
X_cell_sc     = scaler_cell.fit_transform(X_cell_raw.values)

# Full concatenated matrix (77 features)
X_full = np.hstack([X_cyto_sc, X_cell_sc])

# Impute any remaining NaNs with column median
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_full  = imputer.fit_transform(X_full)

print(f"  NaNs after imputation: {np.isnan(X_full).sum()}")

print(f"  Full feature matrix: {X_full.shape}  (log1p cytokines + cell proportions)")
print(f"  Combined PCA matrix: {X_combined_pca.shape}  (low-dim, leakage-free)")

# ============================================================================
# STEP 3: FEATURE SELECTION ON FULL MATRIX
# ============================================================================
print(f"\n[3/8] FEATURE SELECTION (top {FEATURE_SELECTION_N} from full 77 features)...")

all_feature_names = CYTO_COLS + CELL_COLS
selector          = SelectKBest(f_classif, k=FEATURE_SELECTION_N)
X_selected        = selector.fit_transform(X_full, y)
selected_features = np.array(all_feature_names)[selector.get_support()]

f_scores, p_values = f_classif(X_full, y)
anova_df = pd.DataFrame({
    'Feature':     all_feature_names,
    'F_Score':     f_scores.round(4),
    'P_Value':     p_values.round(6),
    'Selected':    selector.get_support(),
    'Significant': p_values < 0.05,
    'Modality':    ['Cytokine'] * len(CYTO_COLS) + ['Cell Type'] * len(CELL_COLS)
}).sort_values('F_Score', ascending=False)

print(f"  Selected: {', '.join(selected_features)}")
sig = anova_df[anova_df['Significant']]['Feature'].tolist()
print(f"  Significant (p<0.05): {sig}")
anova_df.to_csv(OUTPUT_DIR / "anova_feature_selection.csv", index=False)

# ============================================================================
# STEP 4: SEX COMPARISON — CYTOKINES + CELL TYPES
# ============================================================================
print(f"\n[4/8] SEX-BASED FEATURE COMPARISON...")

male_idx   = np.where(y == 1)[0]
female_idx = np.where(y == 0)[0]

comparison_rows = []
for i, name in enumerate(all_feature_names):
    t, p = stats.ttest_ind(X_full[male_idx, i], X_full[female_idx, i],
                            equal_var=False)
    comparison_rows.append({
        'Feature':     name,
        'Male_Mean':   X_full[male_idx, i].mean().round(4),
        'Female_Mean': X_full[female_idx, i].mean().round(4),
        'Diff (M-F)':  (X_full[male_idx, i].mean() - X_full[female_idx, i].mean()).round(4),
        'T_Stat':      round(t, 4),
        'P_Value':     round(p, 6),
        'Significant': p < 0.05,
        'Direction':   '↑ Male' if t > 0 else '↑ Female',
        'Modality':    'Cytokine' if name in CYTO_COLS else 'Cell Type'
    })

sex_comparison_df = pd.DataFrame(comparison_rows).sort_values(
    'P_Value').reset_index(drop=True)
sex_comparison_df.to_csv(OUTPUT_DIR / "sex_feature_comparison.csv", index=False)

sig_features = sex_comparison_df[sex_comparison_df['Significant']]
print(f"  Significant features (p<0.05): {len(sig_features)} / {len(all_feature_names)}")
print(f"  Cytokines significant: "
      f"{len(sig_features[sig_features['Modality']=='Cytokine'])}")
print(f"  Cell types significant: "
      f"{len(sig_features[sig_features['Modality']=='Cell Type'])}")

# ============================================================================
# STEP 5: DEFINE MODELS
# ============================================================================
print(f"\n[5/8] DEFINING MODELS...")

cv_rep = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=REPEATS,
                                  random_state=RANDOM_STATE)
cv5    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective='binary:logistic',
        random_state=RANDOM_STATE, eval_metric='logloss'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        penalty='l2', C=0.1, max_iter=1000,
        random_state=RANDOM_STATE, class_weight='balanced'
    ),
    'SVM': SVC(
        kernel='rbf', C=1.0, gamma='scale',
        probability=True, random_state=RANDOM_STATE
    ),
}

# ============================================================================
# STEP 6: CROSS-VALIDATION ON THREE FEATURE SETS
# ============================================================================
print(f"\n[6/8] CROSS-VALIDATION (3 feature sets x 4 models)...")
print(f"  Feature sets:")
print(f"    1. Combined PCA  : {X_combined_pca.shape[1]} features")
print(f"    2. ANOVA selected: {X_selected.shape[1]} features")
print(f"    3. Full matrix   : {X_full.shape[1]} features")
print()

feature_sets = {
    'Combined PCA (6)':      X_combined_pca,
    f'ANOVA Top {FEATURE_SELECTION_N}': X_selected,
    'Full (77)':             X_full,
}

all_results = []
for feat_name, X_feat in feature_sets.items():
    print(f"  ── {feat_name} ──")
    for model_name, model in models.items():
        acc  = cross_val_score(model, X_feat, y, cv=cv_rep, scoring='accuracy')
        f1   = cross_val_score(model, X_feat, y, cv=cv_rep, scoring='f1')
        aucs = cross_val_score(model, X_feat, y, cv=cv_rep, scoring='roc_auc')
        print(f"    {model_name:<22} Acc={acc.mean():.4f}±{acc.std():.4f}  "
              f"F1={f1.mean():.4f}  AUC={aucs.mean():.4f}")
        all_results.append({
            'Feature Set':  feat_name,
            'Model':        model_name,
            'Accuracy':     round(acc.mean(), 4),
            'Acc Std':      round(acc.std(),  4),
            'F1 Score':     round(f1.mean(),  4),
            'ROC-AUC':      round(aucs.mean(),4),
        })
    print()

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / "model_comparison_all.csv", index=False)

# Best overall
best_row = results_df.loc[results_df['Accuracy'].idxmax()]
print(f"  Best overall: {best_row['Model']} on {best_row['Feature Set']} "
      f"({best_row['Accuracy']:.4f})")

# ============================================================================
# STEP 7: DETAILED EVALUATION — BEST MODEL ON COMBINED PCA
# ============================================================================
print(f"\n[7/8] DETAILED EVALUATION — XGBoost on Combined PCA...")

xgb_best = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    objective='binary:logistic',
    random_state=RANDOM_STATE, eval_metric='logloss'
)
xgb_best.fit(X_combined_pca, y)

y_pred       = cross_val_predict(xgb_best, X_combined_pca, y, cv=cv5)
y_pred_proba = cross_val_predict(xgb_best, X_combined_pca, y, cv=cv5,
                                  method='predict_proba')[:, 1]

accuracy  = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall    = recall_score(y, y_pred)
f1        = f1_score(y, y_pred)
roc_auc   = roc_auc_score(y, y_pred_proba)
cm        = confusion_matrix(y, y_pred)

print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"                  Pred Female  Pred Male")
print(f"  Actual Female :   {cm[0,0]:>4}         {cm[0,1]:>4}")
print(f"  Actual Male   :   {cm[1,0]:>4}         {cm[1,1]:>4}")
print(f"\n  Classification Report:")
print(classification_report(y, y_pred, target_names=['Female', 'Male']))

# XGBoost feature importance on ANOVA-selected features
xgb_imp = XGBClassifier(
    n_estimators=100, max_depth=3, objective='binary:logistic',
    random_state=RANDOM_STATE, eval_metric='logloss'
)
xgb_imp.fit(X_selected, y)
feature_importance_df = pd.DataFrame({
    'Feature':    selected_features,
    'Importance': xgb_imp.feature_importances_,
    'Modality':   ['Cytokine' if f in CYTO_COLS else 'Cell Type'
                   for f in selected_features]
}).sort_values('Importance', ascending=False)
feature_importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

# Save detailed results
detail_df = pd.DataFrame([{
    'Feature Set':    'Combined PCA (6)',
    'Model':          'XGBoost',
    'Accuracy':       round(accuracy, 4),
    'Precision':      round(precision, 4),
    'Recall':         round(recall, 4),
    'F1 Score':       round(f1, 4),
    'ROC-AUC':        round(roc_auc, 4),
}])
detail_df.to_csv(OUTPUT_DIR / "detailed_results_xgb.csv", index=False)
joblib.dump(xgb_best, OUTPUT_DIR / "xgboost_full_features.pkl")

# ============================================================================
# STEP 8: VISUALIZATION
# ============================================================================
print(f"\n[8/8] GENERATING VISUALIZATIONS...")

fig = plt.figure(figsize=(22, 14))
fig.suptitle(
    "Sex Classification — Cytokines + Cell Types\n"
    "XGBoost + Multiple Models Comparison",
    fontsize=14, fontweight='bold'
)

colors_models = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
model_names   = list(models.keys())

# Panel 1: Accuracy by model and feature set
ax1 = plt.subplot(2, 3, 1)
feat_labels = list(feature_sets.keys())
x      = np.arange(len(model_names))
width  = 0.25
colors_fs = ['#4A7FC1', '#E87D7D', '#5BAD72']
for i, (feat_name, color) in enumerate(zip(feat_labels, colors_fs)):
    subset = results_df[results_df['Feature Set'] == feat_name]
    accs   = subset['Accuracy'].values
    stds   = subset['Acc Std'].values
    ax1.bar(x + i*width, accs, width, label=feat_name,
            color=color, alpha=0.8, edgecolor='white',
            yerr=stds, capsize=4)
ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
ax1.set_xticks(x + width)
ax1.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8)
ax1.set_ylabel('CV Accuracy', fontsize=10, fontweight='bold')
ax1.set_title('Model × Feature Set Accuracy', fontsize=11, fontweight='bold')
ax1.set_ylim([0.4, 0.85])
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Panel 2: ROC-AUC comparison
ax2 = plt.subplot(2, 3, 2)
for i, (feat_name, color) in enumerate(zip(feat_labels, colors_fs)):
    subset = results_df[results_df['Feature Set'] == feat_name]
    ax2.bar(x + i*width, subset['ROC-AUC'].values, width,
            label=feat_name, color=color, alpha=0.8, edgecolor='white')
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
ax2.set_xticks(x + width)
ax2.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8)
ax2.set_ylabel('ROC-AUC', fontsize=10, fontweight='bold')
ax2.set_title('Model × Feature Set ROC-AUC', fontsize=11, fontweight='bold')
ax2.set_ylim([0.4, 0.85])
ax2.legend(fontsize=7, loc='upper right')
ax2.grid(axis='y', alpha=0.3)

# Panel 3: Confusion matrix (XGBoost, combined PCA)
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Female', 'Male'],
            yticklabels=['Female', 'Male'],
            cbar=False, ax=ax3,
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax3.set_title(f'Confusion Matrix\nXGBoost + Combined PCA\n'
              f'Acc={accuracy:.3f}  F1={f1:.3f}',
              fontsize=11, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=10)
ax3.set_xlabel('Predicted Label', fontsize=10)

# Panel 4: ROC curve
ax4 = plt.subplot(2, 3, 4)
fpr, tpr, _ = roc_curve(y, y_pred_proba)
ax4.plot(fpr, tpr, color='#4A7FC1', linewidth=2.5,
         label=f'XGBoost + Combined PCA (AUC={roc_auc:.3f})')
ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax4.set_xlabel('False Positive Rate', fontsize=10)
ax4.set_ylabel('True Positive Rate', fontsize=10)
ax4.set_title('ROC Curve', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# Panel 5: Feature importance (ANOVA-selected, colored by modality)
ax5 = plt.subplot(2, 3, 5)
top_imp    = feature_importance_df.head(12)
colors_imp = ['#4A7FC1' if m == 'Cytokine' else '#5BAD72'
              for m in top_imp['Modality']]
ax5.barh(top_imp['Feature'], top_imp['Importance'],
         color=colors_imp, alpha=0.85, edgecolor='white')
ax5.set_xlabel('XGBoost Importance', fontsize=10)
ax5.set_title('Feature Importance\n(blue=Cytokine, green=Cell Type)',
              fontsize=11, fontweight='bold')
ax5.tick_params(labelsize=8)
from matplotlib.patches import Patch
ax5.legend(handles=[Patch(color='#4A7FC1', label='Cytokine'),
                    Patch(color='#5BAD72', label='Cell Type')],
           fontsize=8)

# Panel 6: Top significant features by p-value
ax6 = plt.subplot(2, 3, 6)
top_sig = sex_comparison_df.head(15)
colors_sig = ['#4A7FC1' if m == 'Cytokine' else '#5BAD72'
              for m in top_sig['Modality']]
ax6.barh(top_sig['Feature'].str.replace('CT_', '').str.replace('_', ' '),
         -np.log10(top_sig['P_Value'] + 1e-10),
         color=colors_sig, alpha=0.85, edgecolor='white')
ax6.axvline(-np.log10(0.05), color='red', linestyle='--',
            alpha=0.7, label='p=0.05')
ax6.set_xlabel('-log10(p-value)', fontsize=10)
ax6.set_title('Sex Differentiation by Feature\n(higher = more significant)',
              fontsize=11, fontweight='bold')
ax6.tick_params(labelsize=7)
ax6.legend(fontsize=8)
from matplotlib.patches import Patch
ax6.legend(handles=[Patch(color='#4A7FC1', label='Cytokine'),
                    Patch(color='#5BAD72', label='Cell Type'),
                    plt.Line2D([0], [0], color='red', linestyle='--',
                               label='p=0.05')],
           fontsize=7)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "sex_full_features_results.png",
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Figure saved: sex_full_features_results.png")

print("\n" + "="*90)
print("PIPELINE COMPLETE — SEX CLASSIFICATION: CYTOKINES + CELL TYPES")
print("="*90)
print(f"  Best overall     : {best_row['Model']} on {best_row['Feature Set']}")
print(f"  Best accuracy    : {best_row['Accuracy']:.4f}")
print(f"  XGBoost+CombPCA  : Acc={accuracy:.4f}  F1={f1:.4f}  AUC={roc_auc:.4f}")
print(f"  Sig features     : {len(sig_features)} / {len(all_feature_names)}")
print(f"  Outputs          : {OUTPUT_DIR}")
print(f"  Figures          : {FIGURES_DIR}")
print("="*90)