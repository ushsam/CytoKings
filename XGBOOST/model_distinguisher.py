"""
Model Distinguisher — Beyond Accuracy
======================================
Purpose: Since all models reach ~59-60% CV accuracy, this script provides
         additional analyses to meaningfully differentiate them:

  1. McNemar's Test          — do models make statistically different errors?
  2. Calibration Curves      — do predicted probabilities match actual rates?
  3. Logistic Regression Coefficients — directionality (which sex is higher?)
  4. Permutation Feature Importance  — model-agnostic, no normality assumption

Run from: /Users/home/CytoKings/xgboost/
Command:  python3 model_distinguisher.py

Outputs:
  model_distinguisher_results.csv   — McNemar p-values between model pairs
  logreg_coefficients.csv           — LR coefficients with direction labels
  permutation_importance.csv        — permutation-based feature ranking
  plot_calibration_curves.png       — calibration curves for all models
  plot_logreg_coefficients.png      — coefficient bar chart with sex direction
  plot_permutation_importance.png   — permutation importance bar chart
"""

import time
pipeline_start = time.time()

import matplotlib
matplotlib.use('Agg')

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from scipy.stats import chi2
from pathlib import Path

print("=" * 70)
print("MODEL DISTINGUISHER — Beyond Accuracy")
print("=" * 70)

# ============================================================================
# PATHS
# ============================================================================
try:
    BASE_PATH = Path(__file__).resolve().parent   # XGBOOST folder
except NameError:
    BASE_PATH = Path.cwd()
 
REPO_PATH   = BASE_PATH.parent                    # CytoKings repo root
DATA_PATH   = REPO_PATH / "Data" / "analysis_merged_subject_level.csv"
OUTPUT_DIR  = BASE_PATH / "Outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
 
print("Repo root  :", REPO_PATH)
print("Data file  :", DATA_PATH)
print("Data exists:", DATA_PATH.exists())
print("Output dir :", OUTPUT_DIR)
print ("Figures dir:", FIGURES_DIR)

# ============================================================================
# DATA LOADING & PREPROCESSING (same as main script)
# ============================================================================

df = pd.read_csv(DATA_PATH)

cytokines = [
    'IFN-gamma', 'IL-12p70', 'IL-13', 'IL-1beta', 'IL-2', 'IL-4', 'IL-5',
    'IL-6', 'TNF-alpha', 'GM-CSF', 'IL-18', 'IL-10', 'IL-17A', 'IL-21',
    'IL-22', 'IL-23', 'IL-27', 'IL-9'
]

X_raw = df[cytokines].copy()
y     = (df['SEXC'] == 'Male').astype(int).values

# Same preprocessing as main script
for col in X_raw.columns:
    X_raw[col] = np.minimum(X_raw[col], X_raw[col].quantile(0.99))

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)

# Feature selection — fit on full dataset (same as main script)
selector        = SelectKBest(f_classif, k=12)
X_selected      = selector.fit_transform(X_scaled, y)
mask            = selector.get_support()
selected_cytos  = [cytokines[i] for i in range(len(cytokines)) if mask[i]]

print(f"✓ Data: {X_raw.shape[0]} samples, {len(selected_cytos)} selected features")
print(f"  Selected cytokines: {selected_cytos}\n")

# CV strategy: standard 5-fold (cross_val_predict needs non-repeated CV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================================
# MODELS — using same best params as main script
# ============================================================================
# Load best params from grid search output if available
try:
    table2      = pd.read_csv(OUTPUT_DIR / 'results_table2_hyperparameters.csv')
    xgb_rows = table2[table2['Model'] == 'XGBoost']
    best_n_est  = int(xgb_rows[xgb_rows['Parameter'] == 'n_estimators']['Best Value'].values[0])
    best_depth  = int(xgb_rows[xgb_rows['Parameter'] == 'max_depth']['Best Value'].values[0])
    best_lr     = float(xgb_rows[xgb_rows['Parameter'] == 'learning_rate']['Best Value'].values[0])
    lr_rows     = table2[table2['Model'] == 'LogReg']
    best_C      = float(lr_rows[lr_rows['Parameter'] == 'C']['Best Value'].values[0])
    print(f"✓ Loaded best params from grid search: XGB(depth={best_depth}, lr={best_lr}), LogReg(C={best_C})")
except Exception:
    best_n_est, best_depth, best_lr, best_C = 100, 5, 0.05, 0.01
    print("⚠ Using default params (run main script first for grid-searched params)")

models = {
    'Logistic Regression': LogisticRegression(C=best_C, max_iter=1000,
                                              random_state=42, solver='lbfgs'),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=5,
                                                  random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                      random_state=42),
    'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=best_n_est, max_depth=best_depth,
                                         learning_rate=best_lr, random_state=42,
                                         eval_metric='logloss', verbosity=0),
}

# ============================================================================
# COLLECT CV PREDICTIONS FOR ALL MODELS
# ============================================================================
print("Collecting cross-validated predictions for all models...")

all_preds  = {}   # binary predictions per model
all_probas = {}   # probability scores per model

for name, model in models.items():
    preds  = cross_val_predict(model, X_selected, y, cv=cv, method='predict')
    probas = cross_val_predict(model, X_selected, y, cv=cv,
                               method='predict_proba')[:, 1]
    all_preds[name]  = preds
    all_probas[name] = probas
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probas)
    print(f"  {name:<22} Acc={acc:.3f}  AUC={auc:.3f}")

# ============================================================================
# 1. McNEMAR'S TEST — do models make different errors?
# ============================================================================
# Tests whether two classifiers make statistically different mistakes.
# H0: both classifiers make the same types of errors.
# Low p-value → they predict differently (one is systematically better
# on different samples, not just on average).
# Reference: McNemar (1947), commonly used in ML model comparison.
print("\n" + "=" * 70)
print("1. McNEMAR'S TEST — error differences between model pairs")
print("=" * 70)

model_names   = list(models.keys())
mcnemar_rows  = []

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        name_a = model_names[i]
        name_b = model_names[j]
        preds_a = all_preds[name_a]
        preds_b = all_preds[name_b]

        # Contingency table for McNemar's test
        # b = A wrong, B right  |  c = A right, B wrong
        b = np.sum((preds_a != y) & (preds_b == y))
        c = np.sum((preds_a == y) & (preds_b != y))

        # Chi-squared statistic with continuity correction (Yates)
        if (b + c) == 0:
            chi2_stat, p_val = 0.0, 1.0
        else:
            chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
            p_val     = 1 - chi2.cdf(chi2_stat, df=1)

        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01
              else ("*" if p_val < 0.05 else "ns"))

        mcnemar_rows.append({
            'Model_A': name_a, 'Model_B': name_b,
            'b (A wrong, B right)': b, 'c (A right, B wrong)': c,
            'chi2': round(chi2_stat, 3), 'p_value': round(p_val, 4),
            'Significant': sig,
        })
        print(f"  {name_a:<22} vs {name_b:<22} | b={b}, c={c} | "
              f"p={p_val:.4f} {sig}")

mcnemar_df = pd.DataFrame(mcnemar_rows)
mcnemar_df.to_csv(OUTPUT_DIR / 'model_distinguisher_results.csv', index=False)
print("\n✓ Saved: model_distinguisher_results.csv")
print("\nInterpretation: p < 0.05 means the two models make DIFFERENT errors.")
print("'ns' means models are statistically interchangeable on this dataset.")

# ============================================================================
# 2. CALIBRATION CURVES — do probabilities match actual class rates?
# ============================================================================
# A well-calibrated model saying "70% chance of Male" should be right 70% of
# the time. LogReg is typically best calibrated; XGBoost often overconfident.
print("\n" + "=" * 70)
print("2. CALIBRATION CURVES")
print("=" * 70)

fig_cal, ax_cal = plt.subplots(figsize=(8, 6))
ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')

colors = ['#CC3333', '#3366CC', '#339933', '#FF9900', '#993399']
for (name, probas), color in zip(all_probas.items(), colors):
    fraction_pos, mean_pred = calibration_curve(y, probas, n_bins=5)
    brier = brier_score_loss(y, probas)
    ax_cal.plot(mean_pred, fraction_pos, marker='o', linewidth=1.8,
                color=color, label=f'{name} (Brier={brier:.3f})')

ax_cal.set_xlabel('Mean Predicted Probability', fontsize=11)
ax_cal.set_ylabel('Fraction of Positives (Male)', fontsize=11)
ax_cal.set_title('Calibration Curves — All Models\n'
                 '(Lower Brier score = better calibrated)', fontsize=11,
                 fontweight='bold')
ax_cal.legend(fontsize=8, loc='upper left')
ax_cal.grid(alpha=0.3)
ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'plot_calibration_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Saved: plot_calibration_curves.png")
print("  Brier scores (lower = better calibrated):")
for name, probas in all_probas.items():
    print(f"    {name:<22} {brier_score_loss(y, probas):.4f}")

# ============================================================================
# 3. LOGISTIC REGRESSION COEFFICIENTS — directionality
# ============================================================================
# LR coefficients show: positive = feature pushes toward Male (class 1)
#                       negative = feature pushes toward Female (class 0)
# This answers the key biological question: which sex has higher cytokine levels?
# Note: uses FULL dataset for interpretability (not CV) — same as SHAP.
print("\n" + "=" * 70)
print("3. LOGISTIC REGRESSION COEFFICIENTS — sex directionality")
print("=" * 70)

lr_final = LogisticRegression(C=best_C, max_iter=1000,
                               random_state=42, solver='lbfgs')
lr_final.fit(X_selected, y)

coef_df = pd.DataFrame({
    'Cytokine':    selected_cytos,
    'Coefficient': lr_final.coef_[0].round(4),
}).sort_values('Coefficient')

# Directionality: positive coef → high value predicts Male → higher in Males
coef_df['Higher_In'] = coef_df['Coefficient'].apply(
    lambda c: 'Male' if c > 0 else 'Female'
)
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
coef_df.to_csv(OUTPUT_DIR /'logreg_coefficients.csv', index=False)

print("\n  Coefficients (positive = higher in Males, negative = higher in Females):")
for _, row in coef_df.iterrows():
    bar = '█' * int(abs(row['Coefficient']) / coef_df['Abs_Coef'].max() * 15)
    direction = f"→ {row['Higher_In']:6s}"
    print(f"  {row['Cytokine']:<12} {row['Coefficient']:+.4f}  {direction}  {bar}")

print("\n✓ Saved: logreg_coefficients.csv")

# Coefficient bar chart
fig_coef, ax_coef = plt.subplots(figsize=(8, 6))
bar_colors = ['#6699CC' if c > 0 else '#FF9999'
              for c in coef_df['Coefficient']]
ax_coef.barh(coef_df['Cytokine'], coef_df['Coefficient'], color=bar_colors,
             edgecolor='white')
ax_coef.axvline(0, color='black', linewidth=1.2)
ax_coef.set_xlabel('Logistic Regression Coefficient', fontsize=11)
ax_coef.set_title('Logistic Regression Coefficients\n'
                  'Blue = higher in Males  |  Red = higher in Females',
                  fontsize=11, fontweight='bold')
ax_coef.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR/ 'plot_logreg_coefficients.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Saved: plot_logreg_coefficients.png")

# ============================================================================
# 4. PERMUTATION FEATURE IMPORTANCE — model-agnostic, no normality assumption
# ============================================================================
# Unlike ANOVA (parametric) or SHAP (model-specific), permutation importance:
#   - Works on any model
#   - Makes no distributional assumption
#   - Measures actual drop in accuracy when a feature is randomly shuffled
#   - If shuffling IL-21 drops accuracy by 5%, IL-21 carries 5% of the signal
print("\n" + "=" * 70)
print("4. PERMUTATION FEATURE IMPORTANCE — model-agnostic")
print("=" * 70)

xgb_perm = XGBClassifier(n_estimators=best_n_est, max_depth=best_depth,
                          learning_rate=best_lr, random_state=42,
                          eval_metric='logloss', verbosity=0)
xgb_perm.fit(X_selected, y)

perm_result = permutation_importance(
    xgb_perm, X_selected, y,
    n_repeats=50,        # shuffle each feature 50 times for stable estimate
    random_state=42,
    scoring='accuracy'
)

perm_df = pd.DataFrame({
    'Cytokine':        selected_cytos,
    'Importance_Mean': perm_result.importances_mean.round(4),
    'Importance_Std':  perm_result.importances_std.round(4),
}).sort_values('Importance_Mean', ascending=False)

perm_df.to_csv( OUTPUT_DIR / 'permutation_importance.csv', index=False)

print("\n  Top cytokines by permutation importance:")
for _, row in perm_df.head(8).iterrows():
    bar = '█' * max(1, int(row['Importance_Mean'] /
                            perm_df['Importance_Mean'].max() * 20))
    print(f"  {row['Cytokine']:<12} {row['Importance_Mean']:+.4f} ± "
          f"{row['Importance_Std']:.4f}  {bar}")

print("\n✓ Saved: permutation_importance.csv")

# Permutation importance bar chart with error bars
fig_perm, ax_perm = plt.subplots(figsize=(8, 6))
sorted_df = perm_df.sort_values('Importance_Mean')
ax_perm.barh(sorted_df['Cytokine'], sorted_df['Importance_Mean'],
             xerr=sorted_df['Importance_Std'],
             color='#5577AA', alpha=0.85, capsize=3,
             error_kw={'linewidth': 1, 'capthick': 1})
ax_perm.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax_perm.set_xlabel('Mean Accuracy Drop When Feature Shuffled', fontsize=11)
ax_perm.set_title('Permutation Feature Importance (XGBoost)\n'
                  'n=50 repeats — no normality assumption',
                  fontsize=11, fontweight='bold')
ax_perm.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'plot_permutation_importance.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Saved: plot_permutation_importance.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nMcNemar's test: if p > 0.05 for all pairs → models make the same")
print("  errors → accuracy difference is not meaningful → all models are")
print("  equally limited by the biological signal, not model architecture.")
print("\nCalibration: if LogReg has lowest Brier score → it's most useful")
print("  for probability estimates (not just binary prediction).")
print("\nLogReg coefficients: the directionality table directly answers")
print("  which cytokines are higher in males vs females — use this in")
print("  the Discussion section for biological interpretation.")
print("\nPermutation importance: compare top features vs ANOVA and SHAP.")
print("  Overlap across 3 methods = more trustworthy biological conclusion.")
print("=" * 70)


pipeline_end     = time.time()
elapsed          = pipeline_end - pipeline_start
mins, secs       = divmod(int(elapsed), 60)
print(f"\nTotal runtime: {mins}m {secs}s")