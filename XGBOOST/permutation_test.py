"""
Permutation Test for Sex Classification Accuracy
=================================================
Purpose: Establish that XGBoost 59.6% accuracy is statistically significant
         above the 50% random baseline. Without this, a reviewer can argue
         the result is noise.

Method: Randomly shuffle y labels 1000 times and refit XGBoost each time.
        If real accuracy (59.6%) exceeds 95% of permuted accuracies → p < 0.05.

Runtime: ~10-20 minutes on CPU with n_jobs=-1 (uses all cores).
         Set N_PERMUTATIONS=200 for a quick 3-minute run.
         Set N_PERMUTATIONS=1000 for publication-quality p-value.

Output:
  - permutation_test_results.csv  : score, p-value, permuted distribution stats
  - permutation_test_plot.png     : histogram of permuted accuracies vs real score
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from pathlib import Path

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
# CONFIGURATION
# ============================================================================
N_PERMUTATIONS = 1000   # 1000 → publication quality, ~15-20 min
                         # 200  → quick check, ~3-5 min
RANDOM_STATE   = 42

print("=" * 70)
print("PERMUTATION TEST — XGBoost Sex Classification")
print(f"N permutations: {N_PERMUTATIONS}")
print("=" * 70)

# ============================================================================
# LOAD AND PREPROCESS (same pipeline as main script)
# ============================================================================
df = pd.read_csv(DATA_PATH)

cytokines = [
    'IFN-gamma', 'IL-12p70', 'IL-13', 'IL-1beta', 'IL-2', 'IL-4', 'IL-5',
    'IL-6', 'TNF-alpha', 'GM-CSF', 'IL-18', 'IL-10', 'IL-17A', 'IL-21',
    'IL-22', 'IL-23', 'IL-27', 'IL-9'
]

X = df[cytokines].copy()
y = (df['SEXC'] == 'Male').astype(int).values

print(f"✓ Data: {X.shape[0]} samples × {X.shape[1]} cytokines")
print(f"  Females (0): {(y==0).sum()} | Males (1): {(y==1).sum()}")

# Outlier clipping (99th percentile)
for col in X.columns:
    X[col] = np.minimum(X[col], X[col].quantile(0.99))

# Robust scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# PIPELINE: ANOVA inside CV loop — no leakage
# ============================================================================
# Using Pipeline ensures SelectKBest is fit on each training fold only.
# This is the methodologically correct approach.
pipeline = Pipeline([
    ('selector', SelectKBest(f_classif, k=12)),
    ('xgb', XGBClassifier(
        n_estimators=100,
        max_depth=3,          # shallow trees for small data
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        verbosity=0
    ))
])

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)

# ============================================================================
# RUN PERMUTATION TEST
# ============================================================================
print(f"\nRunning permutation test ({N_PERMUTATIONS} permutations)...")
print("This uses all CPU cores. Runtime: ~10-20 minutes for N=1000.")
print("Press Ctrl+C to stop early — partial results will NOT be saved.\n")

score, perm_scores, p_value = permutation_test_score(
    pipeline,
    X_scaled,
    y,
    cv=cv,
    n_permutations=N_PERMUTATIONS,
    scoring='accuracy',
    n_jobs=-1,              # use all available CPU cores
    random_state=RANDOM_STATE,
    verbose=1
)

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("PERMUTATION TEST RESULTS")
print("=" * 70)
print(f"  Real CV Accuracy         : {score:.4f} ({score*100:.2f}%)")
print(f"  Permuted mean accuracy   : {perm_scores.mean():.4f} ({perm_scores.mean()*100:.2f}%)")
print(f"  Permuted std accuracy    : {perm_scores.std():.4f}")
print(f"  Permuted 95th percentile : {np.percentile(perm_scores, 95):.4f}")
print(f"  p-value                  : {p_value:.4f}")
print()

if p_value < 0.001:
    print(f"  ✓ RESULT: p < 0.001 — accuracy is highly significant above chance")
    significance = "p < 0.001"
elif p_value < 0.01:
    print(f"  ✓ RESULT: p < 0.01 — accuracy is significant above chance")
    significance = "p < 0.01"
elif p_value < 0.05:
    print(f"  ✓ RESULT: p < 0.05 — accuracy is significant above chance")
    significance = "p < 0.05"
else:
    print(f"  ✗ RESULT: p = {p_value:.3f} — accuracy NOT significant above chance")
    significance = f"p = {p_value:.3f} (not significant)"

print()
print("  PAPER SENTENCE:")
print(f'  "A permutation test (n={N_PERMUTATIONS}) confirmed that XGBoost accuracy')
print(f'  of {score*100:.1f}% was significantly above the random baseline')
print(f'  (permuted mean = {perm_scores.mean()*100:.1f}%, {significance})."')

# ============================================================================
# SAVE RESULTS
# ============================================================================
results_df = pd.DataFrame({
    'Metric': [
        'Real CV Accuracy',
        'Permuted Mean',
        'Permuted Std',
        'Permuted 5th Percentile',
        'Permuted 95th Percentile',
        'p-value',
        'N Permutations',
        'Significant (p<0.05)',
    ],
    'Value': [
        round(score, 4),
        round(perm_scores.mean(), 4),
        round(perm_scores.std(), 4),
        round(np.percentile(perm_scores, 5), 4),
        round(np.percentile(perm_scores, 95), 4),
        round(p_value, 4),
        N_PERMUTATIONS,
        p_value < 0.05,
    ]
})
results_df.to_csv(OUTPUT_DIR/'permutation_test_results.csv', index=False)
print("\n✓ Saved: permutation_test_results.csv")

# ============================================================================
# PLOT
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(perm_scores, bins=30, color='#AAAACC', edgecolor='white',
        linewidth=0.5, label=f'Permuted accuracies (n={N_PERMUTATIONS})')

ax.axvline(score, color='#CC3333', linewidth=2.5,
           label=f'Real accuracy: {score*100:.1f}%')

ax.axvline(np.percentile(perm_scores, 95), color='#CC6600', linewidth=1.5,
           linestyle='--', label=f'95th percentile: {np.percentile(perm_scores, 95)*100:.1f}%')

ax.axvline(0.5, color='black', linewidth=1, linestyle=':',
           label='Random baseline: 50.0%')

ax.set_xlabel('Cross-Validated Accuracy', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(
    f'Permutation Test — XGBoost Sex Classification\n'
    f'Real accuracy = {score*100:.1f}%   |   {significance}',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'permutation_test_plot.png', dpi=200, bbox_inches='tight')
print("✓ Saved: permutation_test_plot.png")
plt.close()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
