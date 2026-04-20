"""
================================================================================
Cytokine + Cell Type Covariate Analysis — XGBoost Style Pipeline
================================================================================
Target: Sex, Age Group, Race (as covariates / classification targets)
Features: 18 cytokines + 59 cell types from healthy adults
Method: OLS covariate analysis + XGBoost feature importance per demographic
Cross-validation: 10×5-fold Repeated Stratified K-Fold
Preprocessing: Outlier clipping (99th percentile) + Robust Scaling
================================================================================
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pathlib import Path

import statsmodels.formula.api as smf
import statsmodels.api as sm

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / "Data-Emma" / "analysis_merged_subject_level.csv"
OUTPUT_DIR  = BASE_DIR / "COVAR-XGBoost" / "covariate_outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_FEATURES_SELECT = 12    # ANOVA top-k features per classification target

# ============================================================================
# OUTPUT LOG — mirrors stdout to file (matches original pipeline pattern)
# ============================================================================
class Tee:
    def __init__(self, filepath):
        self._file   = open(filepath, 'w', encoding='utf-8')
        self._stdout = sys.stdout
    def write(self, data):
        self._stdout.write(data); self._file.write(data)
    def flush(self):
        self._stdout.flush(); self._file.flush()
    def close(self):
        self._file.close()
        sys.stdout = self._stdout

sys.stdout = Tee(OUTPUT_DIR / "covariate_pipeline_output.txt")

# ============================================================================
# 1. LOAD & VALIDATE DATA
# ============================================================================
print("=" * 80)
print("1. LOADING DATA")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded: {df.shape[0]} samples × {df.shape[1]} columns")
print(f"  Source: {DATA_PATH}")

DEMO_COLS = ['SUBJECT_ID', 'AGE', 'AGEGR1N', 'AGEGR1C', 'RACEGRP',
             'SEXN', 'SEXC', 'FASFL', 'AGE_BINARY', 'Batch date']
CYTO_COLS = [c for c in df.columns
             if c not in DEMO_COLS and not c.startswith('CT_')]
CELL_COLS = [c for c in df.columns if c.startswith('CT_')]
ALL_FEAT  = CYTO_COLS + CELL_COLS

print(f"  Cytokines  : {len(CYTO_COLS)}")
print(f"  Cell types : {len(CELL_COLS)}")
print(f"  Total feat : {len(ALL_FEAT)}")
print(f"  Sex        : {df['SEXC'].value_counts().to_dict()}")
print(f"  Age groups : {df['AGEGR1C'].value_counts().to_dict()}")
print(f"  Race       : {df['RACEGRP'].value_counts().to_dict()}")

# ============================================================================
# 2. PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("2. PREPROCESSING")
print("=" * 80)

# Log1p cytokines, keep cell types as proportions
X_raw = df[ALL_FEAT].copy().values.astype(float)
for i in range(len(CYTO_COLS)):
    X_raw[:, i] = np.log1p(X_raw[:, i])

# Median impute cell types
for j in range(len(CYTO_COLS), len(ALL_FEAT)):
    col = X_raw[:, j]
    col[np.isnan(col)] = np.nanmedian(col)
    X_raw[:, j] = col

# Outlier clipping — 99th percentile (matches original pipeline)
X_clipped = X_raw.copy()
for j in range(X_clipped.shape[1]):
    p99 = np.nanpercentile(X_clipped[:, j], 99)
    X_clipped[:, j] = np.minimum(X_clipped[:, j], p99)
print(f"✓ Outlier clipping applied (99th percentile)")

# Robust scaling
scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X_clipped)
print(f"✓ RobustScaler applied")

# Build model dataframe for OLS
df_model = df[["SEXC", "AGEGR1C", "RACEGRP"]].copy()
for i, feat in enumerate(ALL_FEAT):
    safe = (feat.replace("-", "_").replace(" ", "_")
               .replace("+", "pos").replace("(", "").replace(")", ""))
    df_model[safe] = X_raw[:, i]

ALL_SAFE = [
    (feat.replace("-", "_").replace(" ", "_")
         .replace("+", "pos").replace("(", "").replace(")", ""))
    for feat in ALL_FEAT
]
print(f"✓ Model dataframe built: {df_model.shape}")

# ============================================================================
# 3. OLS COVARIATE ANALYSIS — ALL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("3. OLS COVARIATE ANALYSIS (Sex + Age + Race)")
print("=" * 80)

def run_ols(feat_names, safe_names, modality):
    rows = []
    for feat, safe in zip(feat_names, safe_names):
        formula = f"{safe} ~ C(SEXC) + C(AGEGR1C) + C(RACEGRP)"
        try:
            model       = smf.ols(formula, data=df_model).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            total_ss    = anova_table["sum_sq"].sum()

            def ss(key):
                return anova_table.loc[key, "sum_sq"] \
                       if key in anova_table.index else 0

            pr2_sex  = ss("C(SEXC)")   / total_ss if total_ss > 0 else 0
            pr2_age  = ss("C(AGEGR1C)")/ total_ss if total_ss > 0 else 0
            pr2_race = ss("C(RACEGRP)")/ total_ss if total_ss > 0 else 0

            pval_sex  = model.pvalues.get("C(SEXC)[T.Male]", np.nan)
            pval_age  = min([v for k, v in model.pvalues.items()
                             if "AGEGR1C" in k] or [np.nan])
            pval_race = min([v for k, v in model.pvalues.items()
                             if "RACEGRP" in k] or [np.nan])

            rows.append({
                "Feature":         feat,
                "Modality":        modality,
                "R2_full":         round(model.rsquared, 4),
                "Partial_R2_Sex":  round(pr2_sex,  4),
                "Partial_R2_Age":  round(pr2_age,  4),
                "Partial_R2_Race": round(pr2_race, 4),
                "Pval_Sex":        round(float(pval_sex),  6) if not np.isnan(pval_sex)  else np.nan,
                "Pval_Age":        round(float(pval_age),  6) if not np.isnan(pval_age)  else np.nan,
                "Pval_Race":       round(float(pval_race), 6) if not np.isnan(pval_race) else np.nan,
                "Sig_Sex":         bool(pval_sex  < 0.05) if not np.isnan(pval_sex)  else False,
                "Sig_Age":         bool(pval_age  < 0.05) if not np.isnan(pval_age)  else False,
                "Sig_Race":        bool(pval_race < 0.05) if not np.isnan(pval_race) else False,
            })
        except Exception as e:
            print(f"  WARNING: OLS failed for {feat}: {e}")
    return pd.DataFrame(rows).sort_values("R2_full", ascending=False).reset_index(drop=True)

cyto_safe = [f.replace("-","_").replace(" ","_") for f in CYTO_COLS]
cell_safe  = [f.replace("-","_").replace(" ","_").replace("+","pos")
               .replace("(","").replace(")","") for f in CELL_COLS]

ols_cyto = run_ols(CYTO_COLS, cyto_safe, "Cytokine")
ols_cell = run_ols(CELL_COLS, cell_safe, "Cell Type")
ols_all  = pd.concat([ols_cyto, ols_cell], ignore_index=True)

ols_cyto.to_csv(OUTPUT_DIR / "ols_cytokine_results.csv", index=False)
ols_cell.to_csv(OUTPUT_DIR / "ols_celltype_results.csv", index=False)
ols_all.to_csv(OUTPUT_DIR  / "ols_all_results.csv",      index=False)

print(f"\n  CYTOKINES — Mean R²: {ols_cyto['R2_full'].mean():.4f}  "
      f"Max R²: {ols_cyto['R2_full'].max():.4f}")
print(f"  Sig for sex  : {ols_cyto[ols_cyto['Sig_Sex']]['Feature'].tolist()}")
print(f"  Sig for age  : {ols_cyto[ols_cyto['Sig_Age']]['Feature'].tolist()}")
print(f"  Sig for race : {ols_cyto[ols_cyto['Sig_Race']]['Feature'].tolist()}")
print(f"\n  CELL TYPES  — Mean R²: {ols_cell['R2_full'].mean():.4f}  "
      f"Max R²: {ols_cell['R2_full'].max():.4f}")
print(f"  Sig for sex  (n): {ols_cell['Sig_Sex'].sum()}")
print(f"  Sig for age  (n): {ols_cell['Sig_Age'].sum()}")
print(f"  Sig for race (n): {ols_cell['Sig_Race'].sum()}")
print(f"\n✓ Saved: ols_cytokine_results.csv, ols_celltype_results.csv, ols_all_results.csv")

# ============================================================================
# 4. XGBOOST CLASSIFICATION — SEX / AGE / RACE
# ============================================================================
print("\n" + "=" * 80)
print("4. XGBOOST + MODEL COMPARISON (Sex, Age, Race)")
print("=" * 80)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

targets = {
    "Sex":  (df["SEXC"].map({"Male": 1, "Female": 0}).values,
             ["Female", "Male"],   "SEXC"),
    "Age":  (LabelEncoder().fit_transform(df["AGEGR1C"].values),
             sorted(df["AGEGR1C"].unique()), "AGEGR1C"),
    "Race": (LabelEncoder().fit_transform(df["RACEGRP"].values),
             sorted(df["RACEGRP"].unique()),  "RACEGRP"),
}

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42,
                                               solver='lbfgs'),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=5,
                                                   random_state=42, n_jobs=-1),
    'XGBoost':             XGBClassifier(n_estimators=100, max_depth=5,
                                         learning_rate=0.1, random_state=42,
                                         eval_metric='logloss', verbosity=0),
}

all_cv_results   = {}
all_feat_imp     = {}
all_cm           = {}
all_roc          = {}

for target_name, (y_target, class_names, col_name) in targets.items():
    print(f"\n  ── TARGET: {target_name} ──")
    n_classes  = len(np.unique(y_target))
    is_binary  = n_classes == 2
    auc_avg    = "binary" if is_binary else "macro"
    f1_avg     = "binary" if is_binary else "weighted"
    n_select   = min(N_FEATURES_SELECT, X_scaled.shape[1])

    # ANOVA feature selection per target
    selector   = SelectKBest(f_classif, k=n_select)
    X_sel      = selector.fit_transform(X_scaled, y_target)
    sel_mask   = selector.get_support()
    sel_feats  = [ALL_FEAT[i] for i in range(len(ALL_FEAT)) if sel_mask[i]]
    print(f"  Selected features: {sel_feats}")

    cv_results = {}
    for model_name, model in models.items():
        accs, f1s, aucs = [], [], []
        for train_idx, test_idx in cv.split(X_sel, y_target):
            Xtr, Xte = X_sel[train_idx], X_sel[test_idx]
            ytr, yte = y_target[train_idx], y_target[test_idx]
            model.fit(Xtr, ytr)
            ypred  = model.predict(Xte)
            yproba = model.predict_proba(Xte)
            accs.append(accuracy_score(yte, ypred))
            f1s.append(f1_score(yte, ypred, average=f1_avg, zero_division=0))
            try:
                if is_binary:
                    aucs.append(roc_auc_score(yte, yproba[:, 1]))
                else:
                    aucs.append(roc_auc_score(yte, yproba,
                                              multi_class="ovr", average="weighted"))
            except Exception:
                aucs.append(0.0)

        cv_results[model_name] = {
            "Accuracy":      f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
            "F1-Score":      f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
            "ROC-AUC":       f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
            "Accuracy_Mean": np.mean(accs),
            "F1_Mean":       np.mean(f1s),
            "AUC_Mean":      np.mean(aucs),
        }
        print(f"    {model_name:<22} Acc={np.mean(accs):.4f}  "
              f"F1={np.mean(f1s):.4f}  AUC={np.mean(aucs):.4f}")

    all_cv_results[target_name] = cv_results

    # Save model comparison
    pd.DataFrame(cv_results).T.to_csv(
        OUTPUT_DIR / f"model_comparison_{target_name.lower()}.csv")

    # Train final XGBoost on full data
    xgb_final = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                               random_state=42, eval_metric='logloss', verbosity=0)
    xgb_final.fit(X_sel, y_target)

    feat_imp = pd.DataFrame({
        "Feature":    sel_feats,
        "Importance": xgb_final.feature_importances_,
        "Target":     target_name
    }).sort_values("Importance", ascending=False)
    feat_imp.to_csv(OUTPUT_DIR / f"xgb_feature_importance_{target_name.lower()}.csv",
                    index=False)
    all_feat_imp[target_name] = feat_imp

    ypred_full  = xgb_final.predict(X_sel)
    yproba_full = xgb_final.predict_proba(X_sel)
    all_cm[target_name]  = (confusion_matrix(y_target, ypred_full), class_names)
    all_roc[target_name] = (y_target, yproba_full, is_binary, class_names)

    print(f"  ✓ Saved: model_comparison_{target_name.lower()}.csv, "
          f"xgb_feature_importance_{target_name.lower()}.csv")

# ============================================================================
# 5. COMPREHENSIVE VISUALIZATION — matches original pipeline layout
# ============================================================================
print("\n" + "=" * 80)
print("5. GENERATING VISUALIZATIONS")
print("=" * 80)

COLOR_XGB  = '#FF6B6B'
COLOR_OTHER = '#4ECDC4'
COLOR_BAR   = '#4A7FC1'

# ── Figure 1: Model comparison across all three targets ──────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle("XGBoost + Model Comparison — Sex, Age, Race Classification\n"
              "10×5-fold Repeated Stratified CV",
              fontsize=13, fontweight="bold")

for ax, (tname, cv_res) in zip(axes1, all_cv_results.items()):
    sorted_models = sorted(cv_res.items(), key=lambda x: x[1]["Accuracy_Mean"],
                           reverse=True)
    mnames = [m[0] for m in sorted_models]
    accs   = [m[1]["Accuracy_Mean"] for m in sorted_models]
    colors = [COLOR_XGB if m == "XGBoost" else COLOR_OTHER for m in mnames]
    ax.barh(mnames, accs, color=colors)
    ax.set_xlabel("CV Accuracy")
    ax.set_title(f"{tname} Prediction", fontsize=11, fontweight="bold")
    ax.set_xlim(0.2, 0.85)
    for i, v in enumerate(accs):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(1 / len(all_cm[tname][1]), color="red", linestyle="--",
               alpha=0.5, linewidth=1, label="Random baseline")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_model_comparison_all_targets.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: fig1_model_comparison_all_targets.png")

# ── Figure 2: XGBoost feature importance — sex, age, race side by side ───────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 7))
fig2.suptitle("XGBoost Feature Importance — Sex, Age, Race\n"
              "Top 12 ANOVA-selected features",
              fontsize=13, fontweight="bold")

for ax, tname in zip(axes2, ["Sex", "Age", "Race"]):
    fi   = all_feat_imp[tname].head(10)
    clrs = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(fi)))
    ax.barh(fi["Feature"], fi["Importance"], color=clrs)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top Features — {tname}", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_feature_importance_all_targets.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: fig2_feature_importance_all_targets.png")

# ── Figure 3: OLS partial R² heatmaps — cytokines and top-20 cell types ──────
fig3 = plt.figure(figsize=(16, 14))
fig3.suptitle("OLS Covariate Analysis — Partial R² Heatmaps\n"
              "* = significant (p<0.05)",
              fontsize=13, fontweight="bold", y=0.98)
gs3 = fig3.add_gridspec(1, 2, wspace=0.35)

def r2_heatmap(ax, ols_df, title, max_rows=None):
    data = ols_df.head(max_rows) if max_rows else ols_df
    mat  = data[["Partial_R2_Sex", "Partial_R2_Age", "Partial_R2_Race"]].values
    vmax = max(mat.max(), 0.01)
    im   = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Sex", "Age", "Race"], fontsize=11, fontweight="bold")
    labels = [f.replace("CT_","")[:35] if f.startswith("CT_") else f
              for f in data["Feature"].tolist()]
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Partial R²", fraction=0.04, pad=0.04)
    for i in range(len(data)):
        for j, sc in enumerate(["Sig_Sex", "Sig_Age", "Sig_Race"]):
            val = mat[i, j]
            sig = data.iloc[i][sc]
            ax.text(j, i, f"{val:.3f}" + ("*" if sig else ""),
                    ha="center", va="center", fontsize=7,
                    color="white" if val > vmax * 0.6 else "black")

r2_heatmap(fig3.add_subplot(gs3[0, 0]), ols_cyto,
           "Cytokines — Partial R²\n(sorted by total R²)")
r2_heatmap(fig3.add_subplot(gs3[0, 1]), ols_cell,
           "Top 20 Cell Types — Partial R²\n(sorted by total R²)", max_rows=20)

plt.savefig(FIGURES_DIR / "fig3_ols_partial_r2_heatmaps.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: fig3_ols_partial_r2_heatmaps.png")

# ── Figure 4: Confusion matrices for all three targets ───────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
fig4.suptitle("XGBoost Confusion Matrices — Sex, Age, Race\n"
              "(trained on full dataset)",
              fontsize=13, fontweight="bold")

for ax, tname in zip(axes4, ["Sex", "Age", "Race"]):
    cm_data, class_names = all_cm[tname]
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"fontsize": 9})
    ax.set_title(f"{tname} Confusion Matrix", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_confusion_matrices.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: fig4_confusion_matrices.png")

# ── Figure 5: OLS stacked partial R² bars — cytokines + top 20 cell types ────
fig5, axes5 = plt.subplots(1, 2, figsize=(18, 8))
fig5.suptitle("Partial R² Breakdown — Sex vs Age vs Race\n"
              "How much variance in each feature is explained by each covariate",
              fontsize=13, fontweight="bold")

for ax, ols_df, title, max_rows in [
    (axes5[0], ols_cyto, "Cytokines", None),
    (axes5[1], ols_cell, "Top 20 Cell Types", 20)
]:
    data  = ols_df.head(max_rows).sort_values("R2_full", ascending=True) \
            if max_rows else ols_df.sort_values("R2_full", ascending=True)
    labels = [f.replace("CT_","").replace("_"," ")[:35]
              if f.startswith("CT_") else f for f in data["Feature"]]
    y_pos  = np.arange(len(data))
    ax.barh(y_pos, data["Partial_R2_Sex"],  color="#6BAED6", alpha=0.85, label="Sex")
    ax.barh(y_pos, data["Partial_R2_Age"],
            left=data["Partial_R2_Sex"],    color="#F4A460", alpha=0.85, label="Age")
    ax.barh(y_pos, data["Partial_R2_Race"],
            left=data["Partial_R2_Sex"] + data["Partial_R2_Age"],
            color="#5BAD72", alpha=0.85, label="Race")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Partial R²", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig5_partial_r2_stacked.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: fig5_partial_r2_stacked.png")

# ── Figure 6: Combined summary heatmap — XGBoost accuracy across targets ─────
fig6, ax6 = plt.subplots(figsize=(10, 5))
summary_mat = []
summary_rows_labels = []
for tname, cv_res in all_cv_results.items():
    row = [cv_res[m]["Accuracy_Mean"] for m in ["Logistic Regression",
                                                  "Random Forest", "XGBoost"]]
    summary_mat.append(row)
    summary_rows_labels.append(tname)

summary_mat = np.array(summary_mat)
sns.heatmap(summary_mat, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=["Logistic Regression", "Random Forest", "XGBoost"],
            yticklabels=summary_rows_labels,
            cbar_kws={"label": "CV Accuracy"}, ax=ax6,
            annot_kws={"fontsize": 11, "fontweight": "bold"})
ax6.set_title("Model × Target Accuracy Summary\n"
              "(10×5-fold CV — cytokines + cell types)",
              fontsize=12, fontweight="bold")
ax6.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig6_accuracy_summary_heatmap.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: fig6_accuracy_summary_heatmap.png")

# ── Individual panels — matches original pipeline SAVE_INDIVIDUAL_PANELS block
print("\nSaving individual panels...")

for tname in ["Sex", "Age", "Race"]:
    cv_res      = all_cv_results[tname]
    sorted_mods = sorted(cv_res.items(), key=lambda x: x[1]["Accuracy_Mean"],
                         reverse=True)
    mnames = [m[0] for m in sorted_mods]
    accs   = [m[1]["Accuracy_Mean"] for m in sorted_mods]

    # Panel: model accuracy
    fig_p, ax_p = plt.subplots(figsize=(8, 5))
    colors = [COLOR_XGB if m == "XGBoost" else COLOR_OTHER for m in mnames]
    ax_p.barh(mnames, accs, color=colors)
    ax_p.set_xlabel("CV Accuracy")
    ax_p.set_title(f"Model Comparison: {tname} Prediction",
                   fontsize=12, fontweight="bold")
    ax_p.set_xlim(0.2, 0.85)
    for i, v in enumerate(accs):
        ax_p.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"panel_{tname.lower()}_model_accuracy.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # Panel: feature importance
    fi = all_feat_imp[tname].head(10)
    fig_f, ax_f = plt.subplots(figsize=(8, 5))
    clrs = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(fi)))
    ax_f.barh(fi["Feature"], fi["Importance"], color=clrs)
    ax_f.set_xlabel("Importance Score")
    ax_f.set_title(f"Top 10 Features — {tname} (XGBoost)",
                   fontsize=12, fontweight="bold")
    ax_f.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"panel_{tname.lower()}_feature_importance.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # Panel: confusion matrix
    cm_data, class_names = all_cm[tname]
    fig_c, ax_c = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_c,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"fontsize": 9})
    ax_c.set_title(f"XGBoost Confusion Matrix — {tname}",
                   fontsize=12, fontweight="bold")
    ax_c.set_ylabel("True Label")
    ax_c.set_xlabel("Predicted Label")
    plt.setp(ax_c.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"panel_{tname.lower()}_confusion_matrix.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  ✓ Saved individual panels for {tname}")

# ============================================================================
# 6. SAVE RESULTS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("6. SAVING RESULTS SUMMARY")
print("=" * 80)

summary_rows = []
for tname, cv_res in all_cv_results.items():
    best_model = max(cv_res.items(), key=lambda x: x[1]["Accuracy_Mean"])
    xgb_res    = cv_res["XGBoost"]
    summary_rows.append({
        "Target":              tname,
        "XGBoost_Accuracy":    xgb_res["Accuracy"],
        "XGBoost_F1":          xgb_res["F1-Score"],
        "XGBoost_ROC_AUC":     xgb_res["ROC-AUC"],
        "Best_Model":          best_model[0],
        "Best_Model_Accuracy": f"{best_model[1]['Accuracy_Mean']:.4f}",
        "OLS_Mean_R2_Cyto":    f"{ols_cyto['R2_full'].mean():.4f}",
        "OLS_Mean_R2_Cell":    f"{ols_cell['R2_full'].mean():.4f}",
        "OLS_Sig_Sex_Cyto":    ols_cyto['Sig_Sex'].sum(),
        "OLS_Sig_Age_Cell":    ols_cell['Sig_Age'].sum(),
        "OLS_Sig_Race_Cell":   ols_cell['Sig_Race'].sum(),
    })

pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "covariate_results_summary.csv",
                                   index=False)
print("✓ Saved: covariate_results_summary.csv")

# ============================================================================
# 7. FINAL PRINT SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY — COVARIATE ANALYSIS + CLASSIFICATION RESULTS")
print("=" * 80)

print(f"\n  Dataset: {len(df)} samples  |  "
      f"{len(CYTO_COLS)} cytokines + {len(CELL_COLS)} cell types")

print(f"\n  OLS Covariate Analysis:")
print(f"    Cytokines  mean R² : {ols_cyto['R2_full'].mean():.4f}  "
      f"(max {ols_cyto['R2_full'].max():.4f})")
print(f"    Cell types mean R² : {ols_cell['R2_full'].mean():.4f}  "
      f"(max {ols_cell['R2_full'].max():.4f})")
print(f"    Cell/cytokine R² ratio : "
      f"{ols_cell['R2_full'].mean() / max(ols_cyto['R2_full'].mean(), 1e-6):.1f}x")

print(f"\n  XGBoost CV Accuracy (10×5-fold):")
for tname, cv_res in all_cv_results.items():
    xgb = cv_res["XGBoost"]
    best = max(cv_res.items(), key=lambda x: x[1]["Accuracy_Mean"])
    print(f"    {tname:<6} : XGBoost={xgb['Accuracy_Mean']:.4f}  "
          f"Best={best[0]} ({best[1]['Accuracy_Mean']:.4f})")

print(f"\n  Top cytokines for sex (XGBoost):")
for _, row in all_feat_imp["Sex"].head(3).iterrows():
    print(f"    {row['Feature']:<12} importance={row['Importance']:.3f}")

print(f"\n  All outputs saved to: {OUTPUT_DIR}")
print("=" * 80)
print("✓ Pipeline completed successfully!")
print("=" * 80)

sys.stdout.close()