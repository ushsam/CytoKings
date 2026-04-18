"""
================================================================================
Cytokines + Cell Types PCA + KNN Sex Classification Pipeline
================================================================================
Approach:
  - Run two SEPARATE PCAs: one on cytokines, one on cell types
  - Concatenate top N PCs from each → combined matrix
  - This prevents cell types (59 features) from dominating by count
Outputs → PCA+KNN-Emma/Outputs/cytokine+celltype/
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ================================================================================
# PATHS
# ================================================================================
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "Data-Emma"
OUTPUT_DIR = BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokine+celltype"
FIGURES_DIR    = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# SECTION 1: DATA LOADING
# ================================================================================
# ================================================================================
# SECTION 1: DATA LOADING
# ================================================================================
df = pd.read_csv(DATA_DIR / "analysis_merged_subject_level.csv")

CYTO_COLS = [
    "IFN-gamma", "IL-12p70", "IL-13", "IL-1beta", "IL-2", "IL-4", "IL-5",
    "IL-6", "TNF-alpha", "GM-CSF", "IL-18", "IL-10", "IL-17A", "IL-21",
    "IL-22", "IL-23", "IL-27", "IL-9"
]
CELL_COLS = [c for c in df.columns if c.startswith("CT_")]

if len(CELL_COLS) == 0:
    raise ValueError(
        "No CT_ columns found in analysis_merged_subject_level.csv. "
        "Run the full dataprep script first to generate the merged file with cell types."
    )

X_cyto    = df[CYTO_COLS].values.astype(float)

# impute missing cell type values with column median before PCA
X_cell_df = df[CELL_COLS].copy()
for col in CELL_COLS:
    X_cell_df[col] = X_cell_df[col].fillna(X_cell_df[col].median())
X_cell = X_cell_df.values.astype(float)

y         = (df["SEXC"] == "Male").astype(int).values
n_samples = len(df)

print("=" * 65)
print("SECTION 1: DATASET SUMMARY  (cytokines + cell types)")
print("=" * 65)
print(f"  Total subjects    : {n_samples}")
print(f"  Cytokine features : {len(CYTO_COLS)}")
print(f"  Cell type features: {len(CELL_COLS)}")
print(f"  Total features    : {len(CYTO_COLS) + len(CELL_COLS)}")
print(f"  Males             : {y.sum()}  ({100*y.sum()/n_samples:.1f}%)")
print(f"  Females           : {(y==0).sum()}  ({100*(y==0).sum()/n_samples:.1f}%)")
print(f"  Missing (cyto)    : {np.isnan(X_cyto).sum()}")
print(f"  Missing (cell)    : {np.isnan(X_cell).sum()}")
print()

# ================================================================================
# SECTION 2: UTILITY FUNCTIONS
# ================================================================================

def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def pca_fit(X_train_scaled, n_components):
    train_mean = X_train_scaled.mean(axis=0)
    X_centered = X_train_scaled - train_mean
    X_centered = np.nan_to_num(X_centered, nan=0.0, posinf=0.0, neginf=0.0)
    X_centered[:, X_centered.std(axis=0) == 0] = 0
    _, S, Vt        = np.linalg.svd(X_centered, full_matrices=False)
    all_eigenvalues = (S ** 2) / (len(X_train_scaled) - 1)
    total_variance  = all_eigenvalues.sum()
    components      = Vt[:n_components]
    var_ratio       = all_eigenvalues[:n_components] / total_variance
    return components, var_ratio, train_mean


def pca_transform(X_scaled, train_mean, components):
    return np.dot(X_scaled - train_mean, components.T)


def knn_predict(X_train_pca, y_train, X_test_pca, k):
    preds = []
    for x in X_test_pca:
        distances    = np.linalg.norm(X_train_pca - x, axis=1)
        neighbor_idx = np.argsort(distances)[:k]
        votes        = y_train[neighbor_idx]
        preds.append(np.bincount(votes, minlength=2).argmax())
    return np.array(preds)


def compute_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])


def classification_metrics(cm):
    TN, FP = cm[0]
    FN, TP = cm[1]
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
    bal_acc     = (sensitivity + specificity) / 2
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0)
    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    return {
        "Accuracy":          round(accuracy,    4),
        "Balanced Accuracy": round(bal_acc,     4),
        "Sensitivity":       round(sensitivity, 4),
        "Specificity":       round(specificity, 4),
        "Precision":         round(precision,   4),
        "F1 Score":          round(f1,          4),
    }


def stratified_kfold_split(y, n_splits=5, seed=42):
    np.random.seed(seed)
    idx0 = np.where(y == 0)[0];  np.random.shuffle(idx0)
    idx1 = np.where(y == 1)[0];  np.random.shuffle(idx1)
    folds0 = np.array_split(idx0, n_splits)
    folds1 = np.array_split(idx1, n_splits)
    splits = []
    for i in range(n_splits):
        test_idx  = np.concatenate([folds0[i], folds1[i]])
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        splits.append((train_idx, test_idx))
    return splits

def save_panel(fig, ax, filename, pad=1.15):
    """Save a single panel from the figure by extracting its bounding box."""
    fig.canvas.draw()
    extent = ax.get_tightbbox(fig.canvas.get_renderer())
    extent = extent.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(FIGURES_DIR / filename, bbox_inches=extent.expanded(pad, pad),
                dpi=150, facecolor="white")
    print(f"  Panel saved: {filename}")


# ================================================================================
# SECTION 3: GLOBAL PCA — run separately on each modality
# ================================================================================
def global_pca(X_raw, label):
    mean     = X_raw.mean(axis=0)
    std      = X_raw.std(axis=0);  std[std == 0] = 1
    X_sc     = (X_raw - mean) / std
    X_c      = X_sc - X_sc.mean(axis=0)
    X_c      = np.nan_to_num(X_c, nan=0.0, posinf=0.0, neginf=0.0)
    X_c[:, X_c.std(axis=0) == 0] = 0
    _, S, _  = np.linalg.svd(X_c, full_matrices=False)
    eig      = (S ** 2) / (len(X_sc) - 1)
    vr       = eig / eig.sum()
    cv       = np.cumsum(vr)
    print(f"\n  {label} PCA:")
    for i in range(min(5, len(vr))):
        print(f"    PC{i+1}: {vr[i]*100:.1f}%  (cumul: {cv[i]*100:.1f}%)")
    print(f"    PCs for 80%: {int(np.searchsorted(cv, 0.80)) + 1}")
    print(f"    PCs for 90%: {int(np.searchsorted(cv, 0.90)) + 1}")
    return vr, cv

print("=" * 65)
print("SECTION 3: GLOBAL PCA — INTRINSIC DIMENSIONALITY")
print("=" * 65)
vr_cyto_global, cv_cyto_global = global_pca(X_cyto, "Cytokine")
vr_cell_global, cv_cell_global = global_pca(X_cell, "Cell Type")
print()

# ================================================================================
# SECTION 4: CV SETUP
# ================================================================================
N_SPLITS    = 5
PCA_OPTIONS = list(range(2, 11))
K_OPTIONS   = list(range(3, 8))
folds       = stratified_kfold_split(y, N_SPLITS)

print("=" * 65)
print("SECTION 4: CROSS-VALIDATION SETUP")
print("=" * 65)
print(f"  Strategy          : Stratified {N_SPLITS}-Fold CV")
print(f"  Approach          : Separate PCA per modality, concatenate top PCs")
print(f"  KNN range         : K = {K_OPTIONS[0]}–{K_OPTIONS[-1]}")
print(f"  PCA range/modality: {PCA_OPTIONS[0]}–{PCA_OPTIONS[-1]}")
print(f"  Total combos      : {len(PCA_OPTIONS) * len(K_OPTIONS)}")
print()

# ================================================================================
# SECTION 5: GRID SEARCH
# ================================================================================
print("=" * 65)
print("SECTION 5: GRID SEARCH")
print("=" * 65)

results = {}
for n_pca in PCA_OPTIONS:
    for k_neighbors in K_OPTIONS:
        fold_accs            = []
        fold_preds           = []
        fold_loadings_cyto   = []
        fold_loadings_cell   = []
        fold_var_ratios_cyto = []
        fold_var_ratios_cell = []
        fold_projections     = []

        for train_idx, test_idx in folds:
            Xc_tr, Xc_te    = X_cyto[train_idx], X_cyto[test_idx]
            Xl_tr, Xl_te    = X_cell[train_idx], X_cell[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            Xc_tr_sc, Xc_te_sc, _, _ = standardize(Xc_tr, Xc_te)
            Xl_tr_sc, Xl_te_sc, _, _ = standardize(Xl_tr, Xl_te)

            comp_c, vr_c, mean_c = pca_fit(Xc_tr_sc, n_pca)
            comp_l, vr_l, mean_l = pca_fit(Xl_tr_sc, n_pca)

            Xc_tr_pca = pca_transform(Xc_tr_sc, mean_c, comp_c)
            Xc_te_pca = pca_transform(Xc_te_sc, mean_c, comp_c)
            Xl_tr_pca = pca_transform(Xl_tr_sc, mean_l, comp_l)
            Xl_te_pca = pca_transform(Xl_te_sc, mean_l, comp_l)

            # combined layout: cols 0..n_pca-1 = cytokine PCs
            #                  cols n_pca..2*n_pca-1 = cell type PCs
            X_train_combined = np.hstack([Xc_tr_pca, Xl_tr_pca])
            X_test_combined  = np.hstack([Xc_te_pca, Xl_te_pca])

            y_pred = knn_predict(X_train_combined, y_train, X_test_combined, k_neighbors)
            acc    = np.mean(y_test == y_pred)

            fold_accs.append(acc)
            fold_preds.append((y_test, y_pred))
            fold_loadings_cyto.append(comp_c)
            fold_loadings_cell.append(comp_l)
            fold_var_ratios_cyto.append(vr_c)
            fold_var_ratios_cell.append(vr_l)
            fold_projections.append((X_train_combined, X_test_combined, y_train, y_test))

        results[(n_pca, k_neighbors)] = {
            "mean_acc":              np.mean(fold_accs),
            "std_acc":               np.std(fold_accs),
            "fold_accs":             fold_accs,
            "fold_preds":            fold_preds,
            "fold_loadings_cyto":    fold_loadings_cyto,
            "fold_loadings_cell":    fold_loadings_cell,
            "fold_var_ratios_cyto":  fold_var_ratios_cyto,
            "fold_var_ratios_cell":  fold_var_ratios_cell,
            "fold_projections":      fold_projections,
        }
        print(f"  PCA={n_pca:2d}/modality, K={k_neighbors} | "
              f"Mean Acc={np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
print()

# ================================================================================
# SECTION 6: HYPERPARAMETER RANKING
# ================================================================================
ranked = sorted(
    results.items(),
    key=lambda x: (round(x[1]["mean_acc"], 4), -x[1]["std_acc"]),
    reverse=True
)

summary_rows = []
for (n_pca, k), res in ranked:
    summary_rows.append({
        "PCA Components (per modality)": n_pca,
        "K Neighbors":                   k,
        "Mean Accuracy":                 round(res["mean_acc"], 4),
        "Std Accuracy":                  round(res["std_acc"],  4),
        "Fold Accs":                     [round(a, 4) for a in res["fold_accs"]],
    })
pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "results_summary.csv", index=False)

print("=" * 65)
print("SECTION 6: HYPERPARAMETER RANKING")
print("=" * 65)
print(f"  {'Rank':<5} {'PCA/mod':<9} {'K':<5} {'Mean Acc':<12} {'Std Acc':<12}")
print("  " + "-" * 44)
for i, ((n_pca, k), res) in enumerate(ranked[:5]):
    marker = " ← BEST" if i == 0 else (" ← 2nd" if i == 1 else "")
    print(f"  {i+1:<5} {n_pca:<9} {k:<5} "
          f"{res['mean_acc']:.4f}       {res['std_acc']:.4f}{marker}")
print()

best_key    = ranked[0][0]
second_key  = ranked[1][0]
best_result = results[best_key]
print(f"  Selected: PCA={best_key[0]}/modality, K={best_key[1]}")
print(f"  Combined feature dim: {best_key[0]*2}")
print(f"  Mean Acc: {best_result['mean_acc']:.4f}  Std: {best_result['std_acc']:.4f}")
print()

# ================================================================================
# SECTION 7: BEST MODEL METRICS
# ================================================================================
cm_best = np.zeros((2, 2), dtype=int)
for y_true, y_pred in best_result["fold_preds"]:
    cm_best += compute_confusion_matrix(y_true, y_pred)
metrics = classification_metrics(cm_best)

print("=" * 65)
print(f"SECTION 7: BEST MODEL METRICS  (PCA={best_key[0]}/mod, K={best_key[1]})")
print("=" * 65)
print(f"\n  Confusion Matrix:")
print(f"                  Pred Female   Pred Male")
print(f"  Actual Female : {cm_best[0,0]:>6}        {cm_best[0,1]:>6}")
print(f"  Actual Male   : {cm_best[1,0]:>6}        {cm_best[1,1]:>6}")
print()
for name, val in metrics.items():
    print(f"    {name:<22}: {val:.4f}")
print()

# ================================================================================
# SECTION 8: FEATURE IMPORTANCE
# ================================================================================
def compute_importance(fold_loadings, fold_var_ratios, feature_names, label):
    fold_importances = []
    for components, var_ratio in zip(fold_loadings, fold_var_ratios):
        weighted = np.abs(components) * var_ratio[:, np.newaxis]
        fold_importances.append(weighted.sum(axis=0))
    mean_imp = np.mean(fold_importances, axis=0)
    std_imp  = np.std(fold_importances,  axis=0)
    rank_idx = np.argsort(mean_imp)[::-1]
    imp_df   = pd.DataFrame({
        "Feature":    [feature_names[i] for i in rank_idx],
        "Importance": mean_imp[rank_idx].round(6),
        "Std":        std_imp[rank_idx].round(6),
        "Modality":   label,
    })
    return imp_df, mean_imp, std_imp, rank_idx

importance_cyto_df, mean_imp_cyto, std_imp_cyto, rank_cyto = compute_importance(
    best_result["fold_loadings_cyto"],
    best_result["fold_var_ratios_cyto"],
    CYTO_COLS, "Cytokine"
)
importance_cell_df, mean_imp_cell, std_imp_cell, rank_cell = compute_importance(
    best_result["fold_loadings_cell"],
    best_result["fold_var_ratios_cell"],
    CELL_COLS, "Cell Type"
)

pd.concat([importance_cyto_df, importance_cell_df], ignore_index=True).to_csv(
    OUTPUT_DIR / "feature_importance.csv", index=False
)

print("=" * 65)
print("SECTION 8: FEATURE IMPORTANCE")
print("=" * 65)
print("  Top 10 CYTOKINES:")
for rank, i in enumerate(rank_cyto[:10]):
    bar = "█" * int(mean_imp_cyto[i] / mean_imp_cyto[rank_cyto[0]] * 20)
    print(f"  {rank+1:>2}. {CYTO_COLS[i]:<14} "
          f"{mean_imp_cyto[i]:.4f} ± {std_imp_cyto[i]:.4f}  {bar}")
print()
print("  Top 10 CELL TYPES:")
for rank, i in enumerate(rank_cell[:10]):
    bar = "█" * int(mean_imp_cell[i] / mean_imp_cell[rank_cell[0]] * 20)
    print(f"  {rank+1:>2}. {CELL_COLS[i]:<40} "
          f"{mean_imp_cell[i]:.4f} ± {std_imp_cell[i]:.4f}  {bar}")
print()

# ================================================================================
# SECTION 8b: SEX-BASED CYTOKINE COMPARISON
# ================================================================================
from scipy import stats

print("=" * 65)
print("SECTION 8b: SEX-BASED CYTOKINE COMPARISON")
print("=" * 65)

male_idx   = np.where(y == 1)[0]
female_idx = np.where(y == 0)[0]

X_cyto_log = np.log1p(X_cyto)

male_means   = X_cyto_log[male_idx].mean(axis=0)
female_means = X_cyto_log[female_idx].mean(axis=0)
male_std     = X_cyto_log[male_idx].std(axis=0)
female_std   = X_cyto_log[female_idx].std(axis=0)
mean_diff    = male_means - female_means

# Welch's t-test per cytokine
pvals  = []
tstats = []
for i in range(len(CYTO_COLS)):
    t, p = stats.ttest_ind(
        X_cyto_log[male_idx, i],
        X_cyto_log[female_idx, i],
        equal_var=False
    )
    pvals.append(p)
    tstats.append(t)

sex_comparison_df = pd.DataFrame({
    "Cytokine":        CYTO_COLS,
    "Male_Mean":       male_means.round(4),
    "Female_Mean":     female_means.round(4),
    "Male_Std":        male_std.round(4),
    "Female_Std":      female_std.round(4),
    "Mean_Diff (M-F)": mean_diff.round(4),
    "Abs_Diff":        np.abs(mean_diff).round(4),
    "T_Statistic":     np.array(tstats).round(4),
    "P_Value":         np.array(pvals).round(6),
    "Significant":     [p < 0.05 for p in pvals],
    "Direction":       ["↑ Male" if d > 0 else "↑ Female" for d in mean_diff],
}).sort_values("Abs_Diff", ascending=False).reset_index(drop=True)

sex_comparison_df.to_csv(OUTPUT_DIR / "cytokine_sex_comparison.csv", index=False)

print("  Top 10 cytokines by male/female mean difference (log1p scale):")
print(f"  {'Cytokine':<12} {'Male':>8} {'Female':>8} {'Diff':>8} {'p-val':>10} {'Sig':>6} {'Direction':>10}")
print("  " + "-" * 65)
for _, row in sex_comparison_df.head(10).iterrows():
    sig = "✓" if row["Significant"] else ""
    print(f"  {row['Cytokine']:<12} {row['Male_Mean']:>8.4f} {row['Female_Mean']:>8.4f} "
          f"{row['Mean_Diff (M-F)']:>8.4f} {row['P_Value']:>10.4f} {sig:>6}  {row['Direction']:>10}")
print()
print(f"  Significant cytokines (p<0.05): "
      f"{sex_comparison_df[sex_comparison_df['Significant']]['Cytokine'].tolist()}")
print()

# ---- Also run t-test on cell type proportions ----
print("  Top 10 cell types by male/female mean difference:")
print(f"  {'Cell Type':<45} {'Diff':>8} {'p-val':>10} {'Sig':>6}")
print("  " + "-" * 75)

cell_pvals, cell_tstats, cell_diffs = [], [], []
for i in range(len(CELL_COLS)):
    t, p = stats.ttest_ind(
        X_cell[male_idx, i],
        X_cell[female_idx, i],
        equal_var=False
    )
    cell_pvals.append(p)
    cell_tstats.append(t)
    cell_diffs.append(X_cell[male_idx, i].mean() - X_cell[female_idx, i].mean())

cell_comparison_df = pd.DataFrame({
    "Cell_Type":       CELL_COLS,
    "Male_Mean":       X_cell[male_idx].mean(axis=0).round(4),
    "Female_Mean":     X_cell[female_idx].mean(axis=0).round(4),
    "Male_Std":        X_cell[male_idx].std(axis=0).round(4),
    "Female_Std":      X_cell[female_idx].std(axis=0).round(4),
    "Mean_Diff (M-F)": np.array(cell_diffs).round(4),
    "Abs_Diff":        np.abs(cell_diffs).round(4),
    "T_Statistic":     np.array(cell_tstats).round(4),
    "P_Value":         np.array(cell_pvals).round(6),
    "Significant":     [p < 0.05 for p in cell_pvals],
    "Direction":       ["↑ Male" if d > 0 else "↑ Female" for d in cell_diffs],
}).sort_values("Abs_Diff", ascending=False).reset_index(drop=True)

cell_comparison_df.to_csv(OUTPUT_DIR / "celltype_sex_comparison.csv", index=False)

for _, row in cell_comparison_df.head(10).iterrows():
    sig = "✓" if row["Significant"] else ""
    print(f"  {row['Cell_Type']:<45} {row['Mean_Diff (M-F)']:>8.4f} "
          f"{row['P_Value']:>10.4f} {sig:>6}  {row['Direction']}")
print()
print(f"  Significant cell types (p<0.05): "
      f"{cell_comparison_df[cell_comparison_df['Significant']]['Cell_Type'].tolist()}")
print()

# ---- Plot: cytokine comparison ----
fig_sex, ax_sex = plt.subplots(figsize=(12, 6))
x     = np.arange(len(CYTO_COLS))
width = 0.35
cyto_ordered = sex_comparison_df["Cytokine"].tolist()

ax_sex.bar(x - width/2,
           sex_comparison_df["Male_Mean"],   width,
           label="Male",   color="#6BAED6", alpha=0.85, edgecolor="white",
           yerr=sex_comparison_df["Male_Std"],   capsize=3)
ax_sex.bar(x + width/2,
           sex_comparison_df["Female_Mean"], width,
           label="Female", color="#E87D7D", alpha=0.85, edgecolor="white",
           yerr=sex_comparison_df["Female_Std"], capsize=3)

for idx, (_, row) in enumerate(sex_comparison_df.iterrows()):
    if row["Significant"]:
        y_pos = max(row["Male_Mean"] + row["Male_Std"],
                    row["Female_Mean"] + row["Female_Std"]) + 0.05
        ax_sex.text(idx, y_pos, "*", ha="center", fontsize=14,
                    color="black", fontweight="bold")

ax_sex.set_xticks(x)
ax_sex.set_xticklabels(cyto_ordered, rotation=45, ha="right", fontsize=9)
ax_sex.set_ylabel("Mean log1p(MFI)", fontsize=10)
ax_sex.set_title(
    "Cytokine Expression by Sex — sorted by largest difference\n"
    "* = statistically significant (p < 0.05, Welch's t-test)",
    fontsize=11, fontweight="bold"
)
ax_sex.legend(fontsize=10)
ax_sex.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cytokine_sex_comparison.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ---- Plot: top 15 cell types by difference ----
top_cells    = cell_comparison_df.head(15)
fig_cell, ax_cell = plt.subplots(figsize=(12, 6))
xc    = np.arange(len(top_cells))
ax_cell.bar(xc - width/2, top_cells["Male_Mean"],   width,
            label="Male",   color="#6BAED6", alpha=0.85, edgecolor="white",
            yerr=top_cells["Male_Std"],   capsize=3)
ax_cell.bar(xc + width/2, top_cells["Female_Mean"], width,
            label="Female", color="#E87D7D", alpha=0.85, edgecolor="white",
            yerr=top_cells["Female_Std"], capsize=3)

for idx, (_, row) in enumerate(top_cells.iterrows()):
    if row["Significant"]:
        y_pos = max(row["Male_Mean"] + row["Male_Std"],
                    row["Female_Mean"] + row["Female_Std"]) + 0.001
        ax_cell.text(idx, y_pos, "*", ha="center", fontsize=14,
                     color="black", fontweight="bold")

ax_cell.set_xticks(xc)
ax_cell.set_xticklabels(
    [c.replace("CT_", "").replace("_", " ") for c in top_cells["Cell_Type"]],
    rotation=45, ha="right", fontsize=8
)
ax_cell.set_ylabel("Mean Proportion (%)", fontsize=10)
ax_cell.set_title(
    "Top 15 Cell Types by Sex Difference\n"
    "* = statistically significant (p < 0.05, Welch's t-test)",
    fontsize=11, fontweight="bold"
)
ax_cell.legend(fontsize=10)
ax_cell.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "celltype_sex_comparison.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figures saved: cytokine_sex_comparison.png, celltype_sex_comparison.png")
print("  CSVs saved:    cytokine_sex_comparison.csv, celltype_sex_comparison.csv")
print()

# ================================================================================
# SECTION 8c: SOFT CLUSTERING (FUZZY C-MEANS) vs KNN
# ================================================================================
print("=" * 65)
print("SECTION 8c: SOFT CLUSTERING — FUZZY C-MEANS vs KNN")
print("=" * 65)

try:
    import skfuzzy as fuzz
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "scikit-fuzzy", "--break-system-packages", "-q"])
    import skfuzzy as fuzz

# Use combined PCA matrix (cytokine PCs + cell type PCs) matching best KNN config
X_combined_full = np.load(OUTPUT_DIR / "X_combined_pca.npy")

N_CLUSTERS_OPTIONS = list(range(2, 7))
FCM_M        = 2.0
FCM_ERROR    = 0.005
FCM_MAX_ITER = 1000

fcm_results = {}
print("  Running Fuzzy C-Means on combined cytokine + cell type PCA space...")
for n_clusters in N_CLUSTERS_OPTIONS:
    data_T = X_combined_full.T

    cntr, u, _, _, _, n_iter, fpc = fuzz.cluster.cmeans(
        data_T,
        c=n_clusters,
        m=FCM_M,
        error=FCM_ERROR,
        maxiter=FCM_MAX_ITER,
        init=None,
        seed=42
    )

    hard_labels    = np.argmax(u, axis=0)
    max_membership = u.max(axis=0)

    cluster_sex = {}
    for c in range(n_clusters):
        mask = hard_labels == c
        if mask.sum() > 0:
            cluster_sex[c] = int(np.round(y[mask].mean()))
        else:
            cluster_sex[c] = 0

    y_pred_fcm = np.array([cluster_sex[c] for c in hard_labels])
    acc_fcm    = np.mean(y_pred_fcm == y)
    cm_fcm     = compute_confusion_matrix(y, y_pred_fcm)
    met_fcm    = classification_metrics(cm_fcm)

    fcm_results[n_clusters] = {
        "cntr":           cntr,
        "u":              u,
        "hard_labels":    hard_labels,
        "max_membership": max_membership,
        "fpc":            fpc,
        "cluster_sex":    cluster_sex,
        "y_pred":         y_pred_fcm,
        "accuracy":       acc_fcm,
        "metrics":        met_fcm,
        "cm":             cm_fcm,
    }

    print(f"  k={n_clusters} | FPC={fpc:.4f} | "
          f"Acc={acc_fcm:.4f} | F1={met_fcm['F1 Score']:.4f} | "
          f"Mean membership={max_membership.mean():.4f}")

print()

best_fcm_k   = max(fcm_results, key=lambda k: fcm_results[k]["accuracy"])
best_fcm_res = fcm_results[best_fcm_k]

print(f"  Best FCM config : k={best_fcm_k}")
print(f"  Accuracy        : {best_fcm_res['accuracy']:.4f}")
print(f"  F1 Score        : {best_fcm_res['metrics']['F1 Score']:.4f}")
print(f"  FPC             : {best_fcm_res['fpc']:.4f}  "
      f"(1.0=crisp, closer to 1/k={1/best_fcm_k:.2f} = fully fuzzy)")
print()

print("  COMPARISON: Fuzzy C-Means vs KNN (cytokines + cell types)")
print(f"  {'Method':<30} {'Accuracy':>10} {'F1 Score':>10} {'Sensitivity':>12} {'Specificity':>12}")
print("  " + "-" * 77)
print(f"  {'KNN (full features)':<30} "
      f"{metrics['Accuracy']:>10.4f} "
      f"{metrics['F1 Score']:>10.4f} "
      f"{metrics['Sensitivity']:>12.4f} "
      f"{metrics['Specificity']:>12.4f}")
print(f"  {'Fuzzy C-Means (k=' + str(best_fcm_k) + ')':<30} "
      f"{best_fcm_res['accuracy']:>10.4f} "
      f"{best_fcm_res['metrics']['F1 Score']:>10.4f} "
      f"{best_fcm_res['metrics']['Sensitivity']:>12.4f} "
      f"{best_fcm_res['metrics']['Specificity']:>12.4f}")
print()

winner = "KNN" if metrics["Accuracy"] >= best_fcm_res["accuracy"] else "Fuzzy C-Means"
print(f"  → {winner} performs better for sex classification (cytokines + cell types)")
print()

comparison_rows = []
comparison_rows.append({
    "Method":      f"KNN (PCA={best_key[0]}/mod, K={best_key[1]})",
    "Accuracy":    metrics["Accuracy"],
    "F1_Score":    metrics["F1 Score"],
    "Sensitivity": metrics["Sensitivity"],
    "Specificity": metrics["Specificity"],
    "Notes":       "Cross-validated, combined cytokine + cell type PCs"
})
for k, res in fcm_results.items():
    comparison_rows.append({
        "Method":      f"Fuzzy C-Means (k={k})",
        "Accuracy":    round(res["accuracy"], 4),
        "F1_Score":    round(res["metrics"]["F1 Score"], 4),
        "Sensitivity": round(res["metrics"]["Sensitivity"], 4),
        "Specificity": round(res["metrics"]["Specificity"], 4),
        "Notes":       f"FPC={res['fpc']:.4f}, full dataset (no CV), combined PCA space"
    })
pd.DataFrame(comparison_rows).to_csv(
    OUTPUT_DIR / "fcm_vs_knn_comparison.csv", index=False
)
print("  CSV saved: fcm_vs_knn_comparison.csv")
print()

# ---- Visualization ----
fig_fcm, axes = plt.subplots(1, 3, figsize=(18, 5))
fig_fcm.suptitle(
    f"Fuzzy C-Means Analysis (best k={best_fcm_k}) vs KNN — Cytokines + Cell Types",
    fontsize=13, fontweight="bold"
)

ax1 = axes[0]
fpcs = [fcm_results[k]["fpc"] for k in N_CLUSTERS_OPTIONS]
accs = [fcm_results[k]["accuracy"] for k in N_CLUSTERS_OPTIONS]
ax1.plot(N_CLUSTERS_OPTIONS, fpcs, marker="o", color="#4A7FC1",
         linewidth=2, markersize=7, label="FPC")
ax1.set_xlabel("Number of Clusters (k)", fontsize=10)
ax1.set_ylabel("Fuzzy Partition Coefficient", fontsize=10, color="#4A7FC1")
ax1.tick_params(axis="y", labelcolor="#4A7FC1")
ax1_twin = ax1.twinx()
ax1_twin.plot(N_CLUSTERS_OPTIONS, accs, marker="s", color="#E87D7D",
              linewidth=2, markersize=7, linestyle="--", label="Accuracy")
ax1_twin.set_ylabel("Accuracy", fontsize=10, color="#E87D7D")
ax1_twin.tick_params(axis="y", labelcolor="#E87D7D")
ax1_twin.axhline(metrics["Accuracy"], color="gray", linestyle=":",
                 linewidth=1.5, label=f"KNN baseline={metrics['Accuracy']:.3f}")
ax1.set_title("FPC & Accuracy vs k", fontsize=11, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

ax2 = axes[1]
hard_labels_best = best_fcm_res["hard_labels"]
cmap_clusters    = plt.cm.get_cmap("tab10", best_fcm_k)
for c in range(best_fcm_k):
    mask     = hard_labels_best == c
    n_male   = y[mask].sum()
    n_female = (y[mask] == 0).sum()
    ax2.scatter(X_combined_full[mask, 0], X_combined_full[mask, 1],
                color=cmap_clusters(c), alpha=0.7, s=40,
                label=f"Cluster {c+1} (M:{n_male} F:{n_female})")
ax2.set_xlabel("Cytokine PC1", fontsize=10)
ax2.set_ylabel("Cytokine PC2", fontsize=10)
ax2.set_title(f"FCM Clusters in Combined PCA Space (k={best_fcm_k})",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=8)
ax2.tick_params(labelsize=8)

ax3 = axes[2]
u_best   = best_fcm_res["u"]
sort_idx = np.lexsort((np.argmax(u_best, axis=0), y))
im3 = ax3.imshow(u_best[:, sort_idx], aspect="auto", cmap="YlOrRd",
                 vmin=0, vmax=1)
ax3.set_xlabel("Samples (sorted by sex then cluster)", fontsize=9)
ax3.set_ylabel("Cluster", fontsize=10)
ax3.set_yticks(range(best_fcm_k))
ax3.set_yticklabels([f"Cluster {c+1}" for c in range(best_fcm_k)], fontsize=9)
ax3.set_title("Membership Matrix\n(brighter = stronger membership)",
              fontsize=11, fontweight="bold")
n_female_sorted = (y[sort_idx] == 0).sum()
ax3.axvline(n_female_sorted - 0.5, color="blue", linewidth=2,
            linestyle="--", label="Sex boundary")
ax3.text(n_female_sorted / 2, -0.7, "Female", ha="center",
         fontsize=9, color="blue")
ax3.text(n_female_sorted + (len(y) - n_female_sorted) / 2, -0.7,
         "Male", ha="center", fontsize=9, color="blue")
plt.colorbar(im3, ax=ax3, label="Membership degree")
ax3.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fcm_vs_knn_analysis.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: fcm_vs_knn_analysis.png")
print()

# ================================================================================
# SECTION 9: VISUALIZATION PANEL
# ================================================================================
COLOR_FEMALE = "#E87D7D"
COLOR_MALE   = "#6BAED6"
COLOR_BAR_C  = "#4A7FC1"
COLOR_BAR_L  = "#5BAD72"

fig = plt.figure(figsize=(20, 13))
fig.suptitle(
    f"Cytokines + Cell Types PCA+KNN — Best Config: PCA={best_key[0]}/modality, K={best_key[1]}\n"
    f"Mean CV Accuracy = {best_result['mean_acc']:.3f} ± {best_result['std_acc']:.3f}",
    fontsize=14, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ---- Panel A: Scree plots — both modalities ----
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(range(1, len(vr_cyto_global)+1), np.cumsum(vr_cyto_global)*100,
          color=COLOR_BAR_C, marker="o", markersize=4, linewidth=1.5, label="Cytokines")
ax_a.plot(range(1, len(vr_cell_global)+1), np.cumsum(vr_cell_global)*100,
          color=COLOR_BAR_L, marker="s", markersize=4, linewidth=1.5, label="Cell Types")
ax_a.axhline(80, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="80%")
ax_a.axhline(90, color="black", linestyle=":",  linewidth=1.0, alpha=0.7, label="90%")
ax_a.set_xlabel("Principal Component", fontsize=9)
ax_a.set_ylabel("Cumulative Variance (%)", fontsize=9)
ax_a.set_title("A: Scree Plot (Both Modalities)", fontsize=10, fontweight="bold")
ax_a.legend(fontsize=8)
ax_a.tick_params(labelsize=8)
ax_a.set_xlim(1, 18)

# ---- Panel B: Grid search heatmap ----
ax_b = fig.add_subplot(gs[0, 1])
heatmap_data = np.array([
    [results[(n_pca, k)]["mean_acc"] for k in K_OPTIONS]
    for n_pca in PCA_OPTIONS
])
im = ax_b.imshow(heatmap_data, cmap="YlOrRd", aspect="auto",
                 vmin=heatmap_data.min(), vmax=heatmap_data.max())
ax_b.set_xticks(range(len(K_OPTIONS)))
ax_b.set_xticklabels([f"K={k}" for k in K_OPTIONS], fontsize=8)
ax_b.set_yticks(range(len(PCA_OPTIONS)))
ax_b.set_yticklabels([f"{p} PCs/mod" for p in PCA_OPTIONS], fontsize=8)
ax_b.set_title("B: CV Accuracy Grid Search", fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax_b, label="Mean Accuracy")
best_i = PCA_OPTIONS.index(best_key[0]);  best_j = K_OPTIONS.index(best_key[1])
ax_b.add_patch(plt.Rectangle((best_j-0.5, best_i-0.5), 1, 1,
               fill=False, edgecolor="blue", linewidth=2.5))
ax_b.text(best_j, best_i, "★", ha="center", va="center",
          fontsize=12, color="blue", fontweight="bold")

# ---- Panel C: Confusion matrix ----
ax_c = fig.add_subplot(gs[0, 2])
ax_c.imshow(cm_best, cmap="Blues")
ax_c.set_xticks([0, 1]);  ax_c.set_xticklabels(["Pred Female", "Pred Male"],  fontsize=9)
ax_c.set_yticks([0, 1]);  ax_c.set_yticklabels(["Act Female",  "Act Male"],   fontsize=9)
ax_c.set_title(f"C: Confusion Matrix\nAcc={metrics['Accuracy']:.3f}  F1={metrics['F1 Score']:.3f}",
               fontsize=10, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax_c.text(j, i, str(cm_best[i, j]), ha="center", va="center",
                  fontsize=16, fontweight="bold",
                  color="white" if cm_best[i, j] > cm_best.max() / 2 else "black")

# ---- Panel D: PCA scatter — cytokine PC1 vs cell type PC1, colored by sex ----
ax_d = fig.add_subplot(gs[1, 0])
X_tr_d, X_te_d, y_tr_d, y_te_d = best_result["fold_projections"][-1]
n_pca_best   = best_key[0]
cyto_pc1_col = 0           # cytokine PC1 is always column 0
cell_pc1_col = n_pca_best  # cell type PCs start right after cytokine PCs

for y_mask_tr, y_mask_te, label, color in [
    (y_tr_d == 0, y_te_d == 0, "Female", COLOR_FEMALE),
    (y_tr_d == 1, y_te_d == 1, "Male",   COLOR_MALE),
]:
    ax_d.scatter(
        X_tr_d[y_mask_tr, cyto_pc1_col],
        X_tr_d[y_mask_tr, cell_pc1_col],
        c=color, marker="o", alpha=0.7, s=40, label=f"{label} (train)"
    )
    ax_d.scatter(
        X_te_d[y_mask_te, cyto_pc1_col],
        X_te_d[y_mask_te, cell_pc1_col],
        c=color, marker="^", alpha=0.7, s=40, label=f"{label} (test)"
    )

ax_d.set_xlabel("Cytokine PC1", fontsize=9)
ax_d.set_ylabel("Cell Type PC1", fontsize=9)
ax_d.set_title("D: PCA Space — Cyto PC1 vs Cell PC1", fontsize=10, fontweight="bold")
ax_d.legend(fontsize=7, ncol=2)
ax_d.tick_params(labelsize=8)

# ---- Panel E: Feature importance — cytokines top 10 ----
ax_e = fig.add_subplot(gs[1, 1])
top_n    = 10
top_cyto = rank_cyto[:top_n]
ax_e.barh(range(top_n), mean_imp_cyto[top_cyto][::-1],
          xerr=std_imp_cyto[top_cyto][::-1],
          color=COLOR_BAR_C, alpha=0.8, edgecolor="white", capsize=3)
ax_e.set_yticks(range(top_n))
ax_e.set_yticklabels([CYTO_COLS[i] for i in top_cyto[::-1]], fontsize=8)
ax_e.set_xlabel("Importance (variance-weighted loading)", fontsize=9)
ax_e.set_title("E: Top Cytokine Features", fontsize=10, fontweight="bold")
ax_e.tick_params(labelsize=8)

# ---- Panel F: Per-fold accuracy ----
ax_f = fig.add_subplot(gs[1, 2])
fold_acc_vals = best_result["fold_accs"]
bar_colors    = [COLOR_BAR_C if a >= best_result["mean_acc"] else "#AAAAAA"
                 for a in fold_acc_vals]
ax_f.bar([f"Fold {i+1}" for i in range(N_SPLITS)], fold_acc_vals,
         color=bar_colors, alpha=0.85, edgecolor="white")
ax_f.axhline(best_result["mean_acc"], color="darkred", linestyle="--",
             linewidth=1.5, label=f"Mean={best_result['mean_acc']:.3f}")
ax_f.fill_between(
    range(N_SPLITS),
    best_result["mean_acc"] - best_result["std_acc"],
    best_result["mean_acc"] + best_result["std_acc"],
    alpha=0.15, color="darkred", label=f"±Std={best_result['std_acc']:.3f}"
)
ax_f.set_ylabel("Accuracy", fontsize=9)
ax_f.set_title(f"F: Per-Fold Accuracy (PCA={best_key[0]}/mod, K={best_key[1]})",
               fontsize=10, fontweight="bold")
ax_f.legend(fontsize=8)
ax_f.set_ylim(0, 1.05)
ax_f.tick_params(labelsize=8)
for i, v in enumerate(fold_acc_vals):
    ax_f.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

# Save full panel figure
plt.savefig(FIGURES_DIR / "pca_knn_full_results.png",
            dpi=150, bbox_inches="tight", facecolor="white")

# Save individual panels
fig.canvas.draw()
save_panel(fig, ax_a,  "panel_A_scree.png")
save_panel(fig, ax_b,  "panel_B_heatmap.png")
save_panel(fig, ax_c,  "panel_C_confusion.png")
save_panel(fig, ax_d,  "panel_D_pca_scatter.png")
save_panel(fig, ax_e,  "panel_E_importance.png")

plt.close()
print("  Figures saved to:", FIGURES_DIR)
print()

# ================================================================================
# SECTION 10: EXPORT
# ================================================================================
def full_dataset_pca(X_raw, n_components):
    mean = X_raw.mean(axis=0)
    std  = X_raw.std(axis=0);  std[std == 0] = 1
    X_sc = (X_raw - mean) / std
    comp, vr, train_mean = pca_fit(X_sc, n_components)
    X_pca = pca_transform(X_sc, train_mean, comp)
    return X_sc, X_pca, comp

X_cyto_sc, X_cyto_pca, comp_cyto = full_dataset_pca(X_cyto, best_key[0])
X_cell_sc, X_cell_pca, comp_cell = full_dataset_pca(X_cell, best_key[0])
X_combined = np.hstack([X_cyto_pca, X_cell_pca])

np.save(OUTPUT_DIR / "X_cytokines_processed.npy",    X_cyto_sc)
np.save(OUTPUT_DIR / "X_cell_processed.npy",         X_cell_sc)
np.save(OUTPUT_DIR / "X_cytokines_pca.npy",          X_cyto_pca)
np.save(OUTPUT_DIR / "X_cell_pca.npy",               X_cell_pca)
np.save(OUTPUT_DIR / "X_combined_pca.npy",           X_combined)
np.save(OUTPUT_DIR / "pca_components_cytokines.npy", comp_cyto)
np.save(OUTPUT_DIR / "pca_components_cell.npy",      comp_cell)
np.save(OUTPUT_DIR / "y_labels.npy",                 y)

print("=" * 65)
print("PIPELINE COMPLETE  (cytokines + cell types)")
print("=" * 65)
print(f"  Best config      : PCA={best_key[0]}/modality, K={best_key[1]}")
print(f"  Combined features: {best_key[0]*2} ({best_key[0]} cyto + {best_key[0]} cell)")
print(f"  Accuracy         : {metrics['Accuracy']:.4f}")
print(f"  Balanced Acc     : {metrics['Balanced Accuracy']:.4f}")
print(f"  Sensitivity      : {metrics['Sensitivity']:.4f}  (Male detection)")
print(f"  Specificity      : {metrics['Specificity']:.4f}  (Female detection)")
print(f"  F1 Score         : {metrics['F1 Score']:.4f}")
print(f"  Top cytokine     : {importance_cyto_df.iloc[0]['Feature']} "
      f"(importance={importance_cyto_df.iloc[0]['Importance']:.4f})")
print(f"  Top cell type    : {importance_cell_df.iloc[0]['Feature']} "
      f"(importance={importance_cell_df.iloc[0]['Importance']:.4f})")
print(f"  Outputs          : {OUTPUT_DIR}")
print("=" * 65)