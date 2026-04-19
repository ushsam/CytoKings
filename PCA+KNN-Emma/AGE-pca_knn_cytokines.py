"""
================================================================================
Cytokine-Only PCA + KNN Binary Age Classification Pipeline
Young (18-39) vs Older (40-66)
================================================================================
Outputs → PCA+KNN-Emma/Outputs/age_binary/
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

# ================================================================================
# PATHS
# ================================================================================
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data-Emma"
OUTPUT_DIR  = BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "age_binary"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# SECTION 1: DATA LOADING
# ================================================================================
df = pd.read_csv(DATA_DIR / "analysis_merged_subject_level_cell-type.csv")

CYTO_COLS = [
    "IFN-gamma", "IL-12p70", "IL-13", "IL-1beta", "IL-2", "IL-4", "IL-5",
    "IL-6", "TNF-alpha", "GM-CSF", "IL-18", "IL-10", "IL-17A", "IL-21",
    "IL-22", "IL-23", "IL-27", "IL-9"
]

X_raw = df[CYTO_COLS].values.astype(float)

# Binary age label: 0 = Young (18-39), 1 = Older (40-66)
df['AGE_BINARY'] = df['AGEGR1C'].apply(
    lambda x: 0 if x in ['18-29', '30-39'] else 1
)
y         = df['AGE_BINARY'].values
n_samples = len(df)
n_missing = np.isnan(X_raw).sum()

print("=" * 65)
print("SECTION 1: DATASET SUMMARY  (cytokines only, binary age)")
print("=" * 65)
print(f"  Total subjects : {n_samples}")
print(f"  Cytokines      : {len(CYTO_COLS)}")
print(f"  Young (18-39)  : {(y==0).sum()}  ({100*(y==0).sum()/n_samples:.1f}%)")
print(f"  Older (40-66)  : {(y==1).sum()}  ({100*(y==1).sum()/n_samples:.1f}%)")
print(f"  Missing values : {n_missing}")
if n_missing > 0:
    print("  WARNING: missing values detected.")
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
    fig.canvas.draw()
    extent = ax.get_tightbbox(fig.canvas.get_renderer())
    extent = extent.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(FIGURES_DIR / filename, bbox_inches=extent.expanded(pad, pad),
                dpi=150, facecolor="white")
    print(f"  Panel saved: {filename}")


# ================================================================================
# SECTION 3: GLOBAL PCA
# ================================================================================
print("=" * 65)
print("SECTION 3: GLOBAL PCA — INTRINSIC DIMENSIONALITY")
print("=" * 65)

X_log_global    = np.log1p(X_raw)
X_global_mean   = X_log_global.mean(axis=0)
X_global_std    = X_log_global.std(axis=0)
X_global_std[X_global_std == 0] = 1
X_global_scaled = (X_log_global - X_global_mean) / X_global_std

X_centered_global = X_global_scaled - X_global_scaled.mean(axis=0)
X_centered_global = np.nan_to_num(X_centered_global,
                                   nan=0.0, posinf=0.0, neginf=0.0)

_, S_global, Vt_global = np.linalg.svd(X_centered_global, full_matrices=False)

eigenvalues_global     = (S_global ** 2) / (n_samples - 1)
explained_ratio_global = eigenvalues_global / eigenvalues_global.sum()
cumulative_var_global  = np.cumsum(explained_ratio_global)

n_pcs_80 = int(np.searchsorted(cumulative_var_global, 0.80)) + 1
n_pcs_90 = int(np.searchsorted(cumulative_var_global, 0.90)) + 1

print(f"  PC1 explains : {explained_ratio_global[0]*100:.1f}%")
print(f"  PC2 explains : {explained_ratio_global[1]*100:.1f}%")
print(f"  PC3 explains : {explained_ratio_global[2]*100:.1f}%")
print(f"  PCs for 80%  : {n_pcs_80}")
print(f"  PCs for 90%  : {n_pcs_90}")
print()

X_global_pca = X_centered_global @ Vt_global[:3].T

pc_labels = ["PC1 — Global Inflammation",
             "PC2 — Th17/Regulatory",
             "PC3 — Th1/Innate (IFN-γ, IL-18)"]

loadings_df = pd.DataFrame(
    Vt_global[:3].T,
    index=CYTO_COLS,
    columns=["PC1", "PC2", "PC3"]
).round(4)
loadings_df.to_csv(OUTPUT_DIR / "age-pca_loadings_global.csv")

print("  PC loadings (top contributors per PC):")
for pc_idx, pc_label in enumerate(pc_labels):
    top3_idx = np.argsort(np.abs(Vt_global[pc_idx]))[::-1][:3]
    top3     = [(CYTO_COLS[i], Vt_global[pc_idx, i]) for i in top3_idx]
    print(f"  {pc_label}:")
    for name, val in top3:
        print(f"    {name:<12} loading={val:+.3f}")
print()

# ---- Global PCA visualization ----
COLOR_YOUNG  = "#E87D7D"
COLOR_OLDER  = "#6BAED6"
COLOR_BAR    = "#4A7FC1"

fig_pca, axes_pca = plt.subplots(1, 3, figsize=(18, 5))
fig_pca.suptitle(
    "Global PCA — Cytokine Expression (full dataset, exploratory)",
    fontsize=13, fontweight="bold"
)

ax_s = axes_pca[0]
x_pcs = range(1, len(explained_ratio_global) + 1)
ax_s.bar(x_pcs, explained_ratio_global * 100, color=COLOR_BAR, alpha=0.7)
ax_s2 = ax_s.twinx()
ax_s2.plot(x_pcs, cumulative_var_global * 100, color="darkred",
           marker="o", markersize=4, linewidth=1.5, label="Cumulative")
ax_s2.axhline(80, color="gray", linestyle="--", linewidth=0.8,
              alpha=0.7, label="80%")
ax_s2.axhline(90, color="gray", linestyle=":",  linewidth=0.8,
              alpha=0.7, label="90%")
ax_s2.set_ylabel("Cumulative Variance (%)", fontsize=9, color="darkred")
ax_s2.tick_params(axis="y", labelcolor="darkred", labelsize=8)
ax_s2.set_ylim(0, 105)
ax_s2.legend(fontsize=8, loc="center right")
ax_s.set_xlabel("Principal Component", fontsize=9)
ax_s.set_ylabel("Explained Variance (%)", fontsize=9)
ax_s.set_title("Scree Plot", fontsize=11, fontweight="bold")
ax_s.tick_params(labelsize=8)
ax_s.text(n_pcs_80 + 0.2, explained_ratio_global[n_pcs_80-1]*100 + 1,
          f"{n_pcs_80} PCs→80%", fontsize=7, color="gray")

ax_p = axes_pca[1]
for age_label, age_val, color in [("Young", 0, COLOR_YOUNG),
                                   ("Older", 1, COLOR_OLDER)]:
    mask = y == age_val
    ax_p.scatter(X_global_pca[mask, 0], X_global_pca[mask, 1],
                 c=color, alpha=0.7, s=40, label=age_label)
ax_p.set_xlabel(f"PC1 ({explained_ratio_global[0]*100:.1f}%) — Global Inflammation",
                fontsize=9)
ax_p.set_ylabel(f"PC2 ({explained_ratio_global[1]*100:.1f}%) — Th17/Regulatory",
                fontsize=9)
ax_p.set_title("PC1 vs PC2 — colored by age group", fontsize=11, fontweight="bold")
ax_p.legend(fontsize=9)
ax_p.tick_params(labelsize=8)

ax_l = axes_pca[2]
loading_matrix = Vt_global[:3].T
im_l = ax_l.imshow(loading_matrix, cmap="RdBu_r", aspect="auto",
                    vmin=-0.5, vmax=0.5)
ax_l.set_xticks([0, 1, 2])
ax_l.set_xticklabels(["PC1\nInflammation", "PC2\nTh17/Reg",
                       "PC3\nTh1/Innate"], fontsize=8)
ax_l.set_yticks(range(len(CYTO_COLS)))
ax_l.set_yticklabels(CYTO_COLS, fontsize=8)
ax_l.set_title("PC Loadings Heatmap\n(red=positive, blue=negative)",
               fontsize=11, fontweight="bold")
plt.colorbar(im_l, ax=ax_l, label="Loading value")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "global_pca.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: global_pca.png")
print("  CSV saved:    age-pca_loadings_global.csv")
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
print(f"  Strategy     : Stratified {N_SPLITS}-Fold CV")
print(f"  PCA range    : {PCA_OPTIONS[0]}–{PCA_OPTIONS[-1]} components")
print(f"  KNN range    : K = {K_OPTIONS[0]}–{K_OPTIONS[-1]}")
print(f"  Total combos : {len(PCA_OPTIONS) * len(K_OPTIONS)}")
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
        fold_accs, fold_preds, fold_loadings = [], [], []
        fold_var_ratios, fold_projections    = [], []

        for train_idx, test_idx in folds:
            X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
            y_train,     y_test     = y[train_idx],     y[test_idx]

            X_train_sc, X_test_sc, _, _ = standardize(X_train_raw, X_test_raw)
            components, var_ratio, train_mean = pca_fit(X_train_sc, n_pca)
            X_train_pca = pca_transform(X_train_sc, train_mean, components)
            X_test_pca  = pca_transform(X_test_sc,  train_mean, components)
            y_pred      = knn_predict(X_train_pca, y_train, X_test_pca, k_neighbors)
            acc         = np.mean(y_test == y_pred)

            fold_accs.append(acc)
            fold_preds.append((y_test, y_pred))
            fold_loadings.append(components)
            fold_var_ratios.append(var_ratio)
            fold_projections.append((X_train_pca, X_test_pca, y_train, y_test))

        results[(n_pca, k_neighbors)] = {
            "mean_acc":         np.mean(fold_accs),
            "std_acc":          np.std(fold_accs),
            "fold_accs":        fold_accs,
            "fold_preds":       fold_preds,
            "fold_loadings":    fold_loadings,
            "fold_var_ratios":  fold_var_ratios,
            "fold_projections": fold_projections,
        }
        print(f"  PCA={n_pca:2d}, K={k_neighbors} | "
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
        "PCA Components": n_pca,
        "K Neighbors":    k,
        "Mean Accuracy":  round(res["mean_acc"], 4),
        "Std Accuracy":   round(res["std_acc"],  4),
        "Fold Accs":      [round(a, 4) for a in res["fold_accs"]],
    })
pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "age-results_summary.csv", index=False)

print("=" * 65)
print("SECTION 6: HYPERPARAMETER RANKING")
print("=" * 65)
print(f"  {'Rank':<5} {'PCA':<6} {'K':<5} {'Mean Acc':<12} {'Std Acc':<12}")
print("  " + "-" * 40)
for i, ((n_pca, k), res) in enumerate(ranked[:5]):
    marker = " ← BEST" if i == 0 else (" ← 2nd" if i == 1 else "")
    print(f"  {i+1:<5} {n_pca:<6} {k:<5} "
          f"{res['mean_acc']:.4f}       {res['std_acc']:.4f}{marker}")
print()

best_key    = ranked[0][0]
best_result = results[best_key]
print(f"  Selected: PCA={best_key[0]}, K={best_key[1]}")
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
print(f"SECTION 7: BEST MODEL METRICS  (PCA={best_key[0]}, K={best_key[1]})")
print("=" * 65)
print(f"\n  Confusion Matrix:")
print(f"                  Pred Young    Pred Older")
print(f"  Actual Young  : {cm_best[0,0]:>6}        {cm_best[0,1]:>6}")
print(f"  Actual Older  : {cm_best[1,0]:>6}        {cm_best[1,1]:>6}")
print()
for name, val in metrics.items():
    print(f"    {name:<22}: {val:.4f}")
print()

# ================================================================================
# SECTION 8: FEATURE IMPORTANCE
# ================================================================================
fold_importances = []
for components, var_ratio in zip(
    best_result["fold_loadings"], best_result["fold_var_ratios"]
):
    weighted = np.abs(components) * var_ratio[:, np.newaxis]
    fold_importances.append(weighted.sum(axis=0))

mean_importance = np.mean(fold_importances, axis=0)
std_importance  = np.std(fold_importances,  axis=0)
ranking_idx     = np.argsort(mean_importance)[::-1]

importance_df = pd.DataFrame({
    "Cytokine":   [CYTO_COLS[i] for i in ranking_idx],
    "Importance": mean_importance[ranking_idx].round(6),
    "Std":        std_importance[ranking_idx].round(6),
})
importance_df.to_csv(OUTPUT_DIR / "age-feature_importance.csv", index=False)

print("=" * 65)
print("SECTION 8: FEATURE IMPORTANCE — Top 10 cytokines")
print("=" * 65)
for rank, i in enumerate(ranking_idx[:10]):
    bar = "█" * int(mean_importance[i] / mean_importance[ranking_idx[0]] * 20)
    print(f"  {rank+1:>2}. {CYTO_COLS[i]:<12} "
          f"{mean_importance[i]:.4f} ± {std_importance[i]:.4f}  {bar}")
print()

# ================================================================================
# SECTION 8b: AGE-BASED CYTOKINE COMPARISON
# ================================================================================
from scipy import stats

print("=" * 65)
print("SECTION 8b: AGE-BASED CYTOKINE COMPARISON (Young vs Older)")
print("=" * 65)

older_idx = np.where(y == 1)[0]
young_idx = np.where(y == 0)[0]

X_log = np.log1p(X_raw)

older_means = X_log[older_idx].mean(axis=0)
young_means = X_log[young_idx].mean(axis=0)
older_std   = X_log[older_idx].std(axis=0)
young_std   = X_log[young_idx].std(axis=0)
mean_diff   = older_means - young_means

pvals  = []
tstats = []
for i in range(len(CYTO_COLS)):
    t, p = stats.ttest_ind(
        X_log[older_idx, i],
        X_log[young_idx, i],
        equal_var=False
    )
    pvals.append(p)
    tstats.append(t)

age_comparison_df = pd.DataFrame({
    "Cytokine":         CYTO_COLS,
    "Older_Mean":       older_means.round(4),
    "Young_Mean":       young_means.round(4),
    "Older_Std":        older_std.round(4),
    "Young_Std":        young_std.round(4),
    "Mean_Diff (O-Y)":  mean_diff.round(4),
    "Abs_Diff":         np.abs(mean_diff).round(4),
    "T_Statistic":      np.array(tstats).round(4),
    "P_Value":          np.array(pvals).round(6),
    "Significant":      [p < 0.05 for p in pvals],
    "Direction":        ["↑ Older" if d > 0 else "↑ Young" for d in mean_diff],
}).sort_values("Abs_Diff", ascending=False).reset_index(drop=True)

age_comparison_df.to_csv(OUTPUT_DIR / "cytokine_age_comparison.csv", index=False)

print("  Top 10 cytokines by older/young mean difference (log1p scale):")
print(f"  {'Cytokine':<12} {'Older':>8} {'Young':>8} {'Diff':>8} {'p-val':>10} {'Sig':>6} {'Direction':>10}")
print("  " + "-" * 65)
for _, row in age_comparison_df.head(10).iterrows():
    sig = "✓" if row["Significant"] else ""
    print(f"  {row['Cytokine']:<12} {row['Older_Mean']:>8.4f} {row['Young_Mean']:>8.4f} "
          f"{row['Mean_Diff (O-Y)']:>8.4f} {row['P_Value']:>10.4f} {sig:>6}  {row['Direction']:>10}")
print()
print(f"  Significant cytokines (p<0.05): "
      f"{age_comparison_df[age_comparison_df['Significant']]['Cytokine'].tolist()}")
print()

cyto_ordered = age_comparison_df["Cytokine"].tolist()
x     = np.arange(len(CYTO_COLS))
width = 0.35

fig_age, ax_age = plt.subplots(figsize=(12, 6))
ax_age.bar(x - width/2, age_comparison_df["Older_Mean"], width,
           label="Older", color=COLOR_OLDER, alpha=0.85,
           edgecolor="white", yerr=age_comparison_df["Older_Std"], capsize=3)
ax_age.bar(x + width/2, age_comparison_df["Young_Mean"], width,
           label="Young", color=COLOR_YOUNG, alpha=0.85,
           edgecolor="white", yerr=age_comparison_df["Young_Std"], capsize=3)

for idx, (_, row) in enumerate(age_comparison_df.iterrows()):
    if row["Significant"]:
        y_pos = max(row["Older_Mean"] + row["Older_Std"],
                    row["Young_Mean"] + row["Young_Std"]) + 0.05
        ax_age.text(idx, y_pos, "*", ha="center", fontsize=14,
                    color="black", fontweight="bold")

ax_age.set_xticks(x)
ax_age.set_xticklabels(cyto_ordered, rotation=45, ha="right", fontsize=9)
ax_age.set_ylabel("Mean log1p(MFI)", fontsize=10)
ax_age.set_title(
    "Cytokine Expression by Age Group — sorted by largest difference\n"
    "* = statistically significant (p < 0.05, Welch's t-test)",
    fontsize=11, fontweight="bold"
)
ax_age.legend(fontsize=10)
ax_age.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cytokine_age_comparison.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: cytokine_age_comparison.png")
print("  CSV saved:    cytokine_age_comparison.csv")
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

X_proc_mean_sc = X_raw.mean(axis=0)
X_proc_std_sc  = X_raw.std(axis=0);  X_proc_std_sc[X_proc_std_sc == 0] = 1
X_sc_full      = (X_raw - X_proc_mean_sc) / X_proc_std_sc
comp_fcm, _, mean_fcm = pca_fit(X_sc_full, best_key[0])
X_pca_full     = pca_transform(X_sc_full, mean_fcm, comp_fcm)

N_CLUSTERS_OPTIONS = list(range(2, 7))
FCM_M        = 2.0
FCM_ERROR    = 0.005
FCM_MAX_ITER = 1000

def run_fcm(X_pca, label):
    fcm_res = {}
    print(f"  Running Fuzzy C-Means on {label} space...")
    for n_clusters in N_CLUSTERS_OPTIONS:
        cntr, u, _, _, _, n_iter, fpc = fuzz.cluster.cmeans(
            X_pca.T, c=n_clusters, m=FCM_M,
            error=FCM_ERROR, maxiter=FCM_MAX_ITER, init=None, seed=42
        )
        hard_labels    = np.argmax(u, axis=0)
        max_membership = u.max(axis=0)

        cluster_age = {}
        for c in range(n_clusters):
            mask = hard_labels == c
            cluster_age[c] = int(np.round(y[mask].mean())) if mask.sum() > 0 else 0

        y_pred_fcm = np.array([cluster_age[c] for c in hard_labels])
        acc_fcm    = np.mean(y_pred_fcm == y)
        cm_fcm     = compute_confusion_matrix(y, y_pred_fcm)
        met_fcm    = classification_metrics(cm_fcm)

        fcm_res[n_clusters] = {
            "cntr": cntr, "u": u, "hard_labels": hard_labels,
            "max_membership": max_membership, "fpc": fpc,
            "cluster_age": cluster_age, "y_pred": y_pred_fcm,
            "accuracy": acc_fcm, "metrics": met_fcm, "cm": cm_fcm,
        }
        print(f"    k={n_clusters} | FPC={fpc:.4f} | "
              f"Acc={acc_fcm:.4f} | F1={met_fcm['F1 Score']:.4f} | "
              f"Mean membership={max_membership.mean():.4f}")
    print()
    return fcm_res

fcm_results_global = run_fcm(X_global_pca[:, :2], "Global PCA (log1p)")
fcm_results_cv     = run_fcm(X_pca_full,           "CV-best PCA")

best_fcm_k_global = max(fcm_results_global, key=lambda k: fcm_results_global[k]["accuracy"])
best_fcm_k_cv     = max(fcm_results_cv,     key=lambda k: fcm_results_cv[k]["accuracy"])
best_fcm_global   = fcm_results_global[best_fcm_k_global]
best_fcm_cv       = fcm_results_cv[best_fcm_k_cv]

print(f"  Best FCM (global PCA) : k={best_fcm_k_global} | "
      f"Acc={best_fcm_global['accuracy']:.4f} | FPC={best_fcm_global['fpc']:.4f}")
print(f"  Best FCM (CV PCA)     : k={best_fcm_k_cv} | "
      f"Acc={best_fcm_cv['accuracy']:.4f} | FPC={best_fcm_cv['fpc']:.4f}")
print()

print("  COMPARISON: Fuzzy C-Means vs KNN")
print(f"  {'Method':<35} {'Accuracy':>10} {'F1 Score':>10} "
      f"{'Sensitivity':>12} {'Specificity':>12}")
print("  " + "-" * 82)
print(f"  {'KNN (binary age)':<35} "
      f"{metrics['Accuracy']:>10.4f} {metrics['F1 Score']:>10.4f} "
      f"{metrics['Sensitivity']:>12.4f} {metrics['Specificity']:>12.4f}")
print(f"  {'FCM — Global PCA (k=' + str(best_fcm_k_global) + ')':<35} "
      f"{best_fcm_global['accuracy']:>10.4f} "
      f"{best_fcm_global['metrics']['F1 Score']:>10.4f} "
      f"{best_fcm_global['metrics']['Sensitivity']:>12.4f} "
      f"{best_fcm_global['metrics']['Specificity']:>12.4f}")
print(f"  {'FCM — CV PCA (k=' + str(best_fcm_k_cv) + ')':<35} "
      f"{best_fcm_cv['accuracy']:>10.4f} "
      f"{best_fcm_cv['metrics']['F1 Score']:>10.4f} "
      f"{best_fcm_cv['metrics']['Sensitivity']:>12.4f} "
      f"{best_fcm_cv['metrics']['Specificity']:>12.4f}")
print()

best_overall = max(
    [("KNN", metrics["Accuracy"]),
     (f"FCM Global k={best_fcm_k_global}", best_fcm_global["accuracy"]),
     (f"FCM CV k={best_fcm_k_cv}",         best_fcm_cv["accuracy"])],
    key=lambda x: x[1]
)
print(f"  → Best overall: {best_overall[0]} (Acc={best_overall[1]:.4f})")
print()

comparison_rows = [{
    "Method":      f"KNN (PCA={best_key[0]}, K={best_key[1]})",
    "Accuracy":    metrics["Accuracy"],
    "F1_Score":    metrics["F1 Score"],
    "Sensitivity": metrics["Sensitivity"],
    "Specificity": metrics["Specificity"],
    "Notes":       "Cross-validated"
}]
for label, fcm_res_dict in [("Global_PCA", fcm_results_global),
                              ("CV_PCA",     fcm_results_cv)]:
    for k, res in fcm_res_dict.items():
        comparison_rows.append({
            "Method":      f"FCM_{label}_k{k}",
            "Accuracy":    round(res["accuracy"], 4),
            "F1_Score":    round(res["metrics"]["F1 Score"], 4),
            "Sensitivity": round(res["metrics"]["Sensitivity"], 4),
            "Specificity": round(res["metrics"]["Specificity"], 4),
            "Notes":       f"FPC={res['fpc']:.4f}, full dataset (no CV), {label}"
        })
pd.DataFrame(comparison_rows).to_csv(
    OUTPUT_DIR / "fcm_vs_knn_comparison.csv", index=False
)
print("  CSV saved: fcm_vs_knn_comparison.csv")
print()

# ---- FCM Visualization ----
fig_fcm, axes = plt.subplots(2, 3, figsize=(18, 10))
fig_fcm.suptitle(
    "Fuzzy C-Means Analysis — Global PCA vs CV PCA Space (Binary Age)",
    fontsize=13, fontweight="bold"
)

def plot_fcm_row(axes_row, fcm_res_dict, best_k, X_pca, row_label):
    fpcs = [fcm_res_dict[k]["fpc"]      for k in N_CLUSTERS_OPTIONS]
    accs = [fcm_res_dict[k]["accuracy"] for k in N_CLUSTERS_OPTIONS]

    ax1      = axes_row[0]
    ax1_twin = ax1.twinx()
    ax1.plot(N_CLUSTERS_OPTIONS, fpcs, marker="o", color="#4A7FC1",
             linewidth=2, markersize=7, label="FPC")
    ax1_twin.plot(N_CLUSTERS_OPTIONS, accs, marker="s", color="#E87D7D",
                  linewidth=2, markersize=7, linestyle="--", label="Accuracy")
    ax1_twin.axhline(metrics["Accuracy"], color="gray", linestyle=":",
                     linewidth=1.5, label=f"KNN={metrics['Accuracy']:.3f}")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=9)
    ax1.set_ylabel("FPC", fontsize=9, color="#4A7FC1")
    ax1.tick_params(axis="y", labelcolor="#4A7FC1", labelsize=8)
    ax1_twin.set_ylabel("Accuracy", fontsize=9, color="#E87D7D")
    ax1_twin.tick_params(axis="y", labelcolor="#E87D7D", labelsize=8)
    ax1.set_title(f"{row_label}\nFPC & Accuracy vs k", fontsize=10, fontweight="bold")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=7)

    ax2              = axes_row[1]
    best_res         = fcm_res_dict[best_k]
    hard_labels_best = best_res["hard_labels"]
    cmap_c           = plt.cm.get_cmap("tab10", best_k)
    for c in range(best_k):
        mask    = hard_labels_best == c
        n_older = y[mask].sum()
        n_young = (y[mask] == 0).sum()
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=cmap_c(c), alpha=0.7, s=40,
                    label=f"C{c+1} (Old:{n_older} Yng:{n_young})")
    ax2.set_xlim(X_pca[:, 0].mean() - 3*X_pca[:, 0].std(),
                 X_pca[:, 0].mean() + 3*X_pca[:, 0].std())
    ax2.set_ylim(X_pca[:, 1].mean() - 3*X_pca[:, 1].std(),
                 X_pca[:, 1].mean() + 3*X_pca[:, 1].std())
    ax2.set_xlabel("PC1", fontsize=9)
    ax2.set_ylabel("PC2", fontsize=9)
    ax2.set_title(f"FCM Clusters (k={best_k})", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=8)

    ax3      = axes_row[2]
    u_best   = best_res["u"]
    sort_idx = np.lexsort((np.argmax(u_best, axis=0), y))
    im3      = ax3.imshow(u_best[:, sort_idx], aspect="auto",
                          cmap="YlOrRd", vmin=0, vmax=1)
    ax3.set_xlabel("Samples (sorted by age then cluster)", fontsize=8)
    ax3.set_ylabel("Cluster", fontsize=9)
    ax3.set_yticks(range(best_k))
    ax3.set_yticklabels([f"C{c+1}" for c in range(best_k)], fontsize=8)
    ax3.set_title("Membership Matrix\n(brighter = stronger)",
                  fontsize=10, fontweight="bold")
    n_young_sorted = (y[sort_idx] == 0).sum()
    ax3.axvline(n_young_sorted - 0.5, color="blue", linewidth=2, linestyle="--")
    ax3.text(n_young_sorted / 2, -0.7, "Yng", ha="center", fontsize=8, color="blue")
    ax3.text(n_young_sorted + (len(y)-n_young_sorted)/2, -0.7,
             "Old", ha="center", fontsize=8, color="blue")
    plt.colorbar(im3, ax=ax3, label="Membership")

plot_fcm_row(axes[0], fcm_results_global, best_fcm_k_global,
             X_global_pca[:, :2], "Global PCA (log1p)")
plot_fcm_row(axes[1], fcm_results_cv, best_fcm_k_cv,
             X_pca_full, "CV-Best PCA")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fcm_vs_knn_analysis.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: fcm_vs_knn_analysis.png")
print()

# ================================================================================
# SECTION 9: VISUALIZATION PANEL
# ================================================================================
fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    f"Cytokines-Only PCA+KNN (Binary Age) — Best Config: PCA={best_key[0]}, K={best_key[1]}\n"
    f"Mean CV Accuracy = {best_result['mean_acc']:.3f} ± {best_result['std_acc']:.3f}",
    fontsize=14, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# Panel A: Scree plot
ax_a = fig.add_subplot(gs[0, 0])
x_pcs = range(1, len(explained_ratio_global) + 1)
ax_a.bar(x_pcs, explained_ratio_global * 100, color=COLOR_BAR, alpha=0.7)
ax_a2 = ax_a.twinx()
ax_a2.plot(x_pcs, cumulative_var_global * 100, color="darkred",
           marker="o", markersize=4, linewidth=1.5)
ax_a2.axhline(80, color="gray", linestyle="--", linewidth=0.8,
              alpha=0.7, label="80%")
ax_a2.axhline(90, color="gray", linestyle=":",  linewidth=0.8,
              alpha=0.7, label="90%")
ax_a2.set_ylabel("Cumulative Variance (%)", fontsize=9, color="darkred")
ax_a2.tick_params(axis="y", labelcolor="darkred", labelsize=8)
ax_a2.set_ylim(0, 105)
ax_a2.legend(fontsize=7, loc="center right")
ax_a.set_xlabel("Principal Component", fontsize=9)
ax_a.set_ylabel("Explained Variance (%)", fontsize=9)
ax_a.set_title("A: Scree Plot (log1p cytokines)", fontsize=10, fontweight="bold")
ax_a.tick_params(labelsize=8)
ax_a.text(n_pcs_80 + 0.2, explained_ratio_global[n_pcs_80-1]*100 + 1,
          f"{n_pcs_80} PCs→80%", fontsize=7, color="gray")

# Panel B: Grid search heatmap
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
ax_b.set_yticklabels([f"{p} PCs" for p in PCA_OPTIONS], fontsize=8)
ax_b.set_title("B: CV Accuracy Grid Search", fontsize=10, fontweight="bold")
cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
cbar.set_label("Mean Accuracy", fontsize=9)
cbar.ax.tick_params(labelsize=8)
best_i = PCA_OPTIONS.index(best_key[0])
best_j = K_OPTIONS.index(best_key[1])
ax_b.add_patch(plt.Rectangle((best_j-0.5, best_i-0.5), 1, 1,
               fill=False, edgecolor="blue", linewidth=2.5))
ax_b.text(best_j, best_i, "★", ha="center", va="center",
          fontsize=12, color="blue", fontweight="bold")

# Panel C: Confusion matrix
ax_c = fig.add_subplot(gs[0, 2])
ax_c.imshow(cm_best, cmap="Blues")
ax_c.set_xticks([0, 1]);  ax_c.set_xticklabels(["Pred Young", "Pred Older"], fontsize=9)
ax_c.set_yticks([0, 1]);  ax_c.set_yticklabels(["Act Young",  "Act Older"],  fontsize=9)
ax_c.set_title(f"C: Confusion Matrix\nAcc={metrics['Accuracy']:.3f}  F1={metrics['F1 Score']:.3f}",
               fontsize=10, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax_c.text(j, i, str(cm_best[i, j]), ha="center", va="center",
                  fontsize=16, fontweight="bold",
                  color="white" if cm_best[i, j] > cm_best.max() / 2 else "black")

# Panel D: Global PCA scatter colored by age group
ax_d = fig.add_subplot(gs[1, 0])
for age_label, age_val, color in [("Young", 0, COLOR_YOUNG),
                                   ("Older", 1, COLOR_OLDER)]:
    mask = y == age_val
    ax_d.scatter(X_global_pca[mask, 0], X_global_pca[mask, 1],
                 c=color, alpha=0.7, s=40, label=age_label)
ax_d.set_xlabel(f"PC1 ({explained_ratio_global[0]*100:.1f}%) — Global Inflammation",
                fontsize=9)
ax_d.set_ylabel(f"PC2 ({explained_ratio_global[1]*100:.1f}%) — Th17/Regulatory",
                fontsize=9)
ax_d.set_title("D: Global PCA — PC1 vs PC2 by Age Group",
               fontsize=10, fontweight="bold")
ax_d.legend(fontsize=8)
ax_d.tick_params(labelsize=8)

# Panel E: Feature importance
ax_e = fig.add_subplot(gs[1, 1])
top_n   = 10
top_idx = ranking_idx[:top_n]
ax_e.barh(range(top_n), mean_importance[top_idx][::-1],
          xerr=std_importance[top_idx][::-1],
          color=COLOR_BAR, alpha=0.8, edgecolor="white", capsize=3)
ax_e.set_yticks(range(top_n))
ax_e.set_yticklabels([CYTO_COLS[i] for i in top_idx[::-1]], fontsize=8)
ax_e.set_xlabel("Importance (variance-weighted loading)", fontsize=9)
ax_e.set_title("E: Cytokine Feature Importance", fontsize=10, fontweight="bold")
ax_e.tick_params(labelsize=8)

# Panel F: Per-fold accuracy
ax_f = fig.add_subplot(gs[1, 2])
fold_acc_vals = best_result["fold_accs"]
fold_labels   = [f"Fold {i+1}" for i in range(N_SPLITS)]
bar_colors    = [COLOR_BAR if a >= best_result["mean_acc"] else "#AAAAAA"
                 for a in fold_acc_vals]
ax_f.bar(fold_labels, fold_acc_vals, color=bar_colors, alpha=0.85, edgecolor="white")
ax_f.axhline(best_result["mean_acc"], color="darkred", linestyle="--",
             linewidth=1.5, label=f"Mean={best_result['mean_acc']:.3f}")
ax_f.fill_between(
    range(N_SPLITS),
    best_result["mean_acc"] - best_result["std_acc"],
    best_result["mean_acc"] + best_result["std_acc"],
    alpha=0.15, color="darkred", label=f"±Std={best_result['std_acc']:.3f}"
)
ax_f.set_ylabel("Accuracy", fontsize=9)
ax_f.set_title(f"F: Per-Fold Accuracy (PCA={best_key[0]}, K={best_key[1]})",
               fontsize=10, fontweight="bold")
ax_f.legend(fontsize=8)
ax_f.set_ylim(0, 1.05)
ax_f.tick_params(labelsize=8)
for i, v in enumerate(fold_acc_vals):
    ax_f.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

plt.savefig(FIGURES_DIR / "pca_knn_age_binary_results.png",
            dpi=150, bbox_inches="tight", facecolor="white")
fig.canvas.draw()
save_panel(fig, ax_a, "panel_A_scree.png")
save_panel(fig, ax_b, "panel_B_heatmap.png")
save_panel(fig, ax_c, "panel_C_confusion.png")
save_panel(fig, ax_d, "panel_D_pca_scatter.png")
save_panel(fig, ax_e, "panel_E_importance.png")
plt.close()
print("  Figures saved to:", FIGURES_DIR)
print()

# ================================================================================
# SECTION 10: EXPORT
# ================================================================================
X_proc_mean = X_raw.mean(axis=0)
X_proc_std  = X_raw.std(axis=0);  X_proc_std[X_proc_std == 0] = 1
X_processed = (X_raw - X_proc_mean) / X_proc_std

components_full, var_ratio_full, mean_full = pca_fit(X_processed, best_key[0])
X_pca_best = pca_transform(X_processed, mean_full, components_full)

np.save(OUTPUT_DIR / "X_processed.npy",    X_processed)
np.save(OUTPUT_DIR / "X_pca_best.npy",     X_pca_best)
np.save(OUTPUT_DIR / "y_labels.npy",       y)
np.save(OUTPUT_DIR / "pca_components.npy", components_full)

print("=" * 65)
print("PIPELINE COMPLETE  (cytokines only — binary age)")
print("=" * 65)
print(f"  Best config  : PCA={best_key[0]}, K={best_key[1]}")
print(f"  Accuracy     : {metrics['Accuracy']:.4f}")
print(f"  Balanced Acc : {metrics['Balanced Accuracy']:.4f}")
print(f"  Sensitivity  : {metrics['Sensitivity']:.4f}  (Older detection)")
print(f"  Specificity  : {metrics['Specificity']:.4f}  (Young detection)")
print(f"  F1 Score     : {metrics['F1 Score']:.4f}")
print(f"  Top cytokine : {importance_df.iloc[0]['Cytokine']} "
      f"(importance={importance_df.iloc[0]['Importance']:.4f})")
print(f"  Data outputs : {OUTPUT_DIR}")
print(f"  Figures      : {FIGURES_DIR}")
print("=" * 65)