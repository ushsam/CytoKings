"""
================================================================================
Cytokine-Only PCA + KNN Sex Classification Pipeline
================================================================================
Outputs → PCA+KNN/Outputs/cytokines_only/
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
DATA_DIR    = BASE_DIR / "Data"
OUTPUT_DIR  = BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokines_only"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# SECTION 1: DATA LOADING
# ================================================================================
df = pd.read_csv(DATA_DIR / "analysis_merged_subject_level.csv")

CYTO_COLS = [
    "IFN-gamma", "IL-12p70", "IL-13", "IL-1beta", "IL-2", "IL-4", "IL-5",
    "IL-6", "TNF-alpha", "GM-CSF", "IL-18", "IL-10", "IL-17A", "IL-21",
    "IL-22", "IL-23", "IL-27", "IL-9"
]

X_raw     = df[CYTO_COLS].values.astype(float)
y         = (df["SEXC"] == "Male").astype(int).values   # 1=Male, 0=Female
n_samples = len(df)
n_missing = np.isnan(X_raw).sum()

print("=" * 65)
print("SECTION 1: DATASET SUMMARY  (cytokines only)")
print("=" * 65)
print(f"  Total subjects : {n_samples}")
print(f"  Cytokines      : {len(CYTO_COLS)}")
print(f"  Males          : {y.sum()}  ({100*y.sum()/n_samples:.1f}%)")
print(f"  Females        : {(y==0).sum()}  ({100*(y==0).sum()/n_samples:.1f}%)")
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
    """Save a single panel from the figure by extracting its bounding box."""
    fig.canvas.draw()
    extent = ax.get_tightbbox(fig.canvas.get_renderer())
    extent = extent.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(FIGURES_DIR / filename, bbox_inches=extent.expanded(pad, pad),
                dpi=150, facecolor="white")
    print(f"  Panel saved: {filename}")


# ================================================================================
# SECTION 3: GLOBAL PCA
# ================================================================================
X_global_mean   = X_raw.mean(axis=0)
X_global_std    = X_raw.std(axis=0);  X_global_std[X_global_std == 0] = 1
X_global_scaled = (X_raw - X_global_mean) / X_global_std
X_centered_global = X_global_scaled - X_global_scaled.mean(axis=0)
X_centered_global = np.nan_to_num(X_centered_global, nan=0.0, posinf=0.0, neginf=0.0)

_, S_global, Vt_global  = np.linalg.svd(X_centered_global, full_matrices=False)
eigenvalues_global      = (S_global ** 2) / (n_samples - 1)
explained_ratio_global  = eigenvalues_global / eigenvalues_global.sum()
cumulative_var_global   = np.cumsum(explained_ratio_global)

n_pcs_80 = int(np.searchsorted(cumulative_var_global, 0.80)) + 1
n_pcs_90 = int(np.searchsorted(cumulative_var_global, 0.90)) + 1

print("=" * 65)
print("SECTION 3: GLOBAL PCA — INTRINSIC DIMENSIONALITY")
print("=" * 65)
print(f"  PC1 explains : {explained_ratio_global[0]*100:.1f}%")
print(f"  PC2 explains : {explained_ratio_global[1]*100:.1f}%")
print(f"  PCs for 80%  : {n_pcs_80}")
print(f"  PCs for 90%  : {n_pcs_90}")
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
pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "results_summary.csv", index=False)

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
second_key  = ranked[1][0]
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
importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

print("=" * 65)
print("SECTION 8: FEATURE IMPORTANCE — Top 10 cytokines")
print("=" * 65)
for rank, i in enumerate(ranking_idx[:10]):
    bar = "█" * int(mean_importance[i] / mean_importance[ranking_idx[0]] * 20)
    print(f"  {rank+1:>2}. {CYTO_COLS[i]:<12} "
          f"{mean_importance[i]:.4f} ± {std_importance[i]:.4f}  {bar}")
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

X_log = np.log1p(X_raw)

male_means   = X_log[male_idx].mean(axis=0)
female_means = X_log[female_idx].mean(axis=0)
male_std     = X_log[male_idx].std(axis=0)
female_std   = X_log[female_idx].std(axis=0)
mean_diff    = male_means - female_means

# Welch's t-test per cytokine (unequal variance, safer for small n)
pvals  = []
tstats = []
for i in range(len(CYTO_COLS)):
    t, p = stats.ttest_ind(
        X_log[male_idx, i],
        X_log[female_idx, i],
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

# ---- Plot ----
order = sex_comparison_df.index.tolist()
cyto_ordered = sex_comparison_df["Cytokine"].tolist()

fig_sex, ax_sex = plt.subplots(figsize=(12, 6))
x     = np.arange(len(CYTO_COLS))
width = 0.35

bars_m = ax_sex.bar(x - width/2,
                    sex_comparison_df["Male_Mean"],   width,
                    label="Male",   color="#6BAED6", alpha=0.85,
                    edgecolor="white",
                    yerr=sex_comparison_df["Male_Std"],   capsize=3)
bars_f = ax_sex.bar(x + width/2,
                    sex_comparison_df["Female_Mean"], width,
                    label="Female", color="#E87D7D", alpha=0.85,
                    edgecolor="white",
                    yerr=sex_comparison_df["Female_Std"], capsize=3)

# Mark significant cytokines with a star above the bars
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
print("  Figure saved: cytokine_sex_comparison.png")
print("  CSV saved:    cytokine_sex_comparison.csv")
print()

# ================================================================================
# SECTION 9: VISUALIZATION PANEL
# ================================================================================
COLOR_FEMALE = "#E87D7D"
COLOR_MALE   = "#6BAED6"
COLOR_BAR    = "#4A7FC1"

fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    f"Cytokines-Only PCA+KNN — Best Config: PCA={best_key[0]}, K={best_key[1]}\n"
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
ax_a2.axhline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax_a2.axhline(90, color="gray", linestyle=":",  linewidth=0.8, alpha=0.7)
ax_a2.set_ylabel("Cumulative Variance (%)", fontsize=9, color="darkred")
ax_a2.tick_params(axis="y", labelcolor="darkred", labelsize=8)
ax_a2.set_ylim(0, 105)
ax_a.set_xlabel("Principal Component", fontsize=9)
ax_a.set_ylabel("Explained Variance (%)", fontsize=9)
ax_a.set_title("A: Scree Plot (Cytokines)", fontsize=10, fontweight="bold")
ax_a.tick_params(labelsize=8)

# Panel B: Grid search heatmap
# ---- Panel B: Grid search heatmap ----
ax_b = fig.add_subplot(gs[0, 1])

heatmap_data = np.array([
    [results[(n_pca, k)]["mean_acc"] for k in K_OPTIONS]
    for n_pca in PCA_OPTIONS
])

im = ax_b.imshow(
    heatmap_data,
    cmap="YlOrRd",
    aspect="auto",
    vmin=heatmap_data.min(),
    vmax=heatmap_data.max()
)

ax_b.set_xticks(range(len(K_OPTIONS)))
ax_b.set_xticklabels([f"K={k}" for k in K_OPTIONS], fontsize=8)

ax_b.set_yticks(range(len(PCA_OPTIONS)))
ax_b.set_yticklabels([f"{p} PCs/mod" for p in PCA_OPTIONS], fontsize=8)

ax_b.set_title("B: CV Accuracy Grid Search", fontsize=10, fontweight="bold")

# =========================
# FIX: COLORBAR WITH NUMBERS
# =========================
cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
cbar.set_label("Mean Accuracy", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# optional: force nice tick formatting (0.01 steps etc. not needed usually)
# cbar.set_ticks(np.linspace(heatmap_data.min(), heatmap_data.max(), 5))

best_i = PCA_OPTIONS.index(best_key[0])
best_j = K_OPTIONS.index(best_key[1])

ax_b.add_patch(plt.Rectangle(
    (best_j - 0.5, best_i - 0.5),
    1, 1,
    fill=False,
    edgecolor="blue",
    linewidth=2.5
))

ax_b.text(
    best_j, best_i, "★",
    ha="center", va="center",
    fontsize=12,
    color="blue",
    fontweight="bold"
)

# Panel C: Confusion matrix
ax_c = fig.add_subplot(gs[0, 2])
im_c = ax_c.imshow(cm_best, cmap="Blues")
ax_c.set_xticks([0, 1]);  ax_c.set_xticklabels(["Pred Female", "Pred Male"],  fontsize=9)
ax_c.set_yticks([0, 1]);  ax_c.set_yticklabels(["Act Female",  "Act Male"],   fontsize=9)
ax_c.set_title(f"C: Confusion Matrix\nAcc={metrics['Accuracy']:.3f}  F1={metrics['F1 Score']:.3f}",
               fontsize=10, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax_c.text(j, i, str(cm_best[i, j]), ha="center", va="center",
                  fontsize=16, fontweight="bold",
                  color="white" if cm_best[i, j] > cm_best.max() / 2 else "black")

# Panel D: PCA scatter colored by sex
ax_d = fig.add_subplot(gs[1, 0])
X_train_pca_d, X_test_pca_d, y_train_d, y_test_d = \
    best_result["fold_projections"][-1]
for subset, label, color, marker in [
    (X_train_pca_d[y_train_d == 0], "Female (train)", COLOR_FEMALE, "o"),
    (X_train_pca_d[y_train_d == 1], "Male (train)",   COLOR_MALE,   "o"),
    (X_test_pca_d[ y_test_d  == 0], "Female (test)",  COLOR_FEMALE, "^"),
    (X_test_pca_d[ y_test_d  == 1], "Male (test)",    COLOR_MALE,   "^"),
]:
    if len(subset) > 0:
        ax_d.scatter(subset[:, 0], subset[:, 1], c=color, marker=marker,
                     alpha=0.7, s=40, label=label)
ax_d.set_xlabel("PC1", fontsize=9)
ax_d.set_ylabel("PC2", fontsize=9)
ax_d.set_title(f"D: PCA Space — Sex (PCA={best_key[0]}, K={best_key[1]})",
               fontsize=10, fontweight="bold")
ax_d.legend(fontsize=7, ncol=2)
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

# Save full panel figure
plt.savefig(FIGURES_DIR / "pca_knn_cytokines_results.png",
            dpi=150, bbox_inches="tight", facecolor="white")

# Save individual panels
fig.canvas.draw()
save_panel(fig, ax_a,  "panel_A_scree.png")
save_panel(fig, ax_b,  "panel_b_heatmap.png")
save_panel(fig, ax_c,  "panel_C_confusion.png")
save_panel(fig, ax_d,  "panel_D_pca_scatter.png")
save_panel(fig, ax_e,  "panel_E_importance.png")

plt.close()
print("  Figures saved to:", FIGURES_DIR)
print()

# ================================================================================
# SECTION 10: EXPORT  (data files → OUTPUT_DIR)
# ================================================================================
X_proc_mean = X_raw.mean(axis=0)
X_proc_std  = X_raw.std(axis=0);  X_proc_std[X_proc_std == 0] = 1
X_processed = (X_raw - X_proc_mean) / X_proc_std

best_n_pca      = best_key[0]
components_full, var_ratio_full, mean_full = pca_fit(X_processed, best_n_pca)
X_pca_best      = pca_transform(X_processed, mean_full, components_full)

np.save(OUTPUT_DIR / "X_processed.npy",    X_processed)
np.save(OUTPUT_DIR / "X_pca_best.npy",     X_pca_best)
np.save(OUTPUT_DIR / "y_labels.npy",       y)
np.save(OUTPUT_DIR / "pca_components.npy", components_full)

print("=" * 65)
print("PIPELINE COMPLETE  (cytokines only)")
print("=" * 65)
print(f"  Best config  : PCA={best_key[0]}, K={best_key[1]}")
print(f"  Accuracy     : {metrics['Accuracy']:.4f}")
print(f"  Balanced Acc : {metrics['Balanced Accuracy']:.4f}")
print(f"  Sensitivity  : {metrics['Sensitivity']:.4f}  (Male detection)")
print(f"  Specificity  : {metrics['Specificity']:.4f}  (Female detection)")
print(f"  F1 Score     : {metrics['F1 Score']:.4f}")
print(f"  Top cytokine : {importance_df.iloc[0]['Cytokine']} "
      f"(importance={importance_df.iloc[0]['Importance']:.4f})")
print(f"  Data outputs : {OUTPUT_DIR}")
print(f"  Figures      : {FIGURES_DIR}")
print("=" * 65)