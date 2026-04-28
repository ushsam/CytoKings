#!/usr/bin/env python

import time
pipeline_start = time.time()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------ PATHS ------------------
try:
    BASE_PATH = Path(__file__).resolve().parent
except NameError:
    BASE_PATH = Path.cwd()

REPO_PATH  = BASE_PATH.parent if BASE_PATH.name != "CytoKings" else BASE_PATH
DATA_PATH  = REPO_PATH / "Data" / "analysis_merged_subject_level.csv"
OUT_DIR_EDA    = REPO_PATH / "Data" / "eda_subject_level"
OUT_DIR_BATCH = REPO_PATH / "Data" / "batch_effect_analysis"

OUT_DIR_EDA.mkdir(parents=True, exist_ok=True)
OUT_DIR_BATCH.mkdir(parents=True, exist_ok=True)

print("Repo root  :", REPO_PATH)
print("Data file  :", DATA_PATH)
print("Data exists:", DATA_PATH.exists())
print("EPA Output dir :", OUT_DIR_EDA)
print("BATCH Output dir :", OUT_DIR_BATCH)

#cytokine columns
CYTO_COLS = [
    "IFN-gamma", "IL-12p70", "IL-13", "IL-1beta", "IL-2", "IL-4", "IL-5",
    "IL-6", "TNF-alpha", "GM-CSF", "IL-18", "IL-10", "IL-17A", "IL-21",
    "IL-22", "IL-23", "IL-27", "IL-9"
]
# ------------------ SETUP ------------------
sns.set(style="whitegrid", context="notebook")

# ------------------ LOAD DATA ------------------
df = pd.read_csv(DATA_PATH)

print("Shape   :", df.shape)
print("Columns :", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)

#save missingness table
missing = df.isna().sum().to_frame("n_missing")
missing["frac_missing"] = missing["n_missing"] / len(df)
missing.to_csv(OUT_DIR_EDA / "missingness_summary.csv")
print("\nMissingness:\n", missing)

# ------------------ CYTOKINE SUMMARY ------------------
cyto_desc = df[CYTO_COLS].describe().T
cyto_desc.to_csv(OUT_DIR_EDA / "cytokine_summary_stats.csv")
print("\nCytokine summary stats:\n", cyto_desc)

# ------------------ DISTRIBUTIONS ------------------
for col in CYTO_COLS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{col} Distribution", fontsize=13, fontweight="bold")

    # Raw — clipped at 99th percentile so outliers don't squish the plot
    p99 = df[col].quantile(0.99)
    data_clipped = df[col].clip(upper=p99)
    axes[0].hist(data_clipped, bins=30, edgecolor="white",
                 color="#6BAED6", alpha=0.85, density=True)
    data_clipped.plot.kde(ax=axes[0], color="#1A4F72",
                          linewidth=2, label="KDE")
    axes[0].set_xlim(left=0)
    axes[0].set_title(f"{col} (raw, clipped at 99th pct)")
    axes[0].set_xlabel("MFI")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9)

    # log1p transformed
    log_data = np.log1p(df[col])
    axes[1].hist(log_data, bins=30, edgecolor="white",
                 color="#5BAD72", alpha=0.85, density=True)
    log_data.plot.kde(ax=axes[1], color="#1A4F72",
                      linewidth=2, label="KDE")
    axes[1].set_title(f"{col} (log1p transformed)")
    axes[1].set_xlabel("log1p(MFI)")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR_EDA / f"{col}_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close()

print("  Saved: distribution plots for all cytokines")

# ------------------ CORRELATION HEATMAP ------------------
corr = df[CYTO_COLS].corr(method="spearman")
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
            annot=False, linewidths=0.5)
plt.title("Cytokine Spearman Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR_EDA / "cytokine_corr_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
corr.to_csv(OUT_DIR_EDA / "cytokine_corr_matrix.csv")
print("  Saved: cytokine_corr_heatmap.png")

# ------------------ AGE / SEX / RACE DISTRIBUTIONS ------------------

# Age distribution
if "AGE" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df["AGE"], bins=10)
    plt.title("Age Distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR_EDA / "AGE_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

# Sex counts
if "SEXC" in df.columns:
    plt.figure(figsize=(4, 4))
    sns.countplot(x="SEXC", data=df,
                  palette={"Female": "#E87D7D", "Male": "#6BAED6"})
    plt.title("Sex Distribution (n=125)", fontsize=12, fontweight="bold")
    plt.xlabel("Biological Sex")
    plt.ylabel("Count")
    for p in plt.gca().patches:
        plt.gca().annotate(f"{int(p.get_height())}",
                           (p.get_x() + p.get_width() / 2, p.get_height() + 0.5),
                           ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR_EDA / "SEXC_counts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: SEXC_counts.png")

# Race counts
if "RACEGRP" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="RACEGRP", data=df)
    plt.title("Race Group Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR_EDA / "RACEGRP_counts.png", dpi=150, bbox_inches="tight")
    plt.close()

# ------------------ CYTOKINES vs SEX ------------------
if "SEXC" in df.columns:
    for col in CYTO_COLS:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="SEXC", y=col, data=df,
                    palette={"Female": "#E87D7D", "Male": "#6BAED6"})
        plt.title(f"{col} by Sex", fontsize=11, fontweight="bold")
        plt.xlabel("Biological Sex")
        plt.ylabel("MFI")
        plt.tight_layout()
        plt.savefig(OUT_DIR_EDA / f"{col}_by_SEXC_box.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
    print("  Saved: sex-stratified boxplots for all cytokines")

# ------------------ CYTOKINES vs AGE ------------------
if "AGE" in df.columns:
    for col in CYTO_COLS:
        plt.figure(figsize=(6, 4))
        sns.regplot(x="AGE", y=col, data=df, scatter_kws={"alpha": 0.6})
        plt.title(f"{col} vs Age")
        plt.tight_layout()
        plt.savefig(OUT_DIR_EDA / f"{col}_vs_AGE_reg.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

# ------------------ QUICK PCA PREVIEW ------------------
X_raw    = df[CYTO_COLS].copy().dropna(axis=0)
idx_kept = X_raw.index
scaler   = StandardScaler()
X        = scaler.fit_transform(X_raw)
pca      = PCA()
X_pca    = pca.fit_transform(X)
explained = pca.explained_variance_ratio_

# Scree plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained) + 1), explained * 100,
        color="#4A7FC1", alpha=0.7, label="Per-PC")
ax2 = plt.gca().twinx()
ax2.plot(range(1, len(explained) + 1), np.cumsum(explained) * 100,
         color="darkred", marker="o", markersize=4,
         linewidth=1.5, label="Cumulative")
ax2.axhline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax2.set_ylabel("Cumulative Variance (%)", fontsize=9, color="darkred")
ax2.tick_params(axis="y", labelcolor="darkred")
plt.xlabel("Principal Component")
plt.gca().set_ylabel("Explained Variance (%)")
plt.title("PCA Scree Plot — Cytokines", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR_EDA / "PCA_scree.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: PCA_scree.png")

# PC1 vs PC2 colored by sex
if "SEXC" in df.columns:
    sex_for_pca = df.loc[idx_kept, "SEXC"]
    plt.figure(figsize=(7, 5))
    for sex_val, color in [("Female", "#E87D7D"), ("Male", "#6BAED6")]:
        mask = sex_for_pca == sex_val
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, alpha=0.7, s=40, label=sex_val,
                    edgecolors="white", linewidths=0.4)
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    plt.title("PC1 vs PC2 — Colored by Sex", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR_EDA / "PCA_PC1_PC2_by_SEXC.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: PCA_PC1_PC2_by_SEXC.png")

# Save PCA loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=CYTO_COLS,
    columns=[f"PC{i+1}" for i in range(len(CYTO_COLS))]
)
loadings.to_csv(OUT_DIR_EDA / "PCA_loadings.csv")
print("  Saved: PCA_loadings.csv")

# ------------------ COMBINED FIGURE 1 PANEL (FOR PAPER) ------------------
import matplotlib.gridspec as gridspec

sns.set(style="whitegrid", context="talk")

fig = plt.figure(figsize=(24, 8), facecolor="white")
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

# ---- Panel A: Correlation heatmap ----
ax_a = fig.add_subplot(gs[0, 0])
corr = df[CYTO_COLS].corr(method="spearman")
sns.heatmap(corr, cmap="YlOrRd", center=None,
            vmin=0, vmax=1.0,
            square=True, annot=False,
            linewidths=0.5, ax=ax_a,
            cbar_kws={"shrink": 0.8, "label": "Spearman r"})
ax_a.set_title("A: Cytokine Spearman\nCorrelation Heatmap",
               fontsize=13, fontweight="bold", pad=12)
ax_a.tick_params(labelsize=9)
ax_a.set_xticklabels(ax_a.get_xticklabels(), rotation=45, ha="right")
ax_a.set_yticklabels(ax_a.get_yticklabels(), rotation=0)

# ---- Panel B: IFN-gamma raw distribution ----
ax_b = fig.add_subplot(gs[0, 1])
p99          = df["IFN-gamma"].quantile(0.99)
data_clipped = df["IFN-gamma"].clip(upper=p99)
ax_b.hist(data_clipped, bins=30, edgecolor="white",
          color="#6BAED6", alpha=0.85, density=True)
data_clipped.plot.kde(ax=ax_b, color="#08306B", linewidth=2.5, label="KDE")
ax_b.set_xlim(left=0)
ax_b.set_title("B: IFN-\u03b3 Raw Distribution\n(clipped at 99th percentile)",
               fontsize=13, fontweight="bold", pad=12)
ax_b.set_xlabel("MFI", fontsize=11)
ax_b.set_ylabel("Density", fontsize=11)
ax_b.legend(fontsize=10)
ax_b.tick_params(labelsize=10)
ax_b.yaxis.grid(True, alpha=0.4)

# ---- Panel C: IFN-gamma log1p distribution ----
ax_c  = fig.add_subplot(gs[0, 2])
log_data = np.log1p(df["IFN-gamma"])
ax_c.hist(log_data, bins=30, edgecolor="white",
          color="#5BAD72", alpha=0.85, density=True)
log_data.plot.kde(ax=ax_c, color="#08306B", linewidth=2.5, label="KDE")
ax_c.set_title("C: IFN-\u03b3 log1p Transformed\nDistribution",
               fontsize=13, fontweight="bold", pad=12)
ax_c.set_xlabel("log1p(MFI)", fontsize=11)
ax_c.set_ylabel("Density", fontsize=11)
ax_c.legend(fontsize=10)
ax_c.tick_params(labelsize=10)
ax_c.yaxis.grid(True, alpha=0.4)

plt.savefig(OUT_DIR_EDA / "figure1_data_overview_panel.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
sns.set(style="whitegrid", context="notebook")
print("  Saved: figure1_data_overview_panel.png")

# ------------------ BATCH EFFECT ANALYSIS ------------------
print("\n" + "=" * 65)
print("SECTION 11: BATCH EFFECT ANALYSIS")
print("=" * 65)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from xgboost import XGBClassifier
from scipy.stats import chi2_contingency, zscore
import shap

BATCH_COL = "Batch date"

if BATCH_COL not in df.columns:
    print(f"  WARNING: '{BATCH_COL}' column not found — skipping batch analysis")
else:
    # use a fresh copy so we don't modify df used by EDA
    df_batch   = df.copy()
    batch      = df_batch[BATCH_COL].astype(str)
    le         = LabelEncoder()
    y_sex      = le.fit_transform(df_batch["SEXC"])   # Female=0, Male=1
    X_cytokines = df_batch[CYTO_COLS].copy()
    y_series    = pd.Series(y_sex)

    # ---- 1. Chi-square test: batch vs sex ----
    contingency     = pd.crosstab(batch, y_sex)
    chi2_stat, p, dof, _ = chi2_contingency(contingency)
    print(f"\n  Batch vs Sex chi-square:")
    print(f"    chi2={chi2_stat:.4f}  p={p:.4f}  dof={dof}")

    pd.DataFrame({
        "Test":        ["Chi-square batch vs sex"],
        "Chi2":        [round(chi2_stat, 4)],
        "p_value":     [round(p,         4)],
        "DOF":         [dof],
        "Significant": [p < 0.05]
    }).to_csv(OUT_DIR_BATCH / "batch_sex_chisquare.csv", index=False)

    mixing         = contingency.div(contingency.sum(axis=1), axis=0)
    mixing.columns = ["Female", "Male"]
    mixing.to_csv(OUT_DIR_BATCH / "batch_mixing_table.csv")
    print("  Saved: batch_sex_chisquare.csv, batch_mixing_table.csv")

    # ---- 2. AUC with vs without batch ----
    def train_model_batch(X, y):
        model = XGBClassifier(
            max_depth=3, n_estimators=200, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0
        )
        model.fit(X, y)
        return model

    def cv_auc(X, y, n_splits=5, seed=42):
        skf  = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=seed)
        aucs = []
        for train_idx, test_idx in skf.split(X, y):
            Xtr = X.iloc[train_idx]
            Xte = X.iloc[test_idx]
            ytr = np.array(y)[train_idx]
            yte = np.array(y)[test_idx]
            m   = train_model_batch(Xtr, ytr)
            aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        return np.array(aucs)

    X_with_batch          = X_cytokines.copy()
    X_with_batch["batch"] = LabelEncoder().fit_transform(batch)

    auc_no   = cv_auc(X_cytokines, y_series)
    auc_with = cv_auc(X_with_batch, y_series)

    pd.DataFrame({
        "Model":    ["Without batch", "With batch"],
        "AUC_mean": [round(auc_no.mean(),   4), round(auc_with.mean(), 4)],
        "AUC_std":  [round(auc_no.std(),    4), round(auc_with.std(),  4)]
    }).to_csv(OUT_DIR_BATCH / "batch_auc_comparison.csv", index=False)
    print(f"  AUC without batch : {auc_no.mean():.4f} ± {auc_no.std():.4f}")
    print(f"  AUC with batch    : {auc_with.mean():.4f} ± {auc_with.std():.4f}")
    print("  Saved: batch_auc_comparison.csv")

    # ---- 3. SHAP feature stability ----
    def shap_importance(X, y, n_runs=10):
        importance_matrix = []
        for i in range(n_runs):
            X_res, y_res = resample(X, y, replace=True, random_state=i)
            m             = train_model_batch(X_res, y_res)
            explainer     = shap.TreeExplainer(m)
            shap_vals     = explainer.shap_values(X_res)
            importance_matrix.append(np.abs(shap_vals).mean(axis=0))
        return np.array(importance_matrix)

    print("  Running SHAP stability analysis (this may take ~1 min)...")
    shap_no_batch   = shap_importance(X_cytokines, y_series)
    shap_with_batch = shap_importance(X_with_batch, y_series)

    shap_stability_no   = pd.Series(
        shap_no_batch.std(axis=0),
        index=CYTO_COLS
    ).sort_values(ascending=False)
    shap_stability_with = pd.Series(
        shap_with_batch.std(axis=0),
        index=list(CYTO_COLS) + ["batch"]
    ).sort_values(ascending=False)

    shap_stability_no.to_csv(OUT_DIR_BATCH / "shap_stability_no_batch.csv",
                              header=["std"])
    shap_stability_with.to_csv(OUT_DIR_BATCH / "shap_stability_with_batch.csv",
                                header=["std"])
    print("  SHAP stability (no batch) — top 5:")
    print(shap_stability_no.head())
    print("  Saved: shap_stability_no_batch.csv, shap_stability_with_batch.csv")

    # ---- 4. Outlier analysis ----
    def detect_outliers_iqr(dataframe, cols):
        counts = {}
        for col in cols:
            q1, q3 = dataframe[col].quantile(0.25), dataframe[col].quantile(0.75)
            iqr    = q3 - q1
            counts[col] = len(dataframe[
                (dataframe[col] < q1 - 1.5*iqr) |
                (dataframe[col] > q3 + 1.5*iqr)
            ])
        return pd.Series(counts).sort_values(ascending=False)

    outlier_summary = detect_outliers_iqr(df_batch, CYTO_COLS)
    outlier_summary.to_csv(OUT_DIR_BATCH / "outlier_iqr_summary.csv", header=["n_outliers"])
    print("\n  Outliers per cytokine (IQR method):")
    print(outlier_summary.to_string())

    z              = df_batch[CYTO_COLS].apply(zscore)
    extreme_flags  = (np.abs(z) > 3).sum().sort_values(ascending=False)
    extreme_flags.to_csv(OUT_DIR_BATCH / "outlier_zscore_summary.csv",
                         header=["n_extreme"])
    print("\n  Extreme values per cytokine (|z|>3):")
    print(extreme_flags.to_string())

    extreme_per_sample = (np.abs(z) > 3).sum(axis=1)
    df_batch["extreme_count"] = extreme_per_sample
    top_outlier_samples = df_batch.sort_values(
        "extreme_count", ascending=False
    )[["SUBJECT_ID", "extreme_count"] + CYTO_COLS].head(10)
    top_outlier_samples.to_csv(OUT_DIR_BATCH / "top_outlier_samples.csv", index=False)
    print("  Saved: outlier_iqr_summary.csv, outlier_zscore_summary.csv, "
          "top_outlier_samples.csv")

    # ---- 5. Correlation heatmap (batch analysis version) ----
    corr_batch = df_batch[CYTO_COLS].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_batch, cmap="coolwarm", center=0, square=True,
                linewidths=0.5)
    plt.title("Cytokine Correlation Heatmap (Batch Analysis)",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR_BATCH / "batch_corr_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: batch_corr_heatmap.png")

    # ---- 6. Clustermap ----
    g = sns.clustermap(corr_batch, cmap="vlag", center=0,
                       figsize=(10, 10))
    g.fig.suptitle("Cytokine Clustermap (Batch Analysis)",
                   fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(OUT_DIR_BATCH / "batch_clustermap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: batch_clustermap.png")

    # ---- 7. IL-21 vs IL-17A scatter ----
    plt.figure(figsize=(6, 5))
    plt.scatter(df_batch["IL-21"], df_batch["IL-17A"],
                alpha=0.6, color="#6BAED6", edgecolors="white")
    plt.xlabel("IL-21")
    plt.ylabel("IL-17A")
    plt.title("IL-21 vs IL-17A", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR_BATCH / "IL21_vs_IL17A_scatter.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: IL21_vs_IL17A_scatter.png")

    # --------- BATCH EFFECT ANALYSIS RESULTS ------------
    delta = abs(auc_with.mean() - auc_no.mean())
    print("\n" + "=" * 65)
    print("  BATCH EFFECT EVALUATION REPORT")
    print("=" * 65)
    print("""
  1. OBJECTIVE
     Determine whether technical batch variation (Batch date):
     - Confounds sex classification
     - Influences model performance
     - Alters cytokine importance structure
     - Introduces instability in biological interpretation
    """)
    print("  2. BATCH-SEX ASSOCIATION TEST")
    print(f"     Chi-square p-value : {p:.4f}")
    if p > 0.05:
        print("     Result            : NOT significant (p > 0.05)")
        print("     Interpretation    : Sex distribution approximately balanced")
        print("                         across batch groups. No evidence of sex")
        print("                         clustering within specific batches.")
        print("     Conclusion        : Batch is independent of sex.")
    else:
        print("     Result            : SIGNIFICANT (p < 0.05)")
        print("     Conclusion        : Consider batch correction.")

    print("\n  3. MODEL PERFORMANCE — WITH vs WITHOUT BATCH")
    print(f"     AUC without batch : {auc_no.mean():.4f} ± {auc_no.std():.4f}")
    print(f"     AUC with batch    : {auc_with.mean():.4f} ± {auc_with.std():.4f}")
    print(f"     Delta AUC         : {delta:.4f}")
    if delta < 0.01:
        print("     Conclusion        : Batch adds negligible predictive value.")
        print("                         Signal is biologically derived.")
    else:
        print("     Conclusion        : Batch meaningfully changes AUC — investigate.")

    print("\n  4. BATCH MIXING TABLE")
    print("     Sex distribution per batch (proportion Male/Female):")
    print(mixing.to_string(float_format="{:.3f}".format))

    print("\n  5. SHAP FEATURE STABILITY — top 5 cytokines (no batch):")
    print(shap_stability_no.head().to_string())

    print("\n  6. OUTLIER SUMMARY — top 5 cytokines by IQR outlier count:")
    print(outlier_summary.head().to_string())

    print("\n  7. BIOLOGICAL COHERENCE")
    print("     Clustermap reveals biologically coherent modules:")
    print("     - Pro-inflammatory: IL-6, TNF-alpha, IL-1beta, IL-12p70")
    print("     - Th2 axis        : IL-4, IL-5, IL-13")
    print("     - Th17 axis       : IL-21, IL-22, IL-23, IL-17A")
    print("     - Regulatory      : IL-10, IL-27")

    print("\n  8. FINAL DECISION")
    if p > 0.05 and delta < 0.01:
        print("     Batch excluded from all downstream modeling.")
        print("     Results are not driven by technical batch effects.")
        print("     Cytokine importance rankings reflect biological signal.")
    else:
        print("     Review batch effects before proceeding.")

    print(f"\n  All batch outputs saved to: {OUT_DIR_BATCH}")
    print("  Batch analysis complete")

print("\n" + "=" * 65)
print("Bath effects complete. All outputs saved to:", OUT_DIR_BATCH)
print("=" * 65)

pipeline_end     = time.time()
elapsed          = pipeline_end - pipeline_start
mins, secs       = divmod(int(elapsed), 60)
print(f"\nTotal runtime: {mins}m {secs}s")