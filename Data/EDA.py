#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.gridspec as gridspec

# ------------------ CONFIG ------------------

try:
    BASE_PATH = Path(__file__).resolve().parent
except NameError:
    BASE_PATH = Path.cwd()

REPO_PATH  = BASE_PATH.parent if BASE_PATH.name != "CytoKings" else BASE_PATH
DATA_PATH  = REPO_PATH / "Data" / "analysis_merged_subject_level.csv"
OUT_DIR    = REPO_PATH / "Data" / "eda_subject_level"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Repo root  :", REPO_PATH)
print("Data file  :", DATA_PATH)
print("Data exists:", DATA_PATH.exists())
print("Output dir :", OUT_DIR)

# Cytokine columns
CYTO_COLS = [
    "IFN-gamma", "IL-12p70", "IL-13", "IL-1beta", "IL-2", "IL-4", "IL-5",
    "IL-6", "TNF-alpha", "GM-CSF", "IL-18", "IL-10", "IL-17A", "IL-21",
    "IL-22", "IL-23", "IL-27", "IL-9"
]

# ------------------ SETUP ------------------

os.makedirs(OUT_DIR, exist_ok=True)

sns.set(style="whitegrid", context="notebook")

# ------------------ LOAD DATA ------------------

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)

# Save missingness table
missing = df.isna().sum().to_frame("n_missing")
missing["frac_missing"] = missing["n_missing"] / len(df)
missing.to_csv(os.path.join(OUT_DIR, "missingness_summary.csv"))

print("\nMissingness:\n", missing)

# ------------------ CYTOKINE SUMMARY ------------------

cyto_desc = df[CYTO_COLS].describe().T
cyto_desc.to_csv(os.path.join(OUT_DIR, "cytokine_summary_stats.csv"))
print("\nCytokine summary stats:\n", cyto_desc)

# ------------------ DISTRIBUTIONS ------------------
'''
for col in CYTO_COLS:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} (raw)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{col}_hist_raw.png"))
    plt.close()

    # log1p in case of zeros
    plt.figure(figsize=(6,4))
    sns.histplot(np.log1p(df[col]), kde=True)
    plt.title(f"{col} (log1p)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{col}_hist_log1p.png"))
    plt.close()
'''

for col in CYTO_COLS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{col} Distribution", fontsize=13, fontweight="bold")

    # Raw — clipped at 99th percentile
    p99 = df[col].quantile(0.99)
    data_clipped = df[col].clip(upper=p99)
    axes[0].hist(data_clipped, bins=30, edgecolor="white",
                 color="#6BAED6", alpha=0.85, density=True)
    data_clipped.plot.kde(ax=axes[0], color="#1A4F72",
                          linewidth=2, label="KDE")
    axes[0].set_title(f"{col} (raw, clipped at 99th pct)")
    axes[0].set_xlabel("MFI")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(left=0)

    # log1p
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
    plt.savefig(OUT_DIR / f"{col}_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close()
# ------------------ CORRELATION HEATMAP ------------------

corr = df[CYTO_COLS].corr(method="spearman")
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
plt.title("Cytokine Spearman correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cytokine_corr_heatmap.png"))
plt.close()

corr.to_csv(os.path.join(OUT_DIR, "cytokine_corr_matrix.csv"))

# ------------------ AGE / SEX / RACE DISTRIBUTIONS ------------------

if "AGE" in df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df["AGE"], bins=10)
    plt.title("Age distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "AGE_hist.png"))
    plt.close()

if "SEXC" in df.columns:
    plt.figure(figsize=(4,4))
    sns.countplot(x="SEXC", data=df)
    plt.title("Sex counts")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "SEXC_counts.png"))
    plt.close()

if "RACEGRP" in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="RACEGRP", data=df)
    plt.title("Race group counts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "RACEGRP_counts.png"))
    plt.close()

# ------------------ CYTOKINES vs SEX ------------------

if "SEXC" in df.columns:
    for col in CYTO_COLS:
        plt.figure(figsize=(6,4))
        sns.boxplot(x="SEXC", y=col, data=df)
        plt.title(f"{col} by sex")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{col}_by_SEXC_box.png"))
        plt.close()

# ------------------ CYTOKINES vs AGE ------------------

if "AGE" in df.columns:
    for col in CYTO_COLS:
        plt.figure(figsize=(6,4))
        sns.regplot(x="AGE", y=col, data=df, scatter_kws={"alpha":0.6})
        plt.title(f"{col} vs AGE")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{col}_vs_AGE_reg.png"))
        plt.close()

# ------------------ QUICK PCA PREVIEW ------------------

# Drop rows with any missing cytokine before PCA
X_raw = df[CYTO_COLS].copy()
X_raw = X_raw.dropna(axis=0)
idx_kept = X_raw.index

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

pca = PCA()
X_pca = pca.fit_transform(X)

explained = pca.explained_variance_ratio_

# Scree plot
plt.figure(figsize=(6,4))
plt.plot(range(1, len(explained)+1), explained, marker="o")
plt.xlabel("PC")
plt.ylabel("Variance explained")
plt.title("PCA scree plot (cytokines)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PCA_scree.png"))
plt.close()

# PC1 vs PC2 colored by sex (if available)
if "SEXC" in df.columns:
    plt.figure(figsize=(6,5))
    sex_for_pca = df.loc[idx_kept, "SEXC"]
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=sex_for_pca)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PC1 vs PC2 by sex")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "PCA_PC1_PC2_by_SEXC.png"))
    plt.close()

# Save PCA loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=CYTO_COLS,
    columns=[f"PC{i+1}" for i in range(len(CYTO_COLS))]
)
loadings.to_csv(os.path.join(OUT_DIR, "PCA_loadings.csv"))

print("EDA complete. Plots written to:", OUT_DIR)

# ------------------ COMBINED FIGURE 1 PANEL (FOR PAPER) ------------------

sns.set(style="whitegrid", context="talk")

fig = plt.figure(figsize=(24, 8), facecolor="white")
#fig.suptitle("Figure 1: Dataset Overview\nCytokine Expression Profiles in 125 Healthy Donors",
#             fontsize=16, fontweight="bold", y=1.02)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

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
p99 = df["IFN-gamma"].quantile(0.99)
data_clipped = df["IFN-gamma"].clip(upper=p99)
ax_b.hist(data_clipped, bins=30, edgecolor="white",
          color="#6BAED6", alpha=0.85, density=True)
data_clipped.plot.kde(ax=ax_b, color="#08306B",
                      linewidth=2.5, label="KDE")
ax_b.set_xlim(left=0)
ax_b.set_title("B: IFN-\u03b3 Raw Distribution\n(clipped at 99th percentile)",
               fontsize=13, fontweight="bold", pad=12)
ax_b.set_xlabel("MFI", fontsize=11)
ax_b.set_ylabel("Density", fontsize=11)
ax_b.legend(fontsize=10)
ax_b.tick_params(labelsize=10)
ax_b.yaxis.grid(True, alpha=0.4)

# ---- Panel C: IFN-gamma log1p distribution ----
ax_c = fig.add_subplot(gs[0, 2])
log_data = np.log1p(df["IFN-gamma"])
ax_c.hist(log_data, bins=30, edgecolor="white",
          color="#5BAD72", alpha=0.85, density=True)
log_data.plot.kde(ax=ax_c, color="#08306B",
                  linewidth=2.5, label="KDE")
ax_c.set_title("C: IFN-\u03b3 log1p Transformed\nDistribution",
               fontsize=13, fontweight="bold", pad=12)
ax_c.set_xlabel("log1p(MFI)", fontsize=11)
ax_c.set_ylabel("Density", fontsize=11)
ax_c.legend(fontsize=10)
ax_c.tick_params(labelsize=10)
ax_c.yaxis.grid(True, alpha=0.4)

plt.savefig(OUT_DIR / "figure1_data_overview_panel.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# reset seaborn style back to default for rest of script
sns.set(style="whitegrid", context="notebook")
print("  Saved: figure1_data_overview_panel.png")