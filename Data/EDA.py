#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

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