"""
================================================================================
Cytokine + Cell Type Expression: Covariate Analysis
================================================================================
Project Goal:
    Quantify how much variance in cytokine expression and immune cell type
    proportions is explained by biological sex, age group, and race using
    OLS linear models with Type II ANOVA partial R² decomposition.

Pipeline Sections:
    1.  Data Loading & Validation
    2.  Preprocessing (log1p cytokines, impute cell types)
    3.  Global PCA — Cytokines (scree + loadings, matches notebook style)
    4.  Global PCA — Cell Types
    5.  Global PCA — Combined (cytokines + cell types)
    6.  OLS Covariate Models — Cytokines
    7.  OLS Covariate Models — Cell Types
    8.  Visualization Panel A  (PCA colored by covariates)
    9.  Visualization Panel B  (Partial R² heatmaps)
    10. Visualization Panel C  (Stacked partial R² bar charts)
    11. Export Results

Covariate model:
    Feature ~ C(SEXC) + C(AGEGR1C) + C(RACEGRP)   [OLS, Type II ANOVA]
    Partial R² = SS_covariate / SS_total   per covariate per feature
================================================================================
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ================================================================================
# PATHS  —  edit to match your directory structure
# ================================================================================
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data-Emma"
OUTPUT_DIR  = BASE_DIR / "PCA+KNN" / "Outputs" / "covariate_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# SECTION 1: DATA LOADING & VALIDATION
# ================================================================================
print("=" * 65)
print("SECTION 1: DATA LOADING & VALIDATION")
print("=" * 65)

df = pd.read_csv(DATA_DIR / "analysis_merged_subject_level.csv")

DEMO_COLS = ['SUBJECT_ID', 'AGE', 'AGEGR1N', 'AGEGR1C', 'RACEGRP',
             'SEXN', 'SEXC', 'FASFL', 'AGE_BINARY', 'Batch date']
CYTO_COLS = [c for c in df.columns
             if c not in DEMO_COLS and not c.startswith('CT_')]
CELL_COLS = [c for c in df.columns if c.startswith('CT_')]

n_samples = len(df)

print(f"  Dataset        : {n_samples} subjects × "
      f"{len(CYTO_COLS)} cytokines + {len(CELL_COLS)} cell types")
print(f"  Cytokines      : {CYTO_COLS}")
print(f"  Sex            : {df['SEXC'].value_counts().to_dict()}")
print(f"  Age groups     : {df['AGEGR1C'].value_counts().to_dict()}")
print(f"  Race           : {df['RACEGRP'].value_counts().to_dict()}")
print(f"  Missing (cyto) : {df[CYTO_COLS].isna().sum().sum()}")
print(f"  Missing (cell) : {df[CELL_COLS].isna().sum().sum()}")
print()

# ================================================================================
# SECTION 2: PREPROCESSING
# ================================================================================
print("=" * 65)
print("SECTION 2: PREPROCESSING")
print("=" * 65)

# Log1p transform cytokines — reduces right skew from extreme MFI values
X_cyto_raw = df[CYTO_COLS].values.astype(float)
X_cyto_log = np.log1p(X_cyto_raw)

# Median impute missing cell type values
df_cell = df[CELL_COLS].copy()
for col in CELL_COLS:
    df_cell[col] = df_cell[col].fillna(df_cell[col].median())
X_cell = df_cell.values.astype(float)

# Labels for covariate analysis
y_sex  = (df["SEXC"] == "Male").astype(int).values          # 1=Male, 0=Female
y_age  = df["AGEGR1C"].values
y_race = df["RACEGRP"].values

# Build model dataframe — sanitize column names for statsmodels formulas
df_model = df[["SEXC", "AGEGR1C", "RACEGRP"]].copy()
for i, cyto in enumerate(CYTO_COLS):
    safe = cyto.replace("-", "_").replace(" ", "_")
    df_model[safe] = X_cyto_log[:, i]
for i, cell in enumerate(CELL_COLS):
    safe = cell.replace("-", "_").replace(" ", "_").replace(
        "+", "pos").replace("(", "").replace(")", "").replace(" ", "_")
    df_model[safe] = X_cell[:, i]

print(f"  log1p transform applied to {len(CYTO_COLS)} cytokines")
print(f"  Median imputation applied to {len(CELL_COLS)} cell type columns")
print()

# ================================================================================
# SECTION 3: GLOBAL PCA — CYTOKINES  (exploratory, matches notebook style)
# ================================================================================
print("=" * 65)
print("SECTION 3: GLOBAL PCA — CYTOKINES")
print("=" * 65)

def global_pca(X_raw_input, label):
    """Standardize → center → SVD. Returns explained ratios, cumulative, PCA scores, Vt."""
    mean = X_raw_input.mean(axis=0)
    std  = X_raw_input.std(axis=0);  std[std == 0] = 1
    Xs   = (X_raw_input - mean) / std
    Xc   = Xs - Xs.mean(axis=0)
    Xc   = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eig  = (S ** 2) / (len(Xc) - 1)
    er   = eig / eig.sum()
    cv   = np.cumsum(er)
    scores = Xc @ Vt[:3].T
    n80  = int(np.searchsorted(cv, 0.80)) + 1
    n90  = int(np.searchsorted(cv, 0.90)) + 1
    print(f"  {label}:")
    print(f"    PC1={er[0]*100:.1f}%  PC2={er[1]*100:.1f}%  PC3={er[2]*100:.1f}%")
    print(f"    PCs for 80%: {n80}   PCs for 90%: {n90}")
    return er, cv, scores, Vt, n80

er_cyto, cv_cyto, pca_cyto, Vt_cyto, n80_cyto = global_pca(X_cyto_log, "Cytokines (log1p)")
er_cell, cv_cell, pca_cell, Vt_cell, n80_cell  = global_pca(X_cell,     "Cell Types")

# Combined PCA — standardize each modality separately then concatenate
def _standardize(X):
    m = X.mean(axis=0); s = X.std(axis=0); s[s == 0] = 1
    Xs = (X - m) / s;  Xc = Xs - Xs.mean(axis=0)
    return np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)

X_combined = np.hstack([_standardize(X_cyto_log), _standardize(X_cell)])
_, S_comb, Vt_comb = np.linalg.svd(X_combined, full_matrices=False)
eig_comb  = (S_comb ** 2) / (n_samples - 1)
er_comb   = eig_comb / eig_comb.sum()
cv_comb   = np.cumsum(er_comb)
pca_comb  = X_combined @ Vt_comb[:3].T
n80_comb  = int(np.searchsorted(cv_comb, 0.80)) + 1

print(f"  Combined (cytokines + cell types):")
print(f"    PC1={er_comb[0]*100:.1f}%  PC2={er_comb[1]*100:.1f}%  "
      f"PC3={er_comb[2]*100:.1f}%  PCs for 80%: {n80_comb}")
print()

# Save loadings
pd.DataFrame(Vt_cyto[:3].T, index=CYTO_COLS,
             columns=["PC1", "PC2", "PC3"]).round(4).to_csv(
    OUTPUT_DIR / "pca_loadings_cytokines.csv")
print("  CSV saved: pca_loadings_cytokines.csv")
print()

# ================================================================================
# SECTION 4: OLS HELPER FUNCTION
# ================================================================================
try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "statsmodels", "--break-system-packages", "-q"])
    import statsmodels.formula.api as smf
    import statsmodels.api as sm


def run_ols_covariate(feature_names, safe_names, modality_label):
    """
    Run OLS for each feature:   Feature ~ C(SEXC) + C(AGEGR1C) + C(RACEGRP)
    Returns DataFrame with R², partial R², and p-values per covariate.
    """
    rows = []
    for feat, safe in zip(feature_names, safe_names):
        formula = f"{safe} ~ C(SEXC) + C(AGEGR1C) + C(RACEGRP)"
        try:
            model       = smf.ols(formula, data=df_model).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            total_ss    = anova_table["sum_sq"].sum()

            ss_sex  = anova_table.loc["C(SEXC)",    "sum_sq"] \
                      if "C(SEXC)"    in anova_table.index else 0
            ss_age  = anova_table.loc["C(AGEGR1C)", "sum_sq"] \
                      if "C(AGEGR1C)" in anova_table.index else 0
            ss_race = anova_table.loc["C(RACEGRP)", "sum_sq"] \
                      if "C(RACEGRP)" in anova_table.index else 0

            pr2_sex  = ss_sex  / total_ss if total_ss > 0 else 0
            pr2_age  = ss_age  / total_ss if total_ss > 0 else 0
            pr2_race = ss_race / total_ss if total_ss > 0 else 0

            pval_sex  = model.pvalues.get("C(SEXC)[T.Male]", np.nan)
            pval_age  = min([v for k, v in model.pvalues.items()
                             if "AGEGR1C" in k] or [np.nan])
            pval_race = min([v for k, v in model.pvalues.items()
                             if "RACEGRP" in k] or [np.nan])

            rows.append({
                "Feature":         feat,
                "Modality":        modality_label,
                "R2_full":         round(model.rsquared,     4),
                "R2_adj":          round(model.rsquared_adj, 4),
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


# ================================================================================
# SECTION 5: OLS — CYTOKINES
# ================================================================================
print("=" * 65)
print("SECTION 5: OLS COVARIATE ANALYSIS — CYTOKINES")
print("=" * 65)

cyto_safe = [c.replace("-", "_").replace(" ", "_") for c in CYTO_COLS]
ols_cyto  = run_ols_covariate(CYTO_COLS, cyto_safe, "Cytokine")
ols_cyto.to_csv(OUTPUT_DIR / "ols_cytokine_results.csv", index=False)

print(f"\n  {'Feature':<12} {'R²':>6} {'pR²Sex':>8} {'pR²Age':>8} "
      f"{'pR²Race':>8} {'pSex':>8} {'pAge':>8} {'pRace':>8}")
print("  " + "-" * 78)
for _, row in ols_cyto.iterrows():
    ss = "✓" if row["Sig_Sex"]  else " "
    sa = "✓" if row["Sig_Age"]  else " "
    sr = "✓" if row["Sig_Race"] else " "
    print(f"  {row['Feature']:<12} {row['R2_full']:>6.3f} "
          f"{row['Partial_R2_Sex']:>8.3f} {row['Partial_R2_Age']:>8.3f} "
          f"{row['Partial_R2_Race']:>8.3f} "
          f"{row['Pval_Sex']:>7.4f}{ss} {row['Pval_Age']:>7.4f}{sa} "
          f"{row['Pval_Race']:>7.4f}{sr}")

print(f"\n  Significant for sex  : "
      f"{ols_cyto[ols_cyto['Sig_Sex']]['Feature'].tolist()}")
print(f"  Significant for age  : "
      f"{ols_cyto[ols_cyto['Sig_Age']]['Feature'].tolist()}")
print(f"  Significant for race : "
      f"{ols_cyto[ols_cyto['Sig_Race']]['Feature'].tolist()}")
print(f"  Mean R²              : {ols_cyto['R2_full'].mean():.4f}")
print(f"  Max R²               : {ols_cyto['R2_full'].max():.4f} "
      f"({ols_cyto.loc[ols_cyto['R2_full'].idxmax(), 'Feature']})")
print()

# ================================================================================
# SECTION 6: OLS — CELL TYPES
# ================================================================================
print("=" * 65)
print("SECTION 6: OLS COVARIATE ANALYSIS — CELL TYPES")
print("=" * 65)

cell_safe = [
    c.replace("-", "_").replace(" ", "_").replace("+", "pos")
     .replace("(", "").replace(")", "")
    for c in CELL_COLS
]
ols_cell = run_ols_covariate(CELL_COLS, cell_safe, "Cell Type")
ols_cell.to_csv(OUTPUT_DIR / "ols_celltype_results.csv", index=False)

print(f"\n  Top 15 cell types by R²:")
print(f"  {'Feature':<45} {'R²':>6} {'pR²Sex':>8} {'pR²Age':>8} {'pR²Race':>8}")
print("  " + "-" * 78)
for _, row in ols_cell.head(15).iterrows():
    ss = "✓" if row["Sig_Sex"]  else " "
    sa = "✓" if row["Sig_Age"]  else " "
    sr = "✓" if row["Sig_Race"] else " "
    feat_short = row['Feature'].replace("CT_", "")[:40]
    print(f"  {feat_short:<45} {row['R2_full']:>6.3f} "
          f"{row['Partial_R2_Sex']:>8.3f} {row['Partial_R2_Age']:>8.3f} "
          f"{row['Partial_R2_Race']:>8.3f}  {ss}{sa}{sr}")

print(f"\n  Sig for sex  (n): {ols_cell['Sig_Sex'].sum()}")
print(f"  Sig for age  (n): {ols_cell['Sig_Age'].sum()}")
print(f"  Sig for race (n): {ols_cell['Sig_Race'].sum()}")
print(f"  Mean R²         : {ols_cell['R2_full'].mean():.4f}")
print(f"  Max R²          : {ols_cell['R2_full'].max():.4f} "
      f"({ols_cell.loc[ols_cell['R2_full'].idxmax(), 'Feature']})")
print()

# Combined export
ols_all = pd.concat([ols_cyto, ols_cell], ignore_index=True)
ols_all.to_csv(OUTPUT_DIR / "ols_all_features_results.csv", index=False)
print("  CSVs saved: ols_cytokine_results.csv, ols_celltype_results.csv, "
      "ols_all_features_results.csv")
print()

# ================================================================================
# SECTION 7: VISUALIZATION — PANEL A
#   Scree plots (cytokines / cell types / combined)  +  PCA scatter by covariate
#   Matches notebook style: bar chart per-PC + twinx cumulative line
# ================================================================================
print("=" * 65)
print("SECTION 7: VISUALIZATION — PANEL A (PCA overview)")
print("=" * 65)

COLOR_FEMALE = "#E87D7D"
COLOR_MALE   = "#6BAED6"
COLOR_BAR    = "#4A7FC1"

AGE_COLORS  = {"18-29": "#1f77b4", "30-39": "#2ca02c",
               "40-49": "#ff7f0e", "50-66": "#d62728"}
RACE_COLORS = {"Caucasian": "#4A7FC1", "African-American": "#E87D7D",
               "Asian": "#5BAD72", "Other": "#F4A460", "Hispanic": "#9B59B6"}

def scree_panel(ax, er, cv, n80, title):
    """Notebook-style scree: bars + twinx cumulative line."""
    x_pcs = range(1, len(er) + 1)
    ax.bar(x_pcs, er * 100, color=COLOR_BAR, alpha=0.7, label="Per-PC")
    ax2 = ax.twinx()
    ax2.plot(x_pcs, cv * 100, color="darkred", marker="o",
             markersize=4, linewidth=1.5, label="Cumulative")
    ax2.axhline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(90, color="gray", linestyle=":",  linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=9, color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred", labelsize=8)
    ax2.set_ylim(0, 105)
    ax.set_xlabel("Principal Component", fontsize=9)
    ax.set_ylabel("Explained Variance (%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.text(n80 + 0.2, er[n80 - 1] * 100 + 0.5,
            f"{n80} PCs→80%", fontsize=7, color="gray")


def scatter_panel(ax, scores, labels, color_map, title, pc1_pct, pc2_pct):
    """PCA scatter colored by a categorical label."""
    for label, color in color_map.items():
        mask = labels == label
        if mask.sum() > 0:
            ax.scatter(scores[mask, 0], scores[mask, 1],
                       c=color, alpha=0.6, s=28,
                       edgecolors="white", linewidths=0.4, label=label)
    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% var)", fontsize=9)
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% var)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, markerscale=1.2)
    ax.tick_params(labelsize=8)


# Figure A: 2 rows × 3 cols
#   Row 1: scree cytokines | scree cell types | scree combined
#   Row 2: combined PCA × sex | combined PCA × age | combined PCA × race
fig_a = plt.figure(figsize=(18, 10))
fig_a.suptitle(
    "Global PCA Overview — Cytokines, Cell Types, Combined\n"
    "Scree plots (top) and PCA scatter colored by demographic covariates (bottom)",
    fontsize=13, fontweight="bold", y=0.98
)
gs_a = gridspec.GridSpec(2, 3, figure=fig_a, hspace=0.48, wspace=0.38)

scree_panel(fig_a.add_subplot(gs_a[0, 0]),
            er_cyto, cv_cyto, n80_cyto,
            "A: Scree — Cytokines (log1p)")
scree_panel(fig_a.add_subplot(gs_a[0, 1]),
            er_cell, cv_cell, n80_cell,
            "B: Scree — Cell Types")
scree_panel(fig_a.add_subplot(gs_a[0, 2]),
            er_comb, cv_comb, n80_comb,
            "C: Scree — Combined")

scatter_panel(fig_a.add_subplot(gs_a[1, 0]),
              pca_comb, df["SEXC"].values,
              {"Female": COLOR_FEMALE, "Male": COLOR_MALE},
              "D: Combined PCA — by Sex",
              er_comb[0] * 100, er_comb[1] * 100)

scatter_panel(fig_a.add_subplot(gs_a[1, 1]),
              pca_comb, df["AGEGR1C"].values,
              AGE_COLORS,
              "E: Combined PCA — by Age Group",
              er_comb[0] * 100, er_comb[1] * 100)

scatter_panel(fig_a.add_subplot(gs_a[1, 2]),
              pca_comb, df["RACEGRP"].values,
              RACE_COLORS,
              "F: Combined PCA — by Race",
              er_comb[0] * 100, er_comb[1] * 100)

plt.savefig(FIGURES_DIR / "panel_A_pca_overview.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: panel_A_pca_overview.png")

# ================================================================================
# SECTION 8: VISUALIZATION — PANEL B
#   Partial R² heatmaps — cytokines and top-20 cell types
#   Rows = features, Cols = Sex | Age | Race
# ================================================================================
print("=" * 65)
print("SECTION 8: VISUALIZATION — PANEL B (Partial R² heatmaps)")
print("=" * 65)

fig_b = plt.figure(figsize=(16, 14))
fig_b.suptitle(
    "Partial R² Heatmaps — Variance Explained by Each Demographic Covariate\n"
    "* = significant (p < 0.05)  |  Values = partial R² (Type II ANOVA)",
    fontsize=13, fontweight="bold", y=0.98
)
gs_b = gridspec.GridSpec(1, 2, figure=fig_b, wspace=0.35)

def r2_heatmap(ax, ols_df, title, max_rows=None):
    data = ols_df.head(max_rows) if max_rows else ols_df
    mat  = data[["Partial_R2_Sex", "Partial_R2_Age", "Partial_R2_Race"]].values
    vmax = max(mat.max(), 0.01)
    im   = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Sex", "Age", "Race"], fontsize=11, fontweight="bold")
    feat_labels = [
        f.replace("CT_", "")[:35] if f.startswith("CT_") else f
        for f in data["Feature"].tolist()
    ]
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Partial R²", fraction=0.04, pad=0.04)
    sig_cols = ["Sig_Sex", "Sig_Age", "Sig_Race"]
    for i in range(len(data)):
        for j in range(3):
            val = mat[i, j]
            sig = data.iloc[i][sig_cols[j]]
            txt = f"{val:.3f}" + ("*" if sig else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if val > vmax * 0.6 else "black")

r2_heatmap(fig_b.add_subplot(gs_b[0, 0]),
           ols_cyto,
           "Cytokines — Partial R²\n(sorted by total R²)")
r2_heatmap(fig_b.add_subplot(gs_b[0, 1]),
           ols_cell,
           "Top 20 Cell Types — Partial R²\n(sorted by total R²)",
           max_rows=20)

plt.savefig(FIGURES_DIR / "panel_B_partial_r2_heatmaps.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: panel_B_partial_r2_heatmaps.png")

# ================================================================================
# SECTION 9: VISUALIZATION — PANEL C
#   4-panel figure matching notebook layout style (2×2 GridSpec)
#   [C1] Total R² bar — cytokines
#   [C2] Stacked partial R² — cytokines
#   [C3] Total R² bar — top 20 cell types
#   [C4] Stacked partial R² — top 20 cell types
# ================================================================================
print("=" * 65)
print("SECTION 9: VISUALIZATION — PANEL C (R² bar charts)")
print("=" * 65)

fig_c = plt.figure(figsize=(18, 16))
fig_c.suptitle(
    "Variance Explained by Demographics — Cytokines vs Cell Types\n"
    "Red bars = at least one covariate significant (p<0.05)",
    fontsize=13, fontweight="bold", y=0.98
)
gs_c = gridspec.GridSpec(2, 2, figure=fig_c, hspace=0.45, wspace=0.38)

def total_r2_bar(ax, ols_df, title, max_rows=None):
    data = ols_df.head(max_rows).sort_values("R2_full", ascending=True) \
           if max_rows else ols_df.sort_values("R2_full", ascending=True)
    feat_labels = [
        f.replace("CT_", "").replace("_", " ")[:35]
        if f.startswith("CT_") else f
        for f in data["Feature"]
    ]
    bar_colors = ["#E87D7D" if (r["Sig_Sex"] or r["Sig_Age"] or r["Sig_Race"])
                  else "#AAAAAA"
                  for _, r in data.iterrows()]
    ax.barh(feat_labels, data["R2_full"], color=bar_colors,
            alpha=0.85, edgecolor="white")
    ax.axvline(0.05, color="gray", linestyle="--", alpha=0.5, linewidth=1,
               label="R²=0.05")
    ax.set_xlabel("Total R² (sex + age + race)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(row["R2_full"] + 0.002, i, f"{row['R2_full']:.3f}",
                va="center", fontsize=7)


def stacked_r2_bar(ax, ols_df, title, max_rows=None):
    data = ols_df.head(max_rows).sort_values("R2_full", ascending=True) \
           if max_rows else ols_df.sort_values("R2_full", ascending=True)
    feat_labels = [
        f.replace("CT_", "").replace("_", " ")[:35]
        if f.startswith("CT_") else f
        for f in data["Feature"]
    ]
    y_pos = np.arange(len(data))
    ax.barh(y_pos, data["Partial_R2_Sex"],
            color="#6BAED6", alpha=0.85, label="Sex")
    ax.barh(y_pos, data["Partial_R2_Age"],
            left=data["Partial_R2_Sex"],
            color="#F4A460", alpha=0.85, label="Age")
    ax.barh(y_pos, data["Partial_R2_Race"],
            left=data["Partial_R2_Sex"] + data["Partial_R2_Age"],
            color="#5BAD72", alpha=0.85, label="Race")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.set_xlabel("Partial R²", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.3)


total_r2_bar(fig_c.add_subplot(gs_c[0, 0]),
             ols_cyto,
             "C1: Cytokines — Total R²\n(red = any covariate p<0.05)")
stacked_r2_bar(fig_c.add_subplot(gs_c[0, 1]),
               ols_cyto,
               "C2: Cytokines — Partial R² Breakdown\nSex vs Age vs Race")
total_r2_bar(fig_c.add_subplot(gs_c[1, 0]),
             ols_cell,
             "C3: Top 20 Cell Types — Total R²\n(red = any covariate p<0.05)",
             max_rows=20)
stacked_r2_bar(fig_c.add_subplot(gs_c[1, 1]),
               ols_cell,
               "C4: Top 20 Cell Types — Partial R² Breakdown\nSex vs Age vs Race",
               max_rows=20)

plt.savefig(FIGURES_DIR / "panel_C_r2_bar_charts.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: panel_C_r2_bar_charts.png")

# ================================================================================
# SECTION 10: VISUALIZATION — PANEL D
#   PC loadings heatmap for cytokines (matches notebook panel style)
# ================================================================================
print("=" * 65)
print("SECTION 10: VISUALIZATION — PANEL D (PC loadings)")
print("=" * 65)

fig_d = plt.figure(figsize=(14, 6))
fig_d.suptitle(
    "PC Loadings — Cytokines\n"
    "Red = positive loading, Blue = negative loading",
    fontsize=13, fontweight="bold", y=1.01
)
gs_d = gridspec.GridSpec(1, 2, figure=fig_d, wspace=0.38)

# PC loadings heatmap
ax_load = fig_d.add_subplot(gs_d[0, 0])
loading_matrix = Vt_cyto[:3].T      # (n_features, 3)
im_l = ax_load.imshow(loading_matrix, cmap="RdBu_r", aspect="auto",
                      vmin=-0.5, vmax=0.5)
ax_load.set_xticks([0, 1, 2])
ax_load.set_xticklabels(
    [f"PC1\n({er_cyto[0]*100:.1f}%)", f"PC2\n({er_cyto[1]*100:.1f}%)",
     f"PC3\n({er_cyto[2]*100:.1f}%)"], fontsize=9
)
ax_load.set_yticks(range(len(CYTO_COLS)))
ax_load.set_yticklabels(CYTO_COLS, fontsize=9)
ax_load.set_title("D: Cytokine PC Loadings Heatmap", fontsize=10, fontweight="bold")
plt.colorbar(im_l, ax=ax_load, label="Loading value")
for i in range(len(CYTO_COLS)):
    for j in range(3):
        ax_load.text(j, i, f"{loading_matrix[i, j]:+.2f}",
                     ha="center", va="center", fontsize=7,
                     color="white" if abs(loading_matrix[i, j]) > 0.3 else "black")

# Top-3 cytokines per PC bar chart
ax_top = fig_d.add_subplot(gs_d[0, 1])
pc_names  = ["PC1", "PC2", "PC3"]
pc_colors = [COLOR_BAR, "#E87D7D", "#5BAD72"]
y_offset  = 0
bar_h     = 0.25
yticks, ylabels = [], []
for pc_idx, (pc_name, pc_color) in enumerate(zip(pc_names, pc_colors)):
    top3_idx = np.argsort(np.abs(Vt_cyto[pc_idx]))[::-1][:5]
    for rank, feat_idx in enumerate(top3_idx):
        ypos = y_offset + rank * bar_h
        ax_top.barh(ypos, Vt_cyto[pc_idx, feat_idx],
                    height=bar_h * 0.8, color=pc_color, alpha=0.8)
        yticks.append(ypos)
        ylabels.append(f"{pc_name}: {CYTO_COLS[feat_idx]}")
    y_offset += (5 + 1) * bar_h

ax_top.set_yticks(yticks)
ax_top.set_yticklabels(ylabels, fontsize=8)
ax_top.axvline(0, color="black", linewidth=0.8)
ax_top.set_xlabel("PC Loading Value", fontsize=9)
ax_top.set_title("E: Top 5 Cytokines per PC\n(sorted by |loading|)",
                 fontsize=10, fontweight="bold")
ax_top.tick_params(labelsize=8)
ax_top.grid(axis="x", alpha=0.3)

plt.savefig(FIGURES_DIR / "panel_D_pc_loadings.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: panel_D_pc_loadings.png")

# ================================================================================
# SECTION 11: SUMMARY PRINT + EXPORT
# ================================================================================
print()
print("=" * 65)
print("SECTION 11: SUMMARY")
print("=" * 65)

dominant_cyto = max(["Sex", "Age", "Race"],
                    key=lambda x: ols_cyto[f"Partial_R2_{x}"].mean())
dominant_cell = max(["Sex", "Age", "Race"],
                    key=lambda x: ols_cell[f"Partial_R2_{x}"].mean())

print(f"""
  CYTOKINES:
    Mean R²              : {ols_cyto['R2_full'].mean():.4f}
    Max R²               : {ols_cyto['R2_full'].max():.4f}
                           ({ols_cyto.loc[ols_cyto['R2_full'].idxmax(), 'Feature']})
    Mean pR² Sex         : {ols_cyto['Partial_R2_Sex'].mean():.4f}
    Mean pR² Age         : {ols_cyto['Partial_R2_Age'].mean():.4f}
    Mean pR² Race        : {ols_cyto['Partial_R2_Race'].mean():.4f}
    Dominant covariate   : {dominant_cyto}
    Sig for sex          : {ols_cyto[ols_cyto['Sig_Sex']]['Feature'].tolist()}
    Sig for age          : {ols_cyto[ols_cyto['Sig_Age']]['Feature'].tolist()}
    Sig for race         : {ols_cyto[ols_cyto['Sig_Race']]['Feature'].tolist()}

  CELL TYPES:
    Mean R²              : {ols_cell['R2_full'].mean():.4f}
    Max R²               : {ols_cell['R2_full'].max():.4f}
                           ({ols_cell.loc[ols_cell['R2_full'].idxmax(), 'Feature']})
    Mean pR² Sex         : {ols_cell['Partial_R2_Sex'].mean():.4f}
    Mean pR² Age         : {ols_cell['Partial_R2_Age'].mean():.4f}
    Mean pR² Race        : {ols_cell['Partial_R2_Race'].mean():.4f}
    Dominant covariate   : {dominant_cell}
    Sig for sex   (n)    : {ols_cell['Sig_Sex'].sum()} / {len(ols_cell)}
    Sig for age   (n)    : {ols_cell['Sig_Age'].sum()} / {len(ols_cell)}
    Sig for race  (n)    : {ols_cell['Sig_Race'].sum()} / {len(ols_cell)}

  COMPARISON:
    Cell types mean R²   : {ols_cell['R2_full'].mean():.4f}  vs  \
cytokines mean R²: {ols_cyto['R2_full'].mean():.4f}
    Ratio                : {ols_cell['R2_full'].mean() / max(ols_cyto['R2_full'].mean(), 1e-6):.1f}x more demographic signal in cell types
""")

print(f"  Outputs : {OUTPUT_DIR}")
print(f"  Figures : {FIGURES_DIR}")
print("=" * 65)