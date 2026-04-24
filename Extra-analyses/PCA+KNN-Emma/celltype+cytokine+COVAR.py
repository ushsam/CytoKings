"""
================================================================================
Cytokine + Cell Type Expression — Covariate Analysis
Sex, Age, and Race as covariates of cytokine and cell type expression
Using OLS linear models to quantify demographic variance explained
================================================================================
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ================================================================================
# PATHS
# ================================================================================
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data-Emma"
OUTPUT_DIR  = BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "COVAR-celltype+cytokine"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# SECTION 1: DATA LOADING
# ================================================================================
print("=" * 65)
print("SECTION 1: DATA LOADING")
print("=" * 65)

df = pd.read_csv(DATA_DIR / "analysis_merged_subject_level.csv")

DEMO_COLS = ['SUBJECT_ID', 'AGE', 'AGEGR1N', 'AGEGR1C', 'RACEGRP',
             'SEXN', 'SEXC', 'FASFL', 'AGE_BINARY', 'Batch date']
CYTO_COLS = [c for c in df.columns
             if c not in DEMO_COLS and not c.startswith('CT_')]
CELL_COLS = [c for c in df.columns if c.startswith('CT_')]

print(f"  Subjects    : {len(df)}")
print(f"  Cytokines   : {len(CYTO_COLS)} — {CYTO_COLS}")
print(f"  Cell types  : {len(CELL_COLS)}")
print(f"  Sex         : {df['SEXC'].value_counts().to_dict()}")
print(f"  Age groups  : {df['AGEGR1C'].value_counts().to_dict()}")
print(f"  Race        : {df['RACEGRP'].value_counts().to_dict()}")
print()

# Log1p transform cytokines; cell types are already proportions
df_model = df.copy()
for cyto in CYTO_COLS:
    df_model[cyto] = np.log1p(df[cyto].values.astype(float))

# Median impute missing cell type values
for col in CELL_COLS:
    df_model[col] = df_model[col].fillna(df_model[col].median())

# Binary age label
df_model['AGE_BINARY'] = df_model['AGEGR1C'].apply(
    lambda x: 0 if x in ['18-29', '30-39'] else 1
)

# ================================================================================
# SECTION 2: OLS HELPER FUNCTION
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


def run_ols(feature_list, label):
    """Run OLS covariate analysis for a list of features."""
    results = []
    for feat in feature_list:
        safe_feat = feat.replace("-", "_").replace(" ", "_").replace(
            "+", "pos").replace("(", "").replace(")", "")
        df_model[safe_feat] = df_model[feat]

        formula = f"{safe_feat} ~ C(SEXC) + C(AGEGR1C) + C(RACEGRP)"
        try:
            model      = smf.ols(formula, data=df_model).fit()
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

            results.append({
                "Feature":         feat,
                "Modality":        label,
                "R2_full":         round(model.rsquared,     4),
                "R2_adj":          round(model.rsquared_adj, 4),
                "Partial_R2_Sex":  round(pr2_sex,  4),
                "Partial_R2_Age":  round(pr2_age,  4),
                "Partial_R2_Race": round(pr2_race, 4),
                "Pval_Sex":        round(pval_sex,  6) if not np.isnan(pval_sex)  else np.nan,
                "Pval_Age":        round(pval_age,  6) if not np.isnan(pval_age)  else np.nan,
                "Pval_Race":       round(pval_race, 6) if not np.isnan(pval_race) else np.nan,
                "Sig_Sex":         pval_sex  < 0.05 if not np.isnan(pval_sex)  else False,
                "Sig_Age":         pval_age  < 0.05 if not np.isnan(pval_age)  else False,
                "Sig_Race":        pval_race < 0.05 if not np.isnan(pval_race) else False,
            })
        except Exception as e:
            print(f"  WARNING: OLS failed for {feat}: {e}")

    return pd.DataFrame(results).sort_values("R2_full", ascending=False)


# ================================================================================
# SECTION 3: RUN OLS — CYTOKINES
# ================================================================================
print("=" * 65)
print("SECTION 3: OLS — CYTOKINES")
print("=" * 65)

ols_cyto = run_ols(CYTO_COLS, "Cytokine")
ols_cyto.to_csv(OUTPUT_DIR / "ols_cytokine_results.csv", index=False)

print(f"\n  {'Feature':<12} {'R²':>6} {'pR²Sex':>8} "
      f"{'pR²Age':>8} {'pR²Race':>8} {'pSex':>8} {'pAge':>8} {'pRace':>8}")
print("  " + "-" * 75)
for _, row in ols_cyto.iterrows():
    ss = "✓" if row["Sig_Sex"]  else ""
    sa = "✓" if row["Sig_Age"]  else ""
    sr = "✓" if row["Sig_Race"] else ""
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
print()

# ================================================================================
# SECTION 4: RUN OLS — CELL TYPES
# ================================================================================
print("=" * 65)
print("SECTION 4: OLS — CELL TYPES")
print("=" * 65)

ols_cell = run_ols(CELL_COLS, "Cell Type")
ols_cell.to_csv(OUTPUT_DIR / "ols_celltype_results.csv", index=False)

print(f"\n  Top 15 cell types by R²:")
print(f"  {'Feature':<45} {'R²':>6} {'pR²Sex':>8} "
      f"{'pR²Age':>8} {'pR²Race':>8}")
print("  " + "-" * 75)
for _, row in ols_cell.head(15).iterrows():
    ss = "✓" if row["Sig_Sex"]  else ""
    sa = "✓" if row["Sig_Age"]  else ""
    sr = "✓" if row["Sig_Race"] else ""
    print(f"  {row['Feature']:<45} {row['R2_full']:>6.3f} "
          f"{row['Partial_R2_Sex']:>8.3f} {row['Partial_R2_Age']:>8.3f} "
          f"{row['Partial_R2_Race']:>8.3f}  {ss}{sa}{sr}")

print(f"\n  Cell types significant for sex  : "
      f"{ols_cell[ols_cell['Sig_Sex']]['Feature'].tolist()}")
print(f"  Cell types significant for age  : "
      f"{ols_cell[ols_cell['Sig_Age']]['Feature'].tolist()}")
print(f"  Cell types significant for race : "
      f"{ols_cell[ols_cell['Sig_Race']]['Feature'].tolist()}")
print()

# Combined table
ols_all = pd.concat([ols_cyto, ols_cell], ignore_index=True)
ols_all.to_csv(OUTPUT_DIR / "ols_all_features_results.csv", index=False)

# ================================================================================
# SECTION 5: GLOBAL PCA — CYTOKINES + CELL TYPES COMBINED
# ================================================================================
print("=" * 65)
print("SECTION 5: GLOBAL PCA — CYTOKINES + CELL TYPES")
print("=" * 65)

X_cyto = np.log1p(df[CYTO_COLS].values.astype(float))
X_cell = df_model[CELL_COLS].values.astype(float)

# Standardize each modality separately
def standardize_global(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0);  std[std == 0] = 1
    Xs   = (X - mean) / std
    Xc   = Xs - Xs.mean(axis=0)
    return np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)

X_cyto_sc = standardize_global(X_cyto)
X_cell_sc = standardize_global(X_cell)
X_combined = np.hstack([X_cyto_sc, X_cell_sc])

_, S, Vt        = np.linalg.svd(X_combined, full_matrices=False)
eigenvalues     = (S ** 2) / (len(df) - 1)
explained_ratio = eigenvalues / eigenvalues.sum()
cumulative_var  = np.cumsum(explained_ratio)
X_pca           = X_combined @ Vt[:3].T

print(f"  PC1: {explained_ratio[0]*100:.1f}%  "
      f"PC2: {explained_ratio[1]*100:.1f}%  "
      f"PC3: {explained_ratio[2]*100:.1f}%")
print()

# ================================================================================
# SECTION 6: VISUALIZATION
# ================================================================================
print("=" * 65)
print("SECTION 6: VISUALIZATION")
print("=" * 65)

COLOR_BAR    = "#4A7FC1"
race_colors  = {"Caucasian": "#4A7FC1", "African-American": "#E87D7D",
                "Asian": "#5BAD72", "Other": "#F4A460", "Hispanic": "#9B59B6"}
age_colors   = {"18-29": "#1f77b4", "30-39": "#2ca02c",
                "40-49": "#ff7f0e", "50-66": "#d62728"}

# ---- Figure 1: PCA colored by each covariate ----
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle("Global PCA (Cytokines + Cell Types) — colored by covariates",
              fontsize=13, fontweight="bold")

for ax, covariate, color_map, title in zip(
    axes1,
    ["SEXC", "AGEGR1C", "RACEGRP"],
    [{"Female": "#E87D7D", "Male": "#6BAED6"}, age_colors, race_colors],
    ["Sex", "Age Group", "Race"]
):
    for label, color in color_map.items():
        mask = df[covariate].values == label
        if mask.sum() > 0:
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=color, alpha=0.7, s=40, label=label)
    ax.set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}%)", fontsize=9)
    ax.set_title(f"Colored by {title}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "pca_combined_by_covariates.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: pca_combined_by_covariates.png")

# ---- Figure 2: Partial R² heatmap — cytokines ----
fig2, ax2 = plt.subplots(figsize=(5, 12))
r2_cyto = ols_cyto[["Partial_R2_Sex", "Partial_R2_Age",
                      "Partial_R2_Race"]].values
im2 = ax2.imshow(r2_cyto, cmap="YlOrRd", aspect="auto",
                 vmin=0, vmax=max(r2_cyto.max(), 0.01))
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(["Sex", "Age", "Race"], fontsize=11, fontweight="bold")
ax2.set_yticks(range(len(ols_cyto)))
ax2.set_yticklabels(ols_cyto["Feature"].tolist(), fontsize=9)
ax2.set_title("Cytokines — Partial R²\n* = p<0.05",
              fontsize=11, fontweight="bold")
plt.colorbar(im2, ax=ax2, label="Partial R²")
sig_cols = ["Sig_Sex", "Sig_Age", "Sig_Race"]
for i in range(len(ols_cyto)):
    for j in range(3):
        val = r2_cyto[i, j]
        sig = ols_cyto.iloc[i][sig_cols[j]]
        ax2.text(j, i, f"{val:.3f}" + ("*" if sig else ""),
                 ha="center", va="center", fontsize=7,
                 color="white" if val > r2_cyto.max() * 0.6 else "black")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "partial_r2_cytokines.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: partial_r2_cytokines.png")

# ---- Figure 3: Partial R² heatmap — top 20 cell types ----
top_cells = ols_cell.head(20)
fig3, ax3  = plt.subplots(figsize=(5, 12))
r2_cell    = top_cells[["Partial_R2_Sex", "Partial_R2_Age",
                          "Partial_R2_Race"]].values
im3 = ax3.imshow(r2_cell, cmap="YlOrRd", aspect="auto",
                 vmin=0, vmax=max(r2_cell.max(), 0.01))
ax3.set_xticks([0, 1, 2])
ax3.set_xticklabels(["Sex", "Age", "Race"], fontsize=11, fontweight="bold")
ax3.set_yticks(range(len(top_cells)))
ax3.set_yticklabels(
    [f.replace("CT_", "").replace("_", " ")[:35]
     for f in top_cells["Feature"].tolist()], fontsize=7)
ax3.set_title("Top 20 Cell Types — Partial R²\n* = p<0.05",
              fontsize=11, fontweight="bold")
plt.colorbar(im3, ax=ax3, label="Partial R²")
for i in range(len(top_cells)):
    for j in range(3):
        val = r2_cell[i, j]
        sig = top_cells.iloc[i][sig_cols[j]]
        ax3.text(j, i, f"{val:.3f}" + ("*" if sig else ""),
                 ha="center", va="center", fontsize=7,
                 color="white" if val > r2_cell.max() * 0.6 else "black")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "partial_r2_celltypes.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: partial_r2_celltypes.png")

# ---- Figure 4: R² comparison — cytokines vs cell types ----
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
fig4.suptitle("Total R² — Demographics explain cytokine vs cell type variance",
              fontsize=13, fontweight="bold")

for ax, ols_data, title, color in zip(
    axes4,
    [ols_cyto, ols_cell.head(25)],
    ["Cytokines", "Top 25 Cell Types"],
    ["#4A7FC1", "#5BAD72"]
):
    data_s = ols_data.sort_values("R2_full", ascending=True)
    bar_colors = ["#E87D7D" if (row["Sig_Sex"] or row["Sig_Age"] or row["Sig_Race"])
                  else "#AAAAAA"
                  for _, row in data_s.iterrows()]
    feat_labels = [f.replace("CT_", "").replace("_", " ")[:35]
                   if f.startswith("CT_") else f
                   for f in data_s["Feature"]]
    ax.barh(feat_labels, data_s["R2_full"],
            color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("R² (sex + age + race)", fontsize=10)
    ax.set_title(f"{title}\n(red = at least one covariate p<0.05)",
                 fontsize=11, fontweight="bold")
    ax.axvline(0.05, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "r2_cytokines_vs_celltypes.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: r2_cytokines_vs_celltypes.png")

# ---- Figure 5: Stacked partial R² — cytokines ----
fig5, ax5 = plt.subplots(figsize=(10, 6))
ols_s = ols_cyto.sort_values("R2_full", ascending=True)
y_pos = np.arange(len(ols_s))
ax5.barh(y_pos, ols_s["Partial_R2_Sex"],
         color="#6BAED6", alpha=0.85, label="Sex")
ax5.barh(y_pos, ols_s["Partial_R2_Age"],
         left=ols_s["Partial_R2_Sex"],
         color="#F4A460", alpha=0.85, label="Age")
ax5.barh(y_pos, ols_s["Partial_R2_Race"],
         left=ols_s["Partial_R2_Sex"] + ols_s["Partial_R2_Age"],
         color="#5BAD72", alpha=0.85, label="Race")
ax5.set_yticks(y_pos)
ax5.set_yticklabels(ols_s["Feature"], fontsize=9)
ax5.set_xlabel("Partial R²", fontsize=11)
ax5.set_title("Cytokines — Partial R² Breakdown\nSex vs Age vs Race",
              fontsize=11, fontweight="bold")
ax5.legend(fontsize=10)
ax5.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "partial_r2_stacked_cytokines.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: partial_r2_stacked_cytokines.png")

# ---- Figure 6: Stacked partial R² — top 20 cell types ----
fig6, ax6 = plt.subplots(figsize=(10, 10))
ols_c = ols_cell.head(20).sort_values("R2_full", ascending=True)
y_pos = np.arange(len(ols_c))
ax6.barh(y_pos, ols_c["Partial_R2_Sex"],
         color="#6BAED6", alpha=0.85, label="Sex")
ax6.barh(y_pos, ols_c["Partial_R2_Age"],
         left=ols_c["Partial_R2_Sex"],
         color="#F4A460", alpha=0.85, label="Age")
ax6.barh(y_pos, ols_c["Partial_R2_Race"],
         left=ols_c["Partial_R2_Sex"] + ols_c["Partial_R2_Age"],
         color="#5BAD72", alpha=0.85, label="Race")
ax6.set_yticks(y_pos)
ax6.set_yticklabels(
    [f.replace("CT_", "").replace("_", " ")[:40] for f in ols_c["Feature"]],
    fontsize=8)
ax6.set_xlabel("Partial R²", fontsize=11)
ax6.set_title("Top 20 Cell Types — Partial R² Breakdown\nSex vs Age vs Race",
              fontsize=11, fontweight="bold")
ax6.legend(fontsize=10)
ax6.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "partial_r2_stacked_celltypes.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: partial_r2_stacked_celltypes.png")

# ================================================================================
# SECTION 7: SUMMARY
# ================================================================================
print()
print("=" * 65)
print("SECTION 7: SUMMARY")
print("=" * 65)

print("\n  CYTOKINES:")
print(f"    Mean R²           : {ols_cyto['R2_full'].mean():.4f}")
print(f"    Max R²            : {ols_cyto['R2_full'].max():.4f} "
      f"({ols_cyto.loc[ols_cyto['R2_full'].idxmax(),'Feature']})")
print(f"    Mean pR² Sex      : {ols_cyto['Partial_R2_Sex'].mean():.4f}")
print(f"    Mean pR² Age      : {ols_cyto['Partial_R2_Age'].mean():.4f}")
print(f"    Mean pR² Race     : {ols_cyto['Partial_R2_Race'].mean():.4f}")
print(f"    Sig for sex       : {ols_cyto[ols_cyto['Sig_Sex']]['Feature'].tolist()}")
print(f"    Sig for age       : {ols_cyto[ols_cyto['Sig_Age']]['Feature'].tolist()}")
print(f"    Sig for race      : {ols_cyto[ols_cyto['Sig_Race']]['Feature'].tolist()}")

print("\n  CELL TYPES:")
print(f"    Mean R²           : {ols_cell['R2_full'].mean():.4f}")
print(f"    Max R²            : {ols_cell['R2_full'].max():.4f} "
      f"({ols_cell.loc[ols_cell['R2_full'].idxmax(),'Feature']})")
print(f"    Mean pR² Sex      : {ols_cell['Partial_R2_Sex'].mean():.4f}")
print(f"    Mean pR² Age      : {ols_cell['Partial_R2_Age'].mean():.4f}")
print(f"    Mean pR² Race     : {ols_cell['Partial_R2_Race'].mean():.4f}")
print(f"    Sig for sex (n)   : {ols_cell['Sig_Sex'].sum()}")
print(f"    Sig for age (n)   : {ols_cell['Sig_Age'].sum()}")
print(f"    Sig for race (n)  : {ols_cell['Sig_Race'].sum()}")

print(f"\n  COMPARISON:")
cyto_dom = max(["Sex","Age","Race"],
               key=lambda x: ols_cyto[f"Partial_R2_{x}"].mean())
cell_dom  = max(["Sex","Age","Race"],
                key=lambda x: ols_cell[f"Partial_R2_{x}"].mean())
print(f"    Dominant covariate for cytokines   : {cyto_dom}")
print(f"    Dominant covariate for cell types  : {cell_dom}")
print(f"    Cytokines mean R² vs Cell types mean R²: "
      f"{ols_cyto['R2_full'].mean():.4f} vs {ols_cell['R2_full'].mean():.4f}")

print(f"\n  Outputs : {OUTPUT_DIR}")
print(f"  Figures : {FIGURES_DIR}")
print("=" * 65)