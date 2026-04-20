"""
================================================================================
Cytokine Expression — Covariate Analysis
Sex, Age, and Race as covariates of cytokine expression
Using OLS linear models + MANOVA to quantify demographic variance explained
================================================================================
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

# ================================================================================
# PATHS
# ================================================================================
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "Data-Emma"
OUTPUT_DIR  = BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "covariate_analysis"
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

print(f"  Subjects   : {len(df)}")
print(f"  Cytokines  : {len(CYTO_COLS)} — {CYTO_COLS}")
print(f"  Sex        : {df['SEXC'].value_counts().to_dict()}")
print(f"  Age groups : {df['AGEGR1C'].value_counts().to_dict()}")
print(f"  Race       : {df['RACEGRP'].value_counts().to_dict()}")
print()

# Log1p transform cytokines
X_log = np.log1p(df[CYTO_COLS].values.astype(float))
df_log = df.copy()
for i, cyto in enumerate(CYTO_COLS):
    df_log[cyto] = X_log[:, i]

# Binary age label
df_log['AGE_BINARY'] = df_log['AGEGR1C'].apply(
    lambda x: 0 if x in ['18-29', '30-39'] else 1
)

# ================================================================================
# SECTION 2: PER-CYTOKINE OLS — SEX, AGE, RACE AS COVARIATES
# ================================================================================
print("=" * 65)
print("SECTION 2: PER-CYTOKINE OLS — SEX + AGE + RACE")
print("=" * 65)

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "statsmodels", "--break-system-packages", "-q"])
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

ols_results = []

for cyto in CYTO_COLS:
    # Sanitize column name for formula
    safe_cyto = cyto.replace("-", "_").replace(" ", "_")
    df_log[safe_cyto] = df_log[cyto]

    formula = f"{safe_cyto} ~ C(SEXC) + C(AGEGR1C) + C(RACEGRP)"
    model   = smf.ols(formula, data=df_log).fit()

    # Extract p-values for each covariate
    pval_sex  = model.pvalues.get("C(SEXC)[T.Male]", np.nan)
    pval_age  = min([v for k, v in model.pvalues.items()
                     if "AGEGR1C" in k] or [np.nan])
    pval_race = min([v for k, v in model.pvalues.items()
                     if "RACEGRP" in k] or [np.nan])

    # Partial R² per covariate via ANOVA type II
    anova_table = sm.stats.anova_lm(model, typ=2)
    total_ss    = anova_table["sum_sq"].sum()

    ss_sex  = anova_table.loc["C(SEXC)",   "sum_sq"] if "C(SEXC)"   in anova_table.index else 0
    ss_age  = anova_table.loc["C(AGEGR1C)","sum_sq"] if "C(AGEGR1C)" in anova_table.index else 0
    ss_race = anova_table.loc["C(RACEGRP)","sum_sq"] if "C(RACEGRP)" in anova_table.index else 0

    partial_r2_sex  = ss_sex  / total_ss if total_ss > 0 else 0
    partial_r2_age  = ss_age  / total_ss if total_ss > 0 else 0
    partial_r2_race = ss_race / total_ss if total_ss > 0 else 0

    ols_results.append({
        "Cytokine":        cyto,
        "R2_full":         round(model.rsquared, 4),
        "R2_adj":          round(model.rsquared_adj, 4),
        "Partial_R2_Sex":  round(partial_r2_sex,  4),
        "Partial_R2_Age":  round(partial_r2_age,  4),
        "Partial_R2_Race": round(partial_r2_race, 4),
        "Pval_Sex":        round(pval_sex,  6) if not np.isnan(pval_sex)  else np.nan,
        "Pval_Age":        round(pval_age,  6) if not np.isnan(pval_age)  else np.nan,
        "Pval_Race":       round(pval_race, 6) if not np.isnan(pval_race) else np.nan,
        "Sig_Sex":         pval_sex  < 0.05 if not np.isnan(pval_sex)  else False,
        "Sig_Age":         pval_age  < 0.05 if not np.isnan(pval_age)  else False,
        "Sig_Race":        pval_race < 0.05 if not np.isnan(pval_race) else False,
    })

ols_df = pd.DataFrame(ols_results).sort_values("R2_full", ascending=False)
ols_df.to_csv(OUTPUT_DIR / "ols_covariate_results.csv", index=False)

print(f"\n  {'Cytokine':<12} {'R²':>6} {'R²adj':>6} "
      f"{'pR²Sex':>8} {'pR²Age':>8} {'pR²Race':>8} "
      f"{'pSex':>8} {'pAge':>8} {'pRace':>8}")
print("  " + "-" * 80)
for _, row in ols_df.iterrows():
    sig_s = "✓" if row["Sig_Sex"]  else ""
    sig_a = "✓" if row["Sig_Age"]  else ""
    sig_r = "✓" if row["Sig_Race"] else ""
    print(f"  {row['Cytokine']:<12} {row['R2_full']:>6.3f} {row['R2_adj']:>6.3f} "
          f"{row['Partial_R2_Sex']:>8.3f} {row['Partial_R2_Age']:>8.3f} "
          f"{row['Partial_R2_Race']:>8.3f} "
          f"{row['Pval_Sex']:>7.4f}{sig_s} {row['Pval_Age']:>7.4f}{sig_a} "
          f"{row['Pval_Race']:>7.4f}{sig_r}")

print(f"\n  Cytokines significantly associated with sex  (p<0.05): "
      f"{ols_df[ols_df['Sig_Sex']]['Cytokine'].tolist()}")
print(f"  Cytokines significantly associated with age  (p<0.05): "
      f"{ols_df[ols_df['Sig_Age']]['Cytokine'].tolist()}")
print(f"  Cytokines significantly associated with race (p<0.05): "
      f"{ols_df[ols_df['Sig_Race']]['Cytokine'].tolist()}")
print()

# ================================================================================
# SECTION 3: GLOBAL PCA — color by each covariate
# ================================================================================
print("=" * 65)
print("SECTION 3: GLOBAL PCA — COLORED BY COVARIATES")
print("=" * 65)

X_global_mean   = X_log.mean(axis=0)
X_global_std    = X_log.std(axis=0);  X_global_std[X_global_std == 0] = 1
X_global_scaled = (X_log - X_global_mean) / X_global_std
X_centered      = X_global_scaled - X_global_scaled.mean(axis=0)
X_centered      = np.nan_to_num(X_centered, nan=0.0, posinf=0.0, neginf=0.0)
_, S, Vt        = np.linalg.svd(X_centered, full_matrices=False)

eigenvalues     = (S ** 2) / (len(df) - 1)
explained_ratio = eigenvalues / eigenvalues.sum()
cumulative_var  = np.cumsum(explained_ratio)
X_global_pca    = X_centered @ Vt[:3].T

n_pcs_80 = int(np.searchsorted(cumulative_var, 0.80)) + 1

print(f"  PC1: {explained_ratio[0]*100:.1f}%  "
      f"PC2: {explained_ratio[1]*100:.1f}%  "
      f"PC3: {explained_ratio[2]*100:.1f}%")
print(f"  PCs for 80% variance: {n_pcs_80}")
print()

# Save loadings
loadings_df = pd.DataFrame(
    Vt[:3].T, index=CYTO_COLS, columns=["PC1", "PC2", "PC3"]
).round(4)
loadings_df.to_csv(OUTPUT_DIR / "pca_loadings_global.csv")

# ================================================================================
# SECTION 4: VISUALIZATION
# ================================================================================
print("=" * 65)
print("SECTION 4: VISUALIZATION")
print("=" * 65)

COLOR_BAR = "#4A7FC1"

# ---- Figure 1: PCA colored by each covariate ----
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle("Global PCA — PC1 vs PC2 colored by demographic covariates",
              fontsize=13, fontweight="bold")

# Sex
ax = axes1[0]
for label, val, color in [("Female", "Female", "#E87D7D"),
                            ("Male",   "Male",   "#6BAED6")]:
    mask = df["SEXC"].values == val
    ax.scatter(X_global_pca[mask, 0], X_global_pca[mask, 1],
               c=color, alpha=0.7, s=40, label=label)
ax.set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}%)", fontsize=9)
ax.set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}%)", fontsize=9)
ax.set_title("Colored by Sex", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

# Age group
ax = axes1[1]
age_colors = {"18-29": "#1f77b4", "30-39": "#2ca02c",
              "40-49": "#ff7f0e", "50-66": "#d62728"}
for age_grp in sorted(df["AGEGR1C"].unique()):
    mask  = df["AGEGR1C"].values == age_grp
    color = age_colors.get(age_grp, "#999999")
    ax.scatter(X_global_pca[mask, 0], X_global_pca[mask, 1],
               c=color, alpha=0.7, s=40, label=age_grp)
ax.set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}%)", fontsize=9)
ax.set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}%)", fontsize=9)
ax.set_title("Colored by Age Group", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

# Race
ax = axes1[2]
race_colors_map = {"Caucasian":       "#4A7FC1",
                   "African-American": "#E87D7D",
                   "Asian":           "#5BAD72",
                   "Other":           "#F4A460",
                   "Hispanic":        "#9B59B6"}
for race in sorted(df["RACEGRP"].unique()):
    mask  = df["RACEGRP"].values == race
    color = race_colors_map.get(race, "#999999")
    ax.scatter(X_global_pca[mask, 0], X_global_pca[mask, 1],
               c=color, alpha=0.7, s=40, label=race)
ax.set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}%)", fontsize=9)
ax.set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}%)", fontsize=9)
ax.set_title("Colored by Race", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "pca_by_covariates.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: pca_by_covariates.png")

# ---- Figure 2: Partial R² heatmap — cytokines x covariates ----
fig2, ax2 = plt.subplots(figsize=(5, 12))

r2_matrix = ols_df[["Partial_R2_Sex", "Partial_R2_Age",
                     "Partial_R2_Race"]].values
cyto_order = ols_df["Cytokine"].tolist()

im2 = ax2.imshow(r2_matrix, cmap="YlOrRd", aspect="auto",
                 vmin=0, vmax=r2_matrix.max())
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(["Sex", "Age", "Race"], fontsize=11, fontweight="bold")
ax2.set_yticks(range(len(cyto_order)))
ax2.set_yticklabels(cyto_order, fontsize=9)
ax2.set_title("Partial R² per Cytokine\n(variance explained by each covariate)",
              fontsize=11, fontweight="bold")
plt.colorbar(im2, ax=ax2, label="Partial R²")

for i in range(len(cyto_order)):
    for j in range(3):
        val = r2_matrix[i, j]
        sig_cols = ["Sig_Sex", "Sig_Age", "Sig_Race"]
        sig = ols_df.iloc[i][sig_cols[j]]
        txt = f"{val:.3f}" + ("*" if sig else "")
        ax2.text(j, i, txt, ha="center", va="center",
                 fontsize=7,
                 color="white" if val > r2_matrix.max() * 0.6 else "black")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "partial_r2_heatmap.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: partial_r2_heatmap.png")

# ---- Figure 3: Full R² bar chart — how much total variance each cytokine's
#      expression is explained by sex + age + race together ----
fig3, ax3 = plt.subplots(figsize=(10, 6))
ols_sorted = ols_df.sort_values("R2_full", ascending=True)
colors_bar = ["#E87D7D" if row["Sig_Sex"] or row["Sig_Age"] or row["Sig_Race"]
              else "#AAAAAA"
              for _, row in ols_sorted.iterrows()]
ax3.barh(ols_sorted["Cytokine"], ols_sorted["R2_full"],
         color=colors_bar, alpha=0.85, edgecolor="white")
ax3.set_xlabel("R² (sex + age + race combined)", fontsize=11)
ax3.set_title("Total Variance in Cytokine Expression\nExplained by Demographics\n"
              "(red = at least one covariate p<0.05)",
              fontsize=11, fontweight="bold")
ax3.axvline(0.05, color="gray", linestyle="--", alpha=0.5, label="R²=0.05")
ax3.legend(fontsize=9)
ax3.grid(axis="x", alpha=0.3)
for i, (_, row) in enumerate(ols_sorted.iterrows()):
    ax3.text(row["R2_full"] + 0.002, i, f"{row['R2_full']:.3f}",
             va="center", fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "r2_by_cytokine.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: r2_by_cytokine.png")

# ---- Figure 4: Stacked bar — partial R² breakdown per cytokine ----
fig4, ax4 = plt.subplots(figsize=(10, 6))
ols_s = ols_df.sort_values("R2_full", ascending=True)
y_pos = np.arange(len(ols_s))
ax4.barh(y_pos, ols_s["Partial_R2_Sex"],
         color="#6BAED6", alpha=0.85, label="Sex")
ax4.barh(y_pos, ols_s["Partial_R2_Age"],
         left=ols_s["Partial_R2_Sex"],
         color="#F4A460", alpha=0.85, label="Age")
ax4.barh(y_pos, ols_s["Partial_R2_Race"],
         left=ols_s["Partial_R2_Sex"] + ols_s["Partial_R2_Age"],
         color="#5BAD72", alpha=0.85, label="Race")
ax4.set_yticks(y_pos)
ax4.set_yticklabels(ols_s["Cytokine"], fontsize=9)
ax4.set_xlabel("Partial R²", fontsize=11)
ax4.set_title("Partial R² Breakdown per Cytokine\n"
              "How much variance is explained by sex vs age vs race",
              fontsize=11, fontweight="bold")
ax4.legend(fontsize=10)
ax4.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "partial_r2_stacked.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: partial_r2_stacked.png")

# ---- Figure 5: Scree plot ----
fig5, ax5 = plt.subplots(figsize=(8, 5))
x_pcs = range(1, len(explained_ratio) + 1)
ax5.bar(x_pcs, explained_ratio * 100, color=COLOR_BAR, alpha=0.7)
ax5b = ax5.twinx()
ax5b.plot(x_pcs, cumulative_var * 100, color="darkred",
          marker="o", markersize=4, linewidth=1.5)
ax5b.axhline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax5b.set_ylabel("Cumulative Variance (%)", fontsize=9, color="darkred")
ax5b.tick_params(axis="y", labelcolor="darkred", labelsize=8)
ax5b.set_ylim(0, 105)
ax5.set_xlabel("Principal Component", fontsize=9)
ax5.set_ylabel("Explained Variance (%)", fontsize=9)
ax5.set_title("Scree Plot (log1p cytokines)", fontsize=11, fontweight="bold")
ax5.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "scree_plot.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("  Figure saved: scree_plot.png")

# ================================================================================
# SECTION 5: SUMMARY
# ================================================================================
print()
print("=" * 65)
print("SECTION 5: SUMMARY")
print("=" * 65)
print(f"\n  Mean R² across all cytokines : {ols_df['R2_full'].mean():.4f}")
print(f"  Max R² (single cytokine)     : "
      f"{ols_df['R2_full'].max():.4f} ({ols_df.loc[ols_df['R2_full'].idxmax(),'Cytokine']})")
print(f"\n  Mean partial R² — Sex  : {ols_df['Partial_R2_Sex'].mean():.4f}")
print(f"  Mean partial R² — Age  : {ols_df['Partial_R2_Age'].mean():.4f}")
print(f"  Mean partial R² — Race : {ols_df['Partial_R2_Race'].mean():.4f}")

dominant = max(["Sex", "Age", "Race"],
               key=lambda x: ols_df[f"Partial_R2_{x}"].mean())
print(f"\n  → Dominant covariate: {dominant}")
print(f"\n  Cytokines with any significant covariate (p<0.05):")
sig_any = ols_df[ols_df["Sig_Sex"] | ols_df["Sig_Age"] | ols_df["Sig_Race"]]
for _, row in sig_any.iterrows():
    sigs = []
    if row["Sig_Sex"]:  sigs.append(f"sex(p={row['Pval_Sex']:.3f})")
    if row["Sig_Age"]:  sigs.append(f"age(p={row['Pval_Age']:.3f})")
    if row["Sig_Race"]: sigs.append(f"race(p={row['Pval_Race']:.3f})")
    print(f"    {row['Cytokine']:<12} R²={row['R2_full']:.3f}  "
          f"significant for: {', '.join(sigs)}")

print(f"\n  Outputs : {OUTPUT_DIR}")
print(f"  Figures : {FIGURES_DIR}")
print("=" * 65)