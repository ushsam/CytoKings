"""
Feature Importance Comparison — Sex vs Age Prediction
Pulls from existing output files and creates a side-by-side summary figure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR       = Path(__file__).resolve().parent.parent
PCA_DIR        = BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokines_only"
SEX_XGB_DIR    = BASE_DIR / "XGBoost" / "cytokine-only" / "sex" / "Outputs"
SEX_LR_DIR     = BASE_DIR / "XGBoost" / "cytokine-only"/"sex" / "sex_enhanced" / "Outputs"
AGE_DIR        = BASE_DIR / "XGBoost" / "cytokine-only" / "age" / "Outputs" / "binary_age"
OUTPUT_DIR     = BASE_DIR / "XGBoost" / "cytokine-only" / "feature_importance_summary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD
# ============================================================================
sex_xgb_imp = pd.read_csv(SEX_XGB_DIR / "feature_importance.csv")
sex_lr_imp  = pd.read_csv(SEX_LR_DIR  / "feature_importance_enhanced.csv")
age_xgb_imp = pd.read_csv(AGE_DIR     / "results_xgb_importance_age.csv")
age_lr_imp  = pd.read_csv(AGE_DIR     / "results_lr_coefficients_age.csv")
sex_ttest   = pd.read_csv(PCA_DIR     / "cytokine_sex_comparison.csv")
age_anova   = pd.read_csv(AGE_DIR     / "results_anova_age.csv")

# ============================================================================
# NORMALIZE IMPORTANCE SCORES 0-1 for fair comparison
# ============================================================================
def normalize(df, col):
    vals = df[col].abs()
    return vals / vals.max() if vals.max() > 0 else vals

sex_xgb_imp["norm"] = normalize(sex_xgb_imp, "Importance")
age_xgb_imp["norm"] = normalize(age_xgb_imp, "Importance")
sex_lr_imp["norm"]  = normalize(sex_lr_imp,  "Importance")
age_lr_imp["norm"]  = normalize(age_lr_imp,  "Coefficient")

# Significance flags
sex_sig = dict(zip(sex_ttest["Cytokine"], sex_ttest["Significant"]))
age_sig = dict(zip(age_anova["Cytokine"], age_anova["Significant"]))

# ============================================================================
# BUILD MERGED DATAFRAMES
# ============================================================================

# XGBoost merged
merged = sex_xgb_imp[["Cytokine", "norm"]].rename(
    columns={"Cytokine": "Feature", "norm": "Sex_norm"}
).merge(
    age_xgb_imp[["Feature", "norm"]].rename(columns={"norm": "Age_norm"}),
    on="Feature", how="outer"
).fillna(0)

merged["Sex_Sig"]  = merged["Feature"].map(sex_sig).fillna(False)
merged["Age_Sig"]  = merged["Feature"].map(age_sig).fillna(False)
merged["Combined"] = merged["Sex_norm"] + merged["Age_norm"]
merged = merged.sort_values("Combined", ascending=True)
merged.to_csv(OUTPUT_DIR / "feature_importance_comparison.csv", index=False)

# LR merged
merged_lr = sex_lr_imp[["Cytokine", "norm"]].rename(
    columns={"Cytokine": "Feature", "norm": "Sex_LR_norm"}
).merge(
    age_lr_imp[["Feature", "norm"]].rename(columns={"norm": "Age_LR_norm"}),
    on="Feature", how="outer"
).fillna(0)

merged_lr["Sex_Sig"]  = merged_lr["Feature"].map(sex_sig).fillna(False)
merged_lr["Age_Sig"]  = merged_lr["Feature"].map(age_sig).fillna(False)
merged_lr["Combined"] = merged_lr["Sex_LR_norm"] + merged_lr["Age_LR_norm"]
merged_lr = merged_lr.sort_values("Combined", ascending=True)
merged_lr.to_csv(OUTPUT_DIR / "feature_importance_lr_comparison.csv", index=False)

# ============================================================================
# FIGURE 1: XGBoost — Sex vs Age side-by-side bar chart
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
fig.suptitle(
    "Cytokine Importance: XGBoost — Sex vs Age Prediction\n"
    "* = statistically significant (Welch's t-test / ANOVA p<0.05)",
    fontsize=13, fontweight="bold"
)

features  = merged["Feature"].tolist()
y_pos     = np.arange(len(features))
COLOR_SEX = "#6BAED6"
COLOR_AGE = "#F4A460"

ax1 = axes[0]
ax1.barh(y_pos, merged["Sex_norm"], color=COLOR_SEX,
         alpha=0.85, edgecolor="white")
for i, (val, sig) in enumerate(zip(merged["Sex_norm"], merged["Sex_Sig"])):
    if sig:
        ax1.text(val + 0.02, i, "*", va="center", fontsize=12,
                 color="black", fontweight="bold")
ax1.set_yticks(y_pos)
ax1.set_yticklabels(features, fontsize=9)
ax1.set_xlabel("Normalized Importance", fontsize=10)
ax1.set_title("Sex Prediction\n(XGBoost Feature Importance)",
              fontsize=11, fontweight="bold", color=COLOR_SEX)
ax1.set_xlim(0, 1.3)
ax1.axvline(0, color="black", linewidth=0.5)
ax1.grid(axis="x", alpha=0.3)

ax2 = axes[1]
ax2.barh(y_pos, merged["Age_norm"], color=COLOR_AGE,
         alpha=0.85, edgecolor="white")
for i, (val, sig) in enumerate(zip(merged["Age_norm"], merged["Age_Sig"])):
    if sig:
        ax2.text(val + 0.02, i, "*", va="center", fontsize=12,
                 color="black", fontweight="bold")
ax2.set_xlabel("Normalized Importance", fontsize=10)
ax2.set_title("Age Prediction\n(XGBoost Feature Importance)",
              fontsize=11, fontweight="bold", color=COLOR_AGE)
ax2.set_xlim(0, 1.3)
ax2.axvline(0, color="black", linewidth=0.5)
ax2.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_sex_vs_age.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: feature_importance_sex_vs_age.png")

# ============================================================================
# FIGURE 2: Scatter plot — XGBoost sex vs age importance
# ============================================================================
fig2, ax = plt.subplots(figsize=(8, 8))

ax.scatter(merged["Sex_norm"], merged["Age_norm"],
           s=100, alpha=0.8, color="#4A7FC1", edgecolor="white", zorder=3)

for _, row in merged.iterrows():
    ax.annotate(row["Feature"],
                (row["Sex_norm"], row["Age_norm"]),
                textcoords="offset points", xytext=(6, 3),
                fontsize=8, color="black")

max_val = max(merged["Sex_norm"].max(), merged["Age_norm"].max())
ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3,
        linewidth=1, label="Equal importance")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
ax.axvline(0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
ax.text(0.02, 0.98, "Age-specific\n(not sex)", transform=ax.transAxes,
        fontsize=8, color="gray", va="top")
ax.text(0.98, 0.02, "Sex-specific\n(not age)", transform=ax.transAxes,
        fontsize=8, color="gray", ha="right")
ax.text(0.98, 0.98, "Important\nfor both", transform=ax.transAxes,
        fontsize=8, color="gray", ha="right", va="top")
ax.text(0.02, 0.02, "Low importance\nfor both", transform=ax.transAxes,
        fontsize=8, color="gray")
ax.set_xlabel("Sex Prediction Importance (normalized)", fontsize=11)
ax.set_ylabel("Age Prediction Importance (normalized)", fontsize=11)
ax.set_title("XGBoost: Cytokine Importance Sex vs Age\n"
             "Above diagonal = more important for age  |  "
             "Below = more important for sex",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_scatter.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: feature_importance_scatter.png")



# ============================================================================
# FIGURE 4: Logistic Regression — Sex vs Age side-by-side
# ============================================================================
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
fig4.suptitle(
    "Cytokine Importance: Logistic Regression — Sex vs Age\n"
    "* = statistically significant (p<0.05)",
    fontsize=13, fontweight="bold"
)

features_lr  = merged_lr["Feature"].tolist()
y_lr         = np.arange(len(features_lr))
COLOR_SEX_LR = "#5BAD72"
COLOR_AGE_LR = "#E87D7D"

ax4a = axes4[0]
ax4a.barh(y_lr, merged_lr["Sex_LR_norm"], color=COLOR_SEX_LR,
          alpha=0.85, edgecolor="white")
for i, (val, sig) in enumerate(zip(merged_lr["Sex_LR_norm"],
                                    merged_lr["Sex_Sig"])):
    if sig:
        ax4a.text(val + 0.02, i, "*", va="center", fontsize=12,
                  color="black", fontweight="bold")
ax4a.set_yticks(y_lr)
ax4a.set_yticklabels(features_lr, fontsize=9)
ax4a.set_xlabel("Normalized |Coefficient|", fontsize=10)
ax4a.set_title("Sex Prediction\n(Logistic Regression |coefficient|)",
               fontsize=11, fontweight="bold", color=COLOR_SEX_LR)
ax4a.set_xlim(0, 1.3)
ax4a.grid(axis="x", alpha=0.3)

ax4b = axes4[1]
ax4b.barh(y_lr, merged_lr["Age_LR_norm"], color=COLOR_AGE_LR,
          alpha=0.85, edgecolor="white")
for i, (val, sig) in enumerate(zip(merged_lr["Age_LR_norm"],
                                    merged_lr["Age_Sig"])):
    if sig:
        ax4b.text(val + 0.02, i, "*", va="center", fontsize=12,
                  color="black", fontweight="bold")
ax4b.set_xlabel("Normalized |Coefficient|", fontsize=10)
ax4b.set_title("Age Prediction\n(Logistic Regression |coefficient|)",
               fontsize=11, fontweight="bold", color=COLOR_AGE_LR)
ax4b.set_xlim(0, 1.3)
ax4b.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_lr_sex_vs_age.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: feature_importance_lr_sex_vs_age.png")

# ============================================================================
# FIGURE 3: Combined Heatmap — XGBoost + LR, Sex + Age
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(6, 10))

# Build 4-column matrix: XGB Sex, XGB Age, LR Sex, LR Age
# Align on features — use all cytokines present in either
all_features = sorted(set(merged["Feature"]) | set(merged_lr["Feature"]))

xgb_sex = merged.set_index("Feature").reindex(all_features)["Sex_norm"].fillna(0).values
xgb_age = merged.set_index("Feature").reindex(all_features)["Age_norm"].fillna(0).values
lr_sex  = merged_lr.set_index("Feature").reindex(all_features)["Sex_LR_norm"].fillna(0).values
lr_age  = merged_lr.set_index("Feature").reindex(all_features)["Age_LR_norm"].fillna(0).values

heatmap_combined = np.column_stack([xgb_sex, xgb_age, lr_sex, lr_age])

# Sort by combined total importance
sort_order = np.argsort(heatmap_combined.sum(axis=1))
heatmap_combined = heatmap_combined[sort_order]
all_features_sorted = [all_features[i] for i in sort_order]

im3 = ax3.imshow(heatmap_combined, cmap="YlOrRd", aspect="auto",
                 vmin=0, vmax=1)
ax3.set_xticks([0, 1, 2, 3])
ax3.set_xticklabels(["Sex\n(XGBoost)", "Age\n(XGBoost)",
                     "Sex\n(LogReg)", "Age\n(LogReg)"], fontsize=9)
ax3.set_yticks(range(len(all_features_sorted)))
ax3.set_yticklabels(all_features_sorted, fontsize=9)
ax3.set_title("Feature Importance Heatmap\nXGBoost + Logistic Regression\n(normalized 0–1)",
              fontsize=11, fontweight="bold")
plt.colorbar(im3, ax=ax3, label="Normalized Importance")

col_names = ["Sex_XGB", "Age_XGB", "Sex_LR", "Age_LR"]
for i in range(len(all_features_sorted)):
    for j in range(4):
        val = heatmap_combined[i, j]
        ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=7,
                 color="white" if val > 0.6 else "black")

# Add vertical divider between XGBoost and LR columns
ax3.axvline(1.5, color="white", linewidth=2)
ax3.text(0.5, -0.8, "XGBoost", ha="center", fontsize=9,
         color="#4A7FC1", fontweight="bold",
         transform=ax3.get_xaxis_transform())
ax3.text(2.5, -0.8, "Logistic Regression", ha="center", fontsize=9,
         color="#5BAD72", fontweight="bold",
         transform=ax3.get_xaxis_transform())

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance_heatmap_combined.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: feature_importance_heatmap_combined.png")

# ============================================================================
# SUMMARY PRINT
# ============================================================================
print(f"\n✓ All figures saved to: {OUTPUT_DIR}")
print("\nTop cytokines for SEX (XGBoost):")
print(merged.nlargest(5, "Sex_norm")[["Feature", "Sex_norm", "Sex_Sig"]].to_string(index=False))
print("\nTop cytokines for AGE (XGBoost):")
print(merged.nlargest(5, "Age_norm")[["Feature", "Age_norm", "Age_Sig"]].to_string(index=False))
print("\nTop cytokines for SEX (LogReg):")
print(merged_lr.nlargest(5, "Sex_LR_norm")[["Feature", "Sex_LR_norm", "Sex_Sig"]].to_string(index=False))
print("\nTop cytokines for AGE (LogReg):")
print(merged_lr.nlargest(5, "Age_LR_norm")[["Feature", "Age_LR_norm", "Age_Sig"]].to_string(index=False))
print("\nCytokines important for BOTH — XGBoost (top 50% for both):")
both = merged[(merged["Sex_norm"] > 0.5) & (merged["Age_norm"] > 0.5)]
print(both[["Feature", "Sex_norm", "Age_norm"]].to_string(index=False)
      if len(both) > 0 else "  None in top 50% for both")
print("\nCytokines important for BOTH — LogReg (top 50% for both):")
both_lr = merged_lr[(merged_lr["Sex_LR_norm"] > 0.5) & (merged_lr["Age_LR_norm"] > 0.5)]
print(both_lr[["Feature", "Sex_LR_norm", "Age_LR_norm"]].to_string(index=False)
      if len(both_lr) > 0 else "  None in top 50% for both")