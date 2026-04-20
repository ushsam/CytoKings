"""
Cluster-Cytokine Heatmap with Immune Annotation
Rows = clusters from FCM, Columns = top cytokines
Values = mean cytokine MFI per cluster
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "bio-interpretation" / "outputs" / "figures"
BIO_DIR    = BASE_DIR / "bio-interpretation" / "outputs"
FEAT_DIR   = BASE_DIR / "feature_importance_summary" / "Output"
BIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD
# ============================================================================
df = pd.read_csv(DATA_DIR / "analysis_merged_subject_level.csv")

# Extract cytokine columns dynamically
DEMO_COLS = ['SUBJECT_ID', 'AGE', 'AGEGR1N', 'AGEGR1C', 'RACEGRP',
             'SEXN', 'SEXC', 'FASFL', 'AGE_BINARY', 'Batch date']
CYTO_COLS = [c for c in df.columns
             if c not in DEMO_COLS and not c.startswith('CT_')]
print(f"Cytokines detected: {CYTO_COLS}")

X_raw = df[CYTO_COLS].values.astype(float)
X_log = np.log1p(X_raw)
y     = (df["SEXC"] == "Male").astype(int).values

# Top 10 cytokines by combined sex + age importance
feat_imp_path = FEAT_DIR / "feature_importance_comparison.csv"
if feat_imp_path.exists():
    feat_imp      = pd.read_csv(feat_imp_path)
    TOP_CYTOKINES = feat_imp.nlargest(10, "Combined")["Feature"].tolist()
    print(f"Top cytokines (combined): {TOP_CYTOKINES}")
else:
    print(f"  WARNING: feature_importance_comparison.csv not found, using defaults")
    TOP_CYTOKINES = [
        "IL-21", "IL-22", "IL-4", "IL-9", "GM-CSF",
        "IL-18", "IL-6", "IL-10", "IL-1beta", "IL-13"
    ]

# ============================================================================
# PATHWAY ANNOTATION
# ============================================================================
PATHWAY_ANNOTATION = {
    "IL-21":    "Tfh/Th17 → B cell differentiation, germinal center\n★ Top sex + age cytokine",
    "IL-22":    "Th17/NK → Mucosal barrier, tissue repair\n★ Top sex cytokine (LogReg)",
    "IL-4":     "Th2 → B cell class switching, IgE\nImportant for both sex + age",
    "IL-17A":   "Th17 → Mucosal immunity, neutrophil recruitment",
    "GM-CSF":   "T cells/Macrophages → Myeloid differentiation\nImportant for both",
    "IL-13":    "Th2 → Allergic inflammation, IgE",
    "IL-1beta": "Macrophages/DCs → Innate inflammation, inflammaging\n★ Top age cytokine (LogReg)",
    "IFN-gamma":"Th1/NK → Antiviral, macrophage activation",
    "IL-18":    "Macrophages/DCs → IFN-gamma induction, innate\nTop age cytokine (XGBoost)",
    "IL-10":    "Tregs/Macrophages → Anti-inflammatory regulation\nTop age cytokine (XGBoost)",
    "IL-6":     "Macrophages/DCs → Acute phase, Th17 differentiation\nTop age cytokine (XGBoost)",
    "IL-9":     "Th9/Mast cells → Allergic, mast cell activation\nImportant for both",
    "IL-2":     "T cells → Proliferation, Treg maintenance\nImportant for both",
    "IL-5":     "Th2/Mast cells → Eosinophil activation",
    "TNF-alpha":"Macrophages/T cells → Pro-inflammatory, apoptosis",
    "IL-12p70": "DCs/Macrophages → Th1 polarization",
    "IL-23":    "DCs → Th17 maintenance",
    "IL-27":    "DCs/Macrophages → Th1 regulation, anti-inflammatory",
}

pd.DataFrame([
    {"Cytokine": cyto, "Pathway": PATHWAY_ANNOTATION.get(cyto, "Unknown")}
    for cyto in CYTO_COLS
]).to_csv(BIO_DIR / "cytokine_pathway_annotation.csv", index=False)
print(f"✓ Saved: cytokine_pathway_annotation.csv → {BIO_DIR}")

# ============================================================================
# SOFT CLUSTER ASSIGNMENTS — FCM k=3 on global log1p PCA
# ============================================================================
try:
    import skfuzzy as fuzz
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "scikit-fuzzy", "--break-system-packages", "-q"])
    import skfuzzy as fuzz

X_global_mean   = X_log.mean(axis=0)
X_global_std    = X_log.std(axis=0)
X_global_std[X_global_std == 0] = 1
X_global_scaled = (X_log - X_global_mean) / X_global_std
X_centered      = X_global_scaled - X_global_scaled.mean(axis=0)
X_centered      = np.nan_to_num(X_centered, nan=0.0, posinf=0.0, neginf=0.0)
_, S, Vt        = np.linalg.svd(X_centered, full_matrices=False)
X_global_pca    = X_centered @ Vt[:2].T

cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
    X_global_pca.T, c=3, m=2.0,
    error=0.005, maxiter=1000, init=None, seed=42
)
hard_labels = np.argmax(u, axis=0)

print(f"FCM k=3 | FPC={fpc:.4f}")
for c in range(3):
    mask = hard_labels == c
    print(f"  Cluster {c+1}: n={mask.sum()}  "
          f"Male={y[mask].sum()}  Female={(y[mask]==0).sum()}")

# ============================================================================
# BUILD HEATMAP DATA
# ============================================================================
# Filter TOP_CYTOKINES to only those present in CYTO_COLS
TOP_CYTOKINES = [c for c in TOP_CYTOKINES if c in CYTO_COLS]
top_idx       = [CYTO_COLS.index(c) for c in TOP_CYTOKINES]
X_top         = X_log[:, top_idx]

n_clusters     = 3
heatmap_vals   = np.zeros((n_clusters, len(TOP_CYTOKINES)))
cluster_labels = []

for c in range(n_clusters):
    mask = hard_labels == c
    heatmap_vals[c] = X_top[mask].mean(axis=0)
    n_male   = y[mask].sum()
    n_female = (y[mask] == 0).sum()
    cluster_labels.append(
        f"Cluster {c+1}\n(n={mask.sum()}, M:{n_male} F:{n_female})"
    )

# Z-score across clusters per cytokine
heatmap_z = (heatmap_vals - heatmap_vals.mean(axis=0)) / \
            (heatmap_vals.std(axis=0) + 1e-8)

z_max = np.abs(heatmap_z).max()
print(f"Z-score range: {heatmap_z.min():.2f} to {heatmap_z.max():.2f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, ax_main = plt.subplots(figsize=(12, 6))

im = ax_main.imshow(heatmap_z, cmap="RdBu_r", aspect="auto",
                    vmin=-z_max, vmax=z_max)

ax_main.set_xticks(range(len(TOP_CYTOKINES)))
ax_main.set_xticklabels(TOP_CYTOKINES, rotation=45, ha="right", fontsize=10)
ax_main.set_yticks(range(n_clusters))
ax_main.set_yticklabels(cluster_labels, fontsize=10)
ax_main.set_title(
    "Cluster–Cytokine Heatmap\n"
    "Mean log1p(MFI) per FCM cluster (z-scored across clusters)",
    fontsize=12, fontweight="bold"
)

for i in range(n_clusters):
    for j in range(len(TOP_CYTOKINES)):
        ax_main.text(j, i, f"{heatmap_vals[i, j]:.2f}",
                     ha="center", va="center", fontsize=8,
                     color="white" if abs(heatmap_z[i, j]) > 1.2 else "black")

cbar = plt.colorbar(im, ax=ax_main, label="Z-score",
                    fraction=0.03, pad=0.02)
cbar.set_ticks(np.linspace(-z_max, z_max, 7))
cbar.set_ticklabels([f"{v:.1f}" for v in np.linspace(-z_max, z_max, 7)])

plt.suptitle(
    "Top Cytokines by Immune Cluster — Biological Annotation",
    fontsize=13, fontweight="bold", y=1.02
)

plt.tight_layout()
plt.savefig(FEAT_DIR / "cluster_cytokine_heatmap.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: cluster_cytokine_heatmap.png")

# ============================================================================
# SAVE SUMMARY TABLE
# ============================================================================
summary_rows = []
for j, cyto in enumerate(TOP_CYTOKINES):
    row = {
        "Cytokine": cyto,
        "Pathway":  PATHWAY_ANNOTATION.get(cyto, "Unknown")
    }
    for c in range(n_clusters):
        row[f"Cluster{c+1}_Mean_MFI"] = round(heatmap_vals[c, j], 4)
        row[f"Cluster{c+1}_Zscore"]   = round(heatmap_z[c, j], 4)
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(
    OUTPUT_DIR / "cluster_cytokine_summary.csv", index=False
)
print("✓ Saved: cluster_cytokine_summary.csv")