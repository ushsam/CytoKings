"""
Cluster-Cytokine Heatmap with Immune Annotation
Rows = clusters from FCM, Columns = top cytokines
Values = mean cytokine MFI per cluster
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "Data"
PCA_DIR    = BASE_DIR / "PCA+KNN-Emma" / "Outputs" / "cytokines_only"
OUTPUT_DIR = BASE_DIR / "bio-interpretation" / "outputs" / "figures"
BIO_DIR = BASE_DIR / "bio-interpretation" / "outputs"
BIO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD
# ============================================================================
df = pd.read_csv(DATA_DIR / "analysis_merged_subject_level.csv")

CYTO_COLS = [
    "IFN-gamma", "IL-12p70", "IL-13", "IL-1beta", "IL-2", "IL-4", "IL-5",
    "IL-6", "TNF-alpha", "GM-CSF", "IL-18", "IL-10", "IL-17A", "IL-21",
    "IL-22", "IL-23", "IL-27", "IL-9"
]

X_raw = df[CYTO_COLS].values.astype(float)
X_log = np.log1p(X_raw)
y     = (df["SEXC"] == "Male").astype(int).values

# Top cytokines from your feature importance analysis
TOP_CYTOKINES = [
    "IL-5", "GM-CSF", "IL-4", "IL-17A", "IL-21",
    "IL-22", "IL-13", "IL-1beta", "IFN-gamma", "IL-18"
]

# Immune pathway annotation per cytokine
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

# Save pathway annotation to bio-interpretation folder
pd.DataFrame([
    {"Cytokine": cyto, "Pathway": pathway}
    for cyto, pathway in PATHWAY_ANNOTATION.items()
]).to_csv(BIO_DIR / "cytokine_pathway_annotation.png", index=False)
print(f"✓ Saved: cytokine_pathway_annotation.csv → {BIO_DIR}")

# ============================================================================
# SOFT CLUSTER ASSIGNMENTS
# — Load FCM hard labels from saved outputs
# ============================================================================
try:
    import skfuzzy as fuzz
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "scikit-fuzzy", "--break-system-packages", "-q"])
    import skfuzzy as fuzz

# Re-run FCM at k=3 on global log1p PCA space to get cluster labels
X_global_mean   = X_log.mean(axis=0)
X_global_std    = X_log.std(axis=0);  X_global_std[X_global_std == 0] = 1
X_global_scaled = (X_log - X_global_mean) / X_global_std
X_centered      = X_global_scaled - X_global_scaled.mean(axis=0)
X_centered      = np.nan_to_num(X_centered, nan=0.0, posinf=0.0, neginf=0.0)
_, S, Vt        = np.linalg.svd(X_centered, full_matrices=False)
X_global_pca    = X_centered @ Vt[:2].T   # top 2 PCs

cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
    X_global_pca.T, c=3, m=2.0,
    error=0.005, maxiter=1000, init=None, seed=42
)
hard_labels = np.argmax(u, axis=0)

print(f"FCM k=3 | FPC={fpc:.4f}")
for c in range(3):
    mask = hard_labels == c
    n_older = (df["AGEGR1C"].values[mask] != "18-29").sum()
    print(f"  Cluster {c+1}: n={mask.sum()}  "
          f"Male={y[mask].sum()}  Female={(y[mask]==0).sum()}")

# ============================================================================
# BUILD HEATMAP DATA
# mean log1p MFI per cluster for top cytokines
# ============================================================================
top_idx  = [CYTO_COLS.index(c) for c in TOP_CYTOKINES]
X_top    = X_log[:, top_idx]

n_clusters   = 3
heatmap_vals = np.zeros((n_clusters, len(TOP_CYTOKINES)))
cluster_labels = []

for c in range(n_clusters):
    mask = hard_labels == c
    heatmap_vals[c] = X_top[mask].mean(axis=0)
    n_male   = y[mask].sum()
    n_female = (y[mask] == 0).sum()
    cluster_labels.append(
        f"Cluster {c+1}\n(n={mask.sum()}, M:{n_male} F:{n_female})"
    )

# Z-score across clusters per cytokine for better contrast
heatmap_z = (heatmap_vals - heatmap_vals.mean(axis=0)) / \
            (heatmap_vals.std(axis=0) + 1e-8)

# ============================================================================
# VISUALIZATION
# ============================================================================
fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)

# Main heatmap
ax_main = fig.add_subplot(gs[0])
z_max = np.abs(heatmap_z).max()
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

# Annotate cells with mean MFI values
for i in range(n_clusters):
    for j in range(len(TOP_CYTOKINES)):
        ax_main.text(j, i, f"{heatmap_vals[i, j]:.2f}",
                     ha="center", va="center", fontsize=8,
                     color="white" if abs(heatmap_z[i, j]) > 1.2 else "black")

plt.colorbar(im, ax=ax_main, label=f"Z-score (max={z_max:.1f})",
             fraction=0.03, pad=0.02)

# Pathway annotation panel
ax_ann = fig.add_subplot(gs[1])
ax_ann.axis("off")

pathway_text = "Immune Pathway\nAnnotation\n" + "─" * 22 + "\n"
for cyto in TOP_CYTOKINES:
    pathway_text += f"{cyto:<10}: {PATHWAY_ANNOTATION[cyto]}\n"

ax_ann.text(0.05, 0.95, pathway_text,
            transform=ax_ann.transAxes,
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

plt.suptitle(
    "Top Cytokines by Immune Cluster — Biological Annotation",
    fontsize=13, fontweight="bold", y=1.02
)

plt.savefig(OUTPUT_DIR / "cluster_cytokine_heatmap.png",
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ Saved: cluster_cytokine_heatmap.png")

# ============================================================================
# SAVE SUMMARY TABLE
# ============================================================================
summary_rows = []
for j, cyto in enumerate(TOP_CYTOKINES):
    row = {"Cytokine": cyto, "Pathway": PATHWAY_ANNOTATION[cyto]}
    for c in range(n_clusters):
        row[f"Cluster{c+1}_Mean_MFI"] = round(heatmap_vals[c, j], 4)
        row[f"Cluster{c+1}_Zscore"]   = round(heatmap_z[c, j], 4)
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(
    OUTPUT_DIR / "cluster_cytokine_summary.csv", index=False
)
print("✓ Saved: cluster_cytokine_summary.csv")