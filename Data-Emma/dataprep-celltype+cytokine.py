from pathlib import Path
import pandas as pd
import numpy as np

# ---- Paths ----
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "Data-Emma"
OUTPUT_DIR = BASE_DIR / "Data-Emma" / "outputs" / "full_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 0) Load ----
meta_long = pd.read_csv(DATA_DIR / "grifols_data_final.csv")
cyto_raw  = pd.read_csv(DATA_DIR / "grifols_cytokine_data.csv")

# ---- 1) Subject metadata ----
meta_long = meta_long.rename(columns={"SUBJECT ID": "SUBJECT_ID"})
meta_long["SUBJECT_ID"] = meta_long["SUBJECT_ID"].astype(str).str.strip()

subject_meta = (
    meta_long
    .groupby("SUBJECT_ID", as_index=False)
    .agg({"AGE": "first", "AGEGR1C": "first", "RACEGRP": "first",
          "SEXC": "first", "Batch date": "first"})
)
print(f"Subject metadata shape: {subject_meta.shape}")

# ---- 2) Clean cytokines ----
cyto = cyto_raw.iloc[1:].copy()
cyto = cyto.rename(columns={"Unnamed: 1": "SUBJECT_ID"})
cyto["SUBJECT_ID"] = cyto["SUBJECT_ID"].astype(str).str.strip()

CYTO_COLS = [
    "IFN-gamma", "IL-12p70", "IL-13", "IL-1beta", "IL-2", "IL-4", "IL-5",
    "IL-6", "TNF-alpha", "GM-CSF", "IL-18", "IL-10", "IL-17A", "IL-21",
    "IL-22", "IL-23", "IL-27", "IL-9"
]
cyto = cyto[["SUBJECT_ID"] + CYTO_COLS].copy()
for col in CYTO_COLS:
    cyto[col] = pd.to_numeric(cyto[col], errors="coerce")

# ---- 3) Pivot cell types ----
cell_pivot = (
    meta_long
    .groupby(["SUBJECT_ID", "CELL SUBSET"])["VALUE"]
    .mean()
    .unstack(fill_value=0)
    .reset_index()
)
cell_pivot.columns = (
    ["SUBJECT_ID"] +
    [f"CT_{c.replace(' ','_').replace('+','pos').replace('-','neg').replace('/','_')}"
     for c in cell_pivot.columns[1:]]
)
CELL_COLS = [c for c in cell_pivot.columns if c.startswith("CT_")]

# ---- 4) Merge ----
df = (
    cyto
    .merge(cell_pivot,   on="SUBJECT_ID", how="inner")
    .merge(subject_meta, on="SUBJECT_ID", how="inner")
)
print(f"Full merged shape: {df.shape}")

# ---- 5) DL/2 imputation + log1p on cytokines only ----
# (cell types are proportions, not MFI — no log transform needed)
DETECTION_LIMIT = 5.0
df[CYTO_COLS] = df[CYTO_COLS].apply(
    lambda col: col.where(col > 0, DETECTION_LIMIT / 2)
)
df[CYTO_COLS] = np.log1p(df[CYTO_COLS])

# ---- 6) PCA helper ----
def run_pca(X_raw, n_components, label):
    mean = X_raw.mean(axis=0)
    std  = X_raw.std(axis=0)

    constant_mask = std == 0
    if constant_mask.sum() > 0:
        print(f"  [{label}] Dropping {constant_mask.sum()} constant columns before PCA")
    std[constant_mask] = 1
    X_scaled = (X_raw - mean) / std
    X_scaled[:, constant_mask] = 0

    X_centered = X_scaled - X_scaled.mean(axis=0)
    X_centered = np.nan_to_num(X_centered, nan=0.0, posinf=0.0, neginf=0.0)

    _, S, Vt    = np.linalg.svd(X_centered, full_matrices=False)
    eigenvalues = (S ** 2) / (len(X_scaled) - 1)
    total_var   = eigenvalues.sum()
    var_ratio   = eigenvalues / total_var
    cumvar      = np.cumsum(var_ratio)

    print(f"\n{label} PCA — Variance Explained:")
    for i, (v, cv) in enumerate(zip(var_ratio[:8], cumvar[:8])):
        print(f"  PC{i+1}: {v*100:.1f}%  (cumulative: {cv*100:.1f}%)")
    print(f"  PCs for 80% variance: {int(np.searchsorted(cumvar, 0.80)) + 1}")
    print(f"  PCs for 90% variance: {int(np.searchsorted(cumvar, 0.90)) + 1}")

    components = Vt[:n_components]
    X_pca      = X_centered @ components.T
    return X_pca, components, var_ratio, cumvar, X_scaled

# ---- 7) Run separate PCAs ----
X_cyto_pca, comp_cyto, vr_cyto, cv_cyto, X_cyto_scaled = run_pca(
    df[CYTO_COLS].values, n_components=3, label="Cytokine"
)
X_cell_pca, comp_cell, vr_cell, cv_cell, X_cell_scaled = run_pca(
    df[CELL_COLS].values, n_components=3, label="Cell Type"
)

X_combined = np.hstack([X_cyto_pca, X_cell_pca])
print(f"\nCombined matrix shape: {X_combined.shape}  (3 cytokine PCs + 3 cell type PCs)")

for i in range(3):
    df[f"CytoPCA_{i+1}"] = X_cyto_pca[:, i]
    df[f"CellPCA_{i+1}"] = X_cell_pca[:, i]

# ---- 8) Save ----
df.to_csv(OUTPUT_DIR / "full_merged.csv", index=False)
np.save(OUTPUT_DIR / "X_cytokines_scaled.npy",       X_cyto_scaled)
np.save(OUTPUT_DIR / "X_cell_scaled.npy",            X_cell_scaled)
np.save(OUTPUT_DIR / "X_cytokines_pca.npy",          X_cyto_pca)
np.save(OUTPUT_DIR / "X_cell_pca.npy",               X_cell_pca)
np.save(OUTPUT_DIR / "X_combined_pca.npy",           X_combined)
np.save(OUTPUT_DIR / "pca_components_cytokines.npy", comp_cyto)
np.save(OUTPUT_DIR / "pca_components_cell.npy",      comp_cell)
np.save(OUTPUT_DIR / "y_sex.npy",  (df["SEXC"] == "Male").astype(int).values)
np.save(OUTPUT_DIR / "y_age.npy",  df["AGEGR1C"].values)
np.save(OUTPUT_DIR / "y_race.npy", df["RACEGRP"].values)

for label, vr, cv in [("cytokines", vr_cyto, cv_cyto), ("cell", vr_cell, cv_cell)]:
    pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(vr))],
        "Variance_Explained": vr,
        "Cumulative_Variance": cv
    }).to_csv(OUTPUT_DIR / f"pca_variance_{label}.csv", index=False)

pd.DataFrame(
    comp_cyto,
    index=[f"CytoPCA_{i+1}" for i in range(3)],
    columns=CYTO_COLS
).to_csv(OUTPUT_DIR / "pca_loadings_cytokines.csv")

pd.DataFrame(
    comp_cell,
    index=[f"CellPCA_{i+1}" for i in range(3)],
    columns=CELL_COLS
).to_csv(OUTPUT_DIR / "pca_loadings_cell.csv")

print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"Final columns ({len(df.columns)}): {list(df.columns)}")