from pathlib import Path
import pandas as pd
import numpy as np

# ---- Paths ----
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "Data-Emma"
OUTPUT_DIR = BASE_DIR / "Data-Emma" / "outputs" / "cytokines_only"
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

# ---- 3) Merge ----
df = cyto.merge(subject_meta, on="SUBJECT_ID", how="inner")
print(f"Cytokines-only merged shape: {df.shape}")

# ---- 4) Log1p transform + DL/2 imputation ----
DETECTION_LIMIT = 5.0  # adjust to your assay's actual detection limit
df[CYTO_COLS] = df[CYTO_COLS].apply(
    lambda col: col.where(col > 0, DETECTION_LIMIT / 2)
)
df[CYTO_COLS] = np.log1p(df[CYTO_COLS])

# ---- 5) PCA from scratch ----
X = df[CYTO_COLS].values
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_std[X_std == 0] = 1
X_scaled = (X - X_mean) / X_std

X_centered = X_scaled - X_scaled.mean(axis=0)
_, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

eigenvalues  = (S ** 2) / (len(X_scaled) - 1)
total_var    = eigenvalues.sum()
var_ratio    = eigenvalues / total_var
cumvar       = np.cumsum(var_ratio)

print("\nCytokine PCA — Variance Explained:")
for i, (v, cv) in enumerate(zip(var_ratio[:8], cumvar[:8])):
    print(f"  PC{i+1}: {v*100:.1f}%  (cumulative: {cv*100:.1f}%)")

n_pcs_80 = int(np.searchsorted(cumvar, 0.80)) + 1
n_pcs_90 = int(np.searchsorted(cumvar, 0.90)) + 1
print(f"\n  PCs for 80% variance: {n_pcs_80}")
print(f"  PCs for 90% variance: {n_pcs_90}")

# Keep top 5 PCs
N_COMPONENTS = 5
components   = Vt[:N_COMPONENTS]
X_pca        = X_centered @ components.T

pca_cols = [f"CytoPCA_{i+1}" for i in range(N_COMPONENTS)]
df_pca   = pd.DataFrame(X_pca, columns=pca_cols)
df_out   = pd.concat([df.reset_index(drop=True), df_pca], axis=1)

# ---- 6) Save ----
df_out.to_csv(OUTPUT_DIR / "cytokines_merged.csv", index=False)
np.save(OUTPUT_DIR / "X_cytokines_scaled.npy", X_scaled)
np.save(OUTPUT_DIR / "X_cytokines_pca.npy",    X_pca)
np.save(OUTPUT_DIR / "pca_components_cytokines.npy", components)
np.save(OUTPUT_DIR / "y_sex.npy",   (df["SEXC"] == "Male").astype(int).values)
np.save(OUTPUT_DIR / "y_age.npy",   df["AGEGR1C"].values)
np.save(OUTPUT_DIR / "y_race.npy",  df["RACEGRP"].values)

pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(var_ratio))],
    "Variance_Explained": var_ratio,
    "Cumulative_Variance": cumvar
}).to_csv(OUTPUT_DIR / "pca_variance_cytokines.csv", index=False)

pd.DataFrame(
    components,
    index=pca_cols[:N_COMPONENTS],
    columns=CYTO_COLS
).to_csv(OUTPUT_DIR / "pca_loadings_cytokines.csv")

print(f"\nOutputs saved to: {OUTPUT_DIR}")