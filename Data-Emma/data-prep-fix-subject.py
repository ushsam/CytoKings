import pandas as pd
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR   = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "analysis_merged_raw.csv"
OUTPUT_FILE = BASE_DIR / "analysis_merged_subject_level_cell-type.csv"

# ============================================================================
# LOAD
# ============================================================================
df = pd.read_csv(INPUT_FILE)
print(f"Raw data shape: {df.shape}")
print(f"Unique subjects: {df['SUBJECT_ID'].nunique()}")
print(f"Unique cell subsets: {df['CELL SUBSET'].nunique()}")

# ============================================================================
# STEP 1: CYTOKINE + DEMOGRAPHIC COLUMNS
# — these are the same for every row of a given subject, just take first
# ============================================================================
CYTO_COLS = [
    "IFN-gamma", "IL-12p70", "IL-13", "IL-1beta", "IL-2", "IL-4", "IL-5",
    "IL-6", "TNF-alpha", "GM-CSF", "IL-18", "IL-10", "IL-17A", "IL-21",
    "IL-22", "IL-23", "IL-27", "IL-9"
]

DEMO_COLS = [
    "AGE", "AGEGR1N", "AGEGR1C", "RACEGRP",
    "SEXN", "SEXC", "FASFL"
]

subject_level = (
    df.groupby("SUBJECT_ID")[CYTO_COLS + DEMO_COLS]
    .first()
    .reset_index()
)

print(f"Subject-level base shape: {subject_level.shape}")

# ============================================================================
# STEP 2: PIVOT CELL SUBSETS — one column per cell subset
# ============================================================================
cell_pivot = (
    df[["SUBJECT_ID", "CELL SUBSET", "VALUE"]]
    .drop_duplicates(subset=["SUBJECT_ID", "CELL SUBSET"])
    .pivot(index="SUBJECT_ID", columns="CELL SUBSET", values="VALUE")
    .reset_index()
)

# Clean column names — prefix CT_ and replace spaces/special chars
cell_pivot.columns = (
    ["SUBJECT_ID"] +
    [
        "CT_" + str(c).strip()
                      .replace(" ", "_")
                      .replace("+", "pos")
                      .replace("-", "neg")
                      .replace("(", "")
                      .replace(")", "")
        for c in cell_pivot.columns[1:]
    ]
)

print(f"Cell pivot shape: {cell_pivot.shape}")
print(f"Sample CT columns: {cell_pivot.columns[1:6].tolist()}")

# ============================================================================
# STEP 3: MERGE
# ============================================================================
merged = subject_level.merge(cell_pivot, on="SUBJECT_ID", how="left")

print(f"\nFinal merged shape: {merged.shape}")
print(f"Columns: {merged.columns.tolist()}")
print(f"Missing values per column:\n{merged.isnull().sum()[merged.isnull().sum() > 0]}")

# ============================================================================
# SAVE
# ============================================================================
merged = merged.drop(columns=["SUBJECT_ID"])
merged.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved: {OUTPUT_FILE}")
print(f"  Rows    : {len(merged)} (one per subject)")
print(f"  Columns : {len(merged.columns)} "
      f"({len(CYTO_COLS)} cytokines + {len(DEMO_COLS)} demographics "
      f"+ {len(cell_pivot.columns)-1} cell types + SUBJECT_ID)")