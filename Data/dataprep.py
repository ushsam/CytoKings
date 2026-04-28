import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ---- PATHS ----
try:
    BASE_PATH = Path(__file__).resolve().parent
except NameError:
    BASE_PATH = Path.cwd()

REPO_PATH  = BASE_PATH.parent if BASE_PATH.name != "CytoKings" else BASE_PATH
DATA_PATH  = REPO_PATH / "Data"

INPUT_META  = DATA_PATH / "grifols_data_final.csv"
INPUT_CYTO  = DATA_PATH / "grifols_cytokine_data.csv"
OUTPUT_FILE = DATA_PATH / "analysis_merged_subject_level.csv"

print("Repo root  :", REPO_PATH)
print("Meta file  :", INPUT_META)
print("Cyto file  :", INPUT_CYTO)
print("Meta exists:", INPUT_META.exists())
print("Cyto exists:", INPUT_CYTO.exists())

# ---- 0) Read original files ----
meta_long = pd.read_csv(INPUT_META)
cyto_raw  = pd.read_csv(INPUT_CYTO)

# ---- 1) Subject-level metadata (one row per SUBJECT ID) ----
meta_long = meta_long.rename(columns={"SUBJECT ID": "SUBJECT_ID"})
meta_long["SUBJECT_ID"] = meta_long["SUBJECT_ID"].astype(str).str.strip()

# pick columns that are constant per subject
subject_meta = (
    meta_long
    .groupby("SUBJECT_ID", as_index=False)
    .agg({
        "AGE": "first",
        "AGEGR1C": "first",
        "RACEGRP": "first",
        "SEXC": "first",
        "Batch date": "first"
    })
)
print("Subject_meta shape:", subject_meta.shape)

# ---- 2) Clean cytokine table ----
# drop embedded header row
cyto = cyto_raw.iloc[1:].copy()

cyto = cyto.rename(columns={"Unnamed: 1": "SUBJECT_ID"})
cyto["SUBJECT_ID"] = cyto["SUBJECT_ID"].astype(str).str.strip()

cyto = cyto[[
    "SUBJECT_ID",
    "IFN-gamma", "IL-12p70", "IL-13", "IL-1beta", "IL-2", "IL-4", "IL-5",
    "IL-6", "TNF-alpha", "GM-CSF", "IL-18", "IL-10", "IL-17A", "IL-21",
    "IL-22", "IL-23", "IL-27", "IL-9"
]].copy()

for col in cyto.columns:
    if col != "SUBJECT_ID":
        cyto[col] = pd.to_numeric(cyto[col], errors="coerce")

print("Cyto shape:", cyto.shape)

# ---- 3) Merge one-row-per-subject ----
df = cyto.merge(subject_meta, on="SUBJECT_ID", how="inner")
print("Merged shape:", df.shape)
print(df.head())

cyto_cols = [
    "IFN-gamma", "IL-12p70", "IL-13", "IL-1beta", "IL-2", "IL-4", "IL-5",
    "IL-6", "TNF-alpha", "GM-CSF", "IL-18", "IL-10", "IL-17A", "IL-21",
    "IL-22", "IL-23", "IL-27", "IL-9"
]

X_raw = df[cyto_cols].copy()
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

y_age  = df["AGE"].values
y_sex  = (df["SEXC"] == "Male").astype(int).values
y_race = df["RACEGRP"].values

df.to_csv(OUTPUT_FILE, index=False)
print("Saved to:", OUTPUT_FILE)