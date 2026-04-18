from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = DATA_DIR / "eda_subject_level"

# Load raw cytokine data (before log transform / imputation)
cyto_raw = pd.read_csv(DATA_DIR / "grifols_cytokine_data.csv")
cyto = cyto_raw.iloc[1:].copy()
cyto = cyto.rename(columns={"Unnamed: 1": "SUBJECT_ID"})

CYTO_COLS = [
    "IFN-gamma", "IL-12p70", "IL-13", "IL-1beta", "IL-2", "IL-4", "IL-5",
    "IL-6", "TNF-alpha", "GM-CSF", "IL-18", "IL-10", "IL-17A", "IL-21",
    "IL-22", "IL-23", "IL-27", "IL-9"
]
for col in CYTO_COLS:
    cyto[col] = pd.to_numeric(cyto[col], errors="coerce")

DETECTION_LIMIT = 5.0  # adjust to your assay's actual detection limit

# % of donors at or below detection limit per cytokine
n_total = len(cyto)
pct_below = {}
for col in CYTO_COLS:
    pct_below[col] = (cyto[col] <= DETECTION_LIMIT).sum() / n_total * 100

pct_df = pd.Series(pct_below).sort_values(ascending=False)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#4A7FC1" if v < 80 else "#E87D7D" for v in pct_df.values]
bars = ax.bar(pct_df.index, pct_df.values, color=colors, edgecolor="white", alpha=0.85)

ax.axhline(80, color="darkred", linestyle="--", linewidth=1.5,
           label="80% threshold")
ax.set_ylabel("Donors at or below detection limit (%)", fontsize=11)
ax.set_title("Cytokine Detectability Across Donors", fontsize=13, fontweight="bold")
ax.set_ylim(0, 105)
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.legend(fontsize=9)

for bar, val in zip(bars, pct_df.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
            f"{val:.0f}%", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cytokine_detection_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cytokine_detection_rate.png")