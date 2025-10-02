import pandas as pd
import numpy as np
from pathlib import Path

IN_FILE = "data/iso3_centroids.csv"
OUT_FILE = "data/iso3_centroids_cleaned.csv"

# 1) Read CSV
df = pd.read_csv(IN_FILE)

# 2) convert to ISO codes to ISO3 where possible


def iso_to_iso3(iso):
    """Convert ISO2 to ISO3. If already ISO3 or unknown, return as is."""
    iso = str(iso).strip().upper()
    if len(iso) == 3:
        return iso  # Assume already ISO3
    elif len(iso) == 2:
        try:
            from iso3166 import countries

            return countries.get(iso).alpha3
        except KeyError:
            return iso  # Unknown code, return as is
    else:
        return iso  # Invalid code, return as is


df["iso3"] = df["ISO"].apply(iso_to_iso3)

# 3) expose only needed columns
df_clean = (
    df[["iso3", "lat", "lon"]]
    .dropna(subset=["iso3", "lat", "lon"])
    .drop_duplicates(subset=["iso3"])
)
df_clean = df_clean[
    (df_clean["lat"].between(-90, 90)) & (df_clean["lon"].between(-180, 180))
]
df_clean = df_clean.sort_values("iso3").reset_index(drop=True)
df_clean.to_csv(OUT_FILE, index=False)
print(f"Saved cleaned centroids to {OUT_FILE}, {len(df_clean)} entries.")
print(df_clean.head(10))
print(df_clean.tail(10))
