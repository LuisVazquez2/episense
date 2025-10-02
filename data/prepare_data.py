import pandas as pd
import numpy as np
from pathlib import Path

IN_FILE = "paho_all_indicators.csv"
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Read with utf-8-sig in case there's a BOM at the start
df = pd.read_csv(IN_FILE, encoding="utf-8-sig")

# 2) Check required columns
for c in [
    "indicator_name",
    "nombre_indicador",
    "spatial_dim_type",
    "spatial_dim",
    "spatial_dim_en",
    "spatial_dim_es",
    "time_dim_type",
    "time_dim",
    "numeric_value",
]:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# 3) Type coercion and normalization
df["indicator_name_norm"] = df["indicator_name"].astype(str).str.strip().str.lower()
df["nombre_indicador_norm"] = df["nombre_indicador"].astype(str).str.strip().str.lower()
df["spatial_dim_type"] = df["spatial_dim_type"].astype(str).str.strip().str.upper()
df["time_dim_type"] = df["time_dim_type"].astype(str).str.strip().str.upper()

# 4) Basic filter: keep only COUNTRY level
df = df[df["spatial_dim_type"].eq("COUNTRY")].copy()

# 5) Identify dengue rows
is_dengue = df["indicator_name_norm"].str.contains(
    r"\bdengue cases\b", regex=True
) | df["nombre_indicador_norm"].str.contains(
    r"\bdengue\b.*casos|\bcasos\b.*dengue", regex=True
)

dengue = df[is_dengue].copy()

# 6) Identify population rows (varies by dataset; use several heuristics)
is_pop = (
    df["indicator_name_norm"].str.contains("total population", na=False)
    | df["indicator_name_norm"].str.contains("population .*thousand", na=False)
    | df["indicator_name_norm"].str.match(r"^population\b", na=False)
    | df["nombre_indicador_norm"].str.contains("poblaci", na=False)
)

pop = df[is_pop].copy()

# 7) Choose time dimension (prefer YEAR for quick demo)
if "YEAR" in dengue["time_dim_type"].unique():
    dengue_y = dengue[dengue["time_dim_type"].eq("YEAR")].copy()
    pop_y = pop[pop["time_dim_type"].eq("YEAR")].copy()
else:
    # If no YEAR, use MONTH and aggregate to YEAR
    dengue_y = (
        dengue[dengue["time_dim_type"].eq("MONTH")]
        .assign(year=lambda d: d["time_dim"].astype(str).str.slice(0, 4).astype(int))
        .groupby(
            ["spatial_dim", "spatial_dim_en", "spatial_dim_es", "year"], as_index=False
        )["numeric_value"]
        .sum()
    )
    pop_y = (
        pop[pop["time_dim_type"].eq("MONTH")]
        .assign(year=lambda d: d["time_dim"].astype(str).str.slice(0, 4).astype(int))
        .groupby(
            ["spatial_dim", "spatial_dim_en", "spatial_dim_es", "year"], as_index=False
        )["numeric_value"]
        .mean()
    )

# 8) Rename and consolidate
dengue_y = dengue_y.rename(columns={"numeric_value": "dengue_cases"})
pop_y = pop_y.rename(columns={"numeric_value": "population_raw"})

# Detect if population is in thousands by indicator name and adjust
pop_y["is_thousands"] = pop_y["indicator_name_norm"].str.contains(
    "thousand", na=False
) | pop_y["nombre_indicador_norm"].str.contains("mil", na=False)
pop_y["population"] = np.where(
    pop_y["is_thousands"], pop_y["population_raw"] * 1000, pop_y["population_raw"]
)

# 9) Keep key columns and group (in case of duplicates)
key = ["spatial_dim", "spatial_dim_en", "spatial_dim_es", "time_dim"]
if "year" in dengue_y.columns:
    key = ["spatial_dim", "spatial_dim_en", "spatial_dim_es", "year"]

dengue_y = dengue_y.groupby(key, as_index=False)["dengue_cases"].sum()
pop_y = pop_y.groupby(key, as_index=False)["population"].mean()

# 10) Merge cases + population
df_yr = dengue_y.merge(pop_y, on=key, how="left")

# If 'year' not in key, create it from time_dim
if "year" not in df_yr.columns:
    df_yr["year"] = df_yr["time_dim"].astype(str).str.slice(0, 4).astype(int)

# 11) Cases per 100k
df_yr["cases_per_100k"] = (
    df_yr["dengue_cases"] / df_yr["population"].replace(0, np.nan)
) * 1e5

# 12) Sort and generate lags/moving averages by country
df_yr = df_yr.sort_values(["spatial_dim", "year"]).reset_index(drop=True)
df_yr["lag_cases_1"] = df_yr.groupby("spatial_dim")["dengue_cases"].shift(1)
df_yr["lag_cases_2"] = df_yr.groupby("spatial_dim")["dengue_cases"].shift(2)
df_yr["ma3_cases"] = df_yr.groupby("spatial_dim")["dengue_cases"].transform(
    lambda s: s.rolling(3, min_periods=1).mean()
)

# 13) (Optional) If you have annual climate in another CSV: iso3/year -> temp_mean, rain_mm, humidity
# climate = pd.read_csv("climate_by_country_year.csv")
# df_yr = df_yr.merge(climate, left_on=["spatial_dim", "year"], right_on=["iso3", "year"], how="left")

# 14) Save ready for modeling
OUT_FILE = OUT_DIR / "episense_annual.csv"
df_yr.to_csv(OUT_FILE, index=False)
print(f"OK -> {OUT_FILE}  rows={len(df_yr)}")
print(df_yr.head(8))
