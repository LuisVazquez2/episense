import json
import time
import requests
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# =========================
# Load logo and set page config
# =========================
from PIL import Image

logo = Image.open("logo.png")

st.set_page_config(
    page_title="EpiSense – Early Outbreak Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=logo,
)

# =========================
# Editable configuration
# =========================
CSV_PATH_DEFAULT = "data/processed/episense_annual_with_risk.csv"

# Remote GeoJSON for countries (ISO3 in feature.id). If no internet, try local.
GEOJSON_URL_DEFAULT = (
    "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
)
GEOJSON_LOCAL_FALLBACK = "data/world_countries.geojson"

CENTROIDS_CSV = "data/iso3_centroids_cleaned.csv"  # from prepare_centroids.py

# =========================
# Sidebar (parameters)
# =========================
st.sidebar.title("⚙️ Configuration")
st.sidebar.image(logo, use_container_width=True)

csv_path = st.sidebar.text_input("CSV Path", CSV_PATH_DEFAULT)
api_url = st.sidebar.text_input(
    "API (POST /inference)",
    "URL of deployed API"
)
use_cloud = st.sidebar.toggle("Use Cloud to recalculate risk", value=False)
timeout_s = st.sidebar.slider("API Timeout (sec)", 1, 15, 7)


# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=True)
def load_csv(path):
    df_ = pd.read_csv(path)
    # minimal column check
    needed = {
        "spatial_dim",
        "spatial_dim_en",
        "year",
        "risk_score",
        "cases_per_100k",
        "lag_cases_1",
        "lag_cases_2",
        "ma3_cases",
    }
    missing = needed - set(df_.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    # ensure types
    df_["year"] = pd.to_numeric(df_["year"], errors="coerce").astype(int)
    return df_


@st.cache_data(show_spinner=True)
def load_geojson(remote_url, local_fallback):
    # try remote
    try:
        r = requests.get(remote_url, timeout=5)
        r.raise_for_status()
        gj_ = r.json()
        return gj_, "online"
    except Exception:
        # fallback local
        with open(local_fallback, "r", encoding="utf-8") as f:
            gj_ = json.load(f)
        return gj_, "local"


@st.cache_data(show_spinner=True)
def load_centroids(path):
    try:
        cdf = pd.read_csv(path)
        assert {"iso3", "lat", "lon"}.issubset(set(cdf.columns))
        return cdf
    except Exception:
        return None


try:
    df = load_csv(csv_path)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

gj, gj_src = load_geojson(GEOJSON_URL_DEFAULT, GEOJSON_LOCAL_FALLBACK)
centroids = load_centroids(CENTROIDS_CSV)

# =========================
# UI principal
# =========================
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, width=130, output_format="PNG")
with col2:
    st.title("\nEpiSense – Early Outbreak Detection")

st.caption(
    "Country risk map (ISO3), year slider, and risk recalculation via Cloud API"
)

# Selected year + alert threshold
years = sorted(df["year"].dropna().unique().tolist())
c1, c2, c3 = st.columns([1.2, 1, 1.2])
with c1:
    year = st.slider(
        "Year",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=int(max(years)),
    )
with c2:
    ALERT_THRESHOLD = st.slider("Alert threshold (risk_score)", 0, 100, 50, 1)
with c3:
    st.write("")  # spacer
    st.metric("Geographic region", "LATAM / Global")
    st.caption(f"GeoJSON: {gj_src}")

dfy = df[df["year"] == year].copy()

# =========================
# Recalculate risk via Cloud API
# =========================
with st.expander("🔄 Recalculate risk ", expanded=False):
    st.caption(
        "Only the 4 features per country will be sent; 'risk_score' will be updated in the table/choropleth."
    )
    if st.button("Recalculate from Cloud now"):
        payload = (
            dfy[["cases_per_100k", "lag_cases_1", "lag_cases_2", "ma3_cases"]]
            .fillna(0)
            .to_dict("records")
        )
        t0 = time.time()
        try:
            resp = requests.post(api_url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            # support body string
            if isinstance(data, str):
                data = json.loads(data)
            if "risk_score" not in data:
                st.error(f"Response missing 'risk_score': {data}")
            else:
                print(f"Response from API: {data}")  # Debugging line
                dfy["risk_score"] = pd.Series(data["risk_score"])
                st.success(
                    f"Risk updated from Cloud API ✅  (t={int((time.time()-t0)*1000)} ms)"
                )
        except Exception as e:
            st.error(f"Error calling API: {e}")

# =========================
# Alert panel + recommendations
# =========================
alert_df = dfy[dfy["risk_score"] >= ALERT_THRESHOLD - 1e-5][
    [
        "spatial_dim",
        "spatial_dim_en",
        "risk_score",
        "cases_per_100k",
        "lag_cases_1",
        "lag_cases_2",
        "ma3_cases",
    ]
].sort_values("risk_score", ascending=False)

left, right = st.columns([2.3, 1.7])
with left:
    st.subheader("🌡️ Countries on Alert (Threshold)")
    st.dataframe(alert_df.reset_index(drop=True), use_container_width=True, height=300)

with right:
    st.subheader("⚠️ Recommended Actions")
    if alert_df.empty:
        st.success("No countries on alert under the current threshold ✅")
    else:
        top = alert_df.iloc[0]
        st.error(f"Alert in **{top['spatial_dim_en']}** (risk {top['risk_score']:.1f})")
        st.markdown(
            "- **Intensive surveillance** in health centers (48–72h)\n"
            "- **Anti-vector campaign**: targeted fumigation and removal of breeding sites\n"
            "- **Public communication**: SMS/app with recommendations\n"
            "- **Reinforce supplies**: rapid tests and serums in nearby hospitals"
        )

# =========================
# Map Folium (choropleth)
# =========================
st.subheader("🗺️ Risk by Country (ISO3 Choropleth)")

# Bins with contrast relative to the threshold
bins = [
    0,
    ALERT_THRESHOLD * 0.5,
    ALERT_THRESHOLD * 0.8,
    ALERT_THRESHOLD,
    min(100, ALERT_THRESHOLD + 10),
    min(100, ALERT_THRESHOLD + 20),
    min(100, ALERT_THRESHOLD + 30),
    100,
]

# Build dict for tooltips (ISO3 -> html)
info_by_iso = {}
for _, row in dfy.iterrows():
    iso3 = row["spatial_dim"]
    info_by_iso[iso3] = (
        f"<b>{row['spatial_dim_en']}</b><br>"
        f"Risk: {row['risk_score']:.1f}<br>"
        f"Cases/100k: {row['cases_per_100k']:.1f}<br>"
        f"Lag1: {row['lag_cases_1']:.1f} | Lag2: {row['lag_cases_2']:.1f}<br>"
        f"MA3: {row['ma3_cases']:.1f}"
    )

# Create base map
m = folium.Map(location=[15, -60], zoom_start=3, tiles="cartodbpositron")

# Choropleth
choropleth = folium.Choropleth(
    geo_data=gj,
    data=dfy,
    columns=["spatial_dim", "risk_score"],  # spatial_dim = ISO3
    key_on="feature.id",  # the ISO3 is in feature.id
    fill_color="YlOrRd",
    fill_opacity=0.75,
    line_opacity=0.2,
    legend_name=f"Risk Score (umbral {ALERT_THRESHOLD})",
    bins=bins,
    nan_fill_opacity=0.2,
).add_to(m)

# "Ghost" layer for smoother hover
style_function = lambda x: {"fillColor": "#00000000", "color": "#00000000", "weight": 0}
highlight_function = lambda x: {"fillOpacity": 0.9}

folium.GeoJson(
    gj,
    style_function=style_function,
    highlight_function=highlight_function,
    name="hover",
).add_to(m)

# Tooltips for each country
for feature in gj["features"]:
    iso = feature.get("id")
    if iso in info_by_iso:
        folium.GeoJson(
            feature,
            style_function=lambda x: {
                "fillOpacity": 0,
                "color": "#00000000",
                "weight": 0,
            },
            tooltip=folium.Tooltip(info_by_iso[iso], sticky=True),
        ).add_to(m)

# markers if centroids exist
if centroids is not None and not centroids.empty:
    cdict = centroids.set_index("iso3")[["lat", "lon"]].to_dict(orient="index")
    for _, row in dfy.iterrows():
        if row["risk_score"] >= ALERT_THRESHOLD:
            iso3 = row["spatial_dim"]
            if iso3 in cdict:
                lat, lon = cdict[iso3]["lat"], cdict[iso3]["lon"]
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color="#8B0000",
                    weight=2,
                    fill=True,
                    fill_opacity=0.9,
                    popup=folium.Popup(info_by_iso.get(iso3, ""), max_width=300),
                ).add_to(m)

st_folium(m, use_container_width=True, height=600)

# =========================
# Trend per country|
# =========================
st.subheader("📈 Risk Trend by Country")
sel_iso = st.selectbox("Country (ISO3)", sorted(df["spatial_dim"].unique().tolist()))
dfc = df[df["spatial_dim"] == sel_iso].sort_values("year")
if dfc.empty:
    st.warning("No data for the selected country.")
    st.stop()

# Plot with Streamlit line_chart
try:
    chart_data = dfc.set_index("year")["risk_score"]
    chart_data.index = chart_data.index.astype(str)  # year as str
    st.caption(f"Country: {dfc['spatial_dim_en'].iloc[0]} ({sel_iso})")
    st.line_chart(chart_data, use_container_width=True, height=400, color="#0E95DD")
    st.markdown(
        "- The risk score is calculated based on recent case trends.\n"
        "- Sudden increases may indicate potential outbreaks.\n"
        "- Use this trend to inform public health decisions."
    )
except Exception as e:
    st.info("Error plotting the series.")
    st.error(str(e))
    st.warning("Try a different country or check the data.")


# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    "EpiSense • AI for early outbreak detection • Developed by Luis Vazquez - 2025"
)
