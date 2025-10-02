# EpiSense – Early Outbreak Detection in LATAM

**AI-powered early detection of epidemic outbreaks in LATAM**

---

## Problem

Latin America faces recurring outbreaks of **dengue, zika, and chikungunya**.  
Detection is often **delayed**, resulting in overwhelmed hospitals and increased public health costs.

---

## Solution

**EpiSense** enables **early epidemic risk detection** by:

- Training an **IsolationForest model** on open health and climate datasets
- Extracting temporal features: `cases_per_100k`, `lag1`, `lag2`, `MA3`
- Providing real-time **risk scores per country and year**
- Delivering an interactive **map dashboard** with thresholds, alerts, and recommendations

This approach leverages **data science and cloud-native architecture** to reduce time-to-alert for health agencies.

---

### What Was Built

- Data pipeline: preprocessing dengue + population data, feature engineering
- ML model: IsolationForest anomaly detection for outbreak signals
- Visualization: Streamlit interactive map + alert panel
- Integration: Real-time cloud call → risk recalculated on demand

---
## Future Plans
- Expand to more diseases (zika, chikungunya)
- Incorporate additional data sources (mobility, social media)
- Deploy as a fully managed cloud service for health agencies
- Add pandemic simulation and scenario analysis like COVID-19

---

## Tech Stack
- Python, Pandas, Scikit-learn
- Streamlit for dashboard
- Cloud: Any cloud provider for hosting (e.g., AWS, GCP, Azure) using serverless functions and API Gateway
- Open datasets: WHO, World Bank, NOAA

---

## Getting Started

1. **Move to the project directory:**
   ```bash
   cd episense
   ```
2. **Set up the environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```
4. **Access the dashboard:**

   Open your browser and navigate to `http://localhost:8501`

5.  Explore the interactive map, adjust thresholds, and view recommendations. 