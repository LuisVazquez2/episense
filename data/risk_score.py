import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the processed data
df = pd.read_csv("data/processed/episense_annual.csv")

# Features to use for risk scoring
FEATURES = ["cases_per_100k", "lag_cases_1", "lag_cases_2", "ma3_cases"]
df[FEATURES] = df[FEATURES].fillna(0)

# Train IsolationForest model on all data (can be trained per country if needed)
model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
model.fit(df[FEATURES])

# Compute anomaly scores (higher = more anomalous)
raw_scores = -model.score_samples(df[FEATURES])
df["risk_score"] = 100 * (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

# Save the results
df.to_csv("data/processed/episense_annual_with_risk.csv", index=False)
print("OK -> data/processed/episense_annual_with_risk.csv")
