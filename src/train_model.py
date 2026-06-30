import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    ENGINEERED_CLAIMS_CSV,
    FEATURE_COLUMNS,
    ISOLATION_FOREST_PATH,
    MODEL_DIR,
    SCALER_PATH,
    SCORED_CLAIMS_CSV,
)

MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset:")
print(ENGINEERED_CLAIMS_CSV)

df = pd.read_csv(ENGINEERED_CLAIMS_CSV)
print("\nDataset loaded:", df.shape)

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

train_mask = df["is_train"] == 1
test_mask = df["is_train"] == 0

X = df[FEATURE_COLUMNS].copy()

print("\nValidating feature types...")
non_numeric_cols = X.select_dtypes(exclude=["int64", "float64", "int32", "float32"]).columns
if len(non_numeric_cols) > 0:
    print(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
    X = X.drop(columns=non_numeric_cols)

X_train = X[train_mask].copy()
X_test = X[test_mask].copy()

print("\nApplying Standard Scaling...")
scaler = StandardScaler()
scaler.fit(X_train)

X_scaled = scaler.transform(X)

contaminations = [0.01, 0.02, 0.03, 0.05]
seeds = [0, 42, 99]

print("\nRunning Isolation Forest Stability Experiments...\n")

best_model = None
best_preds = None

for c in contaminations:
    print(f"--- Contamination: {c} ---")

    seed_anomalies = []

    for s in seeds:
        model = IsolationForest(
            contamination=c,
            random_state=s,
            n_estimators=100,
            n_jobs=-1,
        )

        model.fit(scaler.transform(X_train))
        preds = model.predict(X_scaled)

        anomaly_indices = set(np.where(preds == -1)[0])
        seed_anomalies.append(anomaly_indices)

        if c == 0.02 and s == 42:
            best_model = model
            best_preds = preds

    intersection = seed_anomalies[0].intersection(seed_anomalies[1]).intersection(seed_anomalies[2])
    union = seed_anomalies[0].union(seed_anomalies[1]).union(seed_anomalies[2])

    stability_score = len(intersection) / len(union) if len(union) > 0 else 0
    print(f"Stability Score (Intersection over Union): {stability_score:.4f}")
    print(f"Average Anomalies Found: {np.mean([len(sa) for sa in seed_anomalies]):.0f}\n")

print("=====================================================")
print("FINAL EVALUATION (Contamination=0.02, Seed=42)")
print("=====================================================")

score = best_model.decision_function(X_scaled)
risk_score = 100 * (1 - (score - score.min()) / (score.max() - score.min() + 1e-8))
risk_score = np.clip(risk_score, 0, 100)

df["raw_risk_score"] = score
df["risk_score"] = risk_score
df["risk_percentile"] = df["risk_score"].rank(pct=True)

df["risk_category"] = "low risk"
df.loc[df["risk_percentile"] >= 0.95, "risk_category"] = "medium risk"
df.loc[df["risk_percentile"] >= 0.99, "risk_category"] = "high risk"
df["anomaly_flag"] = (best_preds == -1).astype(int)

print("\nProvider Risk Concentration (Top 10):")
provider_risk = df.groupby("Provider").agg(
    claim_count=("raw_risk_score", "count"),
    anomaly_rate=("anomaly_flag", "mean"),
)
risky_providers = provider_risk[provider_risk["claim_count"] >= 5].sort_values(
    "anomaly_rate", ascending=False
)
print(risky_providers.head(10))

print("\nRisk Categories:")
print(df["risk_category"].value_counts())

joblib.dump(best_model, ISOLATION_FOREST_PATH)
joblib.dump(scaler, SCALER_PATH)

df.to_csv(SCORED_CLAIMS_CSV, index=False)

print("\nSaved model, scaler, and scored dataset")
print("Output:", SCORED_CLAIMS_CSV)
