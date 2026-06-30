import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    ENGINEERED_CLAIMS_CSV,
    FEATURE_COLUMNS,
    ISOLATION_FOREST_PATH,
    SCALER_PATH,
    SCORED_CLAIMS_CSV,
)

print("Loading model and scaler...")
model = joblib.load(ISOLATION_FOREST_PATH)
scaler = joblib.load(SCALER_PATH)

print("Loading dataset:")
df = pd.read_csv(ENGINEERED_CLAIMS_CSV)
print("Shape:", df.shape)

print("Formatting features...")
X_df = df[FEATURE_COLUMNS].copy()

non_numeric_cols = X_df.select_dtypes(exclude=["int64", "float64", "int32", "float32"]).columns
if len(non_numeric_cols) > 0:
    X_df = X_df.drop(columns=non_numeric_cols)

X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
X_scaled = scaler.transform(X_df)

print("Scoring with Isolation Forest...")
pred = model.predict(X_scaled)

score = model.decision_function(X_scaled)
risk_score = 100 * (1 - (score - score.min()) / (score.max() - score.min() + 1e-8))
risk_score = np.clip(risk_score, 0, 100)

df["raw_risk_score"] = score
df["risk_score"] = risk_score
df["risk_percentile"] = df["risk_score"].rank(pct=True)

df["risk_category"] = "low risk"
df.loc[df["risk_percentile"] >= 0.95, "risk_category"] = "medium risk"
df.loc[df["risk_percentile"] >= 0.99, "risk_category"] = "high risk"
df["anomaly_flag"] = (pred == -1).astype(int)

df.to_csv(SCORED_CLAIMS_CSV, index=False)

print("\nSaved scored dataset:")
print(SCORED_CLAIMS_CSV)

print("\nSummary:")
print(df["risk_category"].value_counts())
