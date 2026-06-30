import sys
from pathlib import Path

import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import EXPLAINED_CLAIMS_CSV, SCORED_CLAIMS_CSV

print("Loading scored dataset...")
df = pd.read_csv(SCORED_CLAIMS_CSV)
print("Shape:", df.shape)


def explain(row):
    reasons = []

    if row.get("claim_vs_provider_avg", 0) > 2:
        reasons.append(
            f"Cost Anomaly: Claim is {row.get('claim_vs_provider_avg', 0):.1f}x higher than provider average"
        )

    if row.get("high_cost_flag", 0) == 1:
        reasons.append("Cost Anomaly: Flagged as a statistically high-cost claim")

    if row.get("claim_difference_from_provider_avg", 0) > 10000:
        reasons.append(
            f"Cost Anomaly: Claim is ${row.get('claim_difference_from_provider_avg', 0):,.2f} above provider average"
        )

    if row.get("provider_claims_per_month", 0) > 50:
        reasons.append(
            f"Frequency Anomaly: Provider has an unusually high claim frequency "
            f"({row.get('provider_claims_per_month', 0):.0f} per month)"
        )

    if row.get("high_frequency_flag", 0) == 1:
        reasons.append("Frequency Anomaly: Provider flagged for top 5% frequency behavior")

    if row.get("provider_claim_count", 0) > 100 and row.get("provider_avg_claim", 0) > 5000:
        reasons.append("Provider Anomaly: High volume and high average cost provider")

    if row.get("risk_score", 0) >= 80:
        reasons.append(
            f"Model Anomaly: High Isolation Forest risk score ({row.get('risk_score', 0):.1f}/100)"
        )
    elif row.get("anomaly_flag", 0) == 1:
        reasons.append("Model Anomaly: Claim flagged as outlier by Isolation Forest")

    if len(reasons) == 0:
        reasons.append("No strong anomaly indicators detected")

    return "; ".join(reasons)


print("Generating explanations...")
df["explanation"] = df.apply(explain, axis=1)

df.to_csv(EXPLAINED_CLAIMS_CSV, index=False)

print("\nSaved explained dataset:")
print(EXPLAINED_CLAIMS_CSV)
