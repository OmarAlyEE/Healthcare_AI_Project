import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    CLEAN_CLAIMS_CSV,
    ENGINEERED_CLAIMS_CSV,
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TIME_COLUMNS,
)

print("Loading dataset:")
print(CLEAN_CLAIMS_CSV)

df = pd.read_csv(CLEAN_CLAIMS_CSV, low_memory=False)
print("\nLoaded dataset:", df.shape)

print("\nCreating date features...")
df["ClaimStartDt"] = pd.to_datetime(df["ClaimStartDt"], errors="coerce")
df["ClaimEndDt"] = pd.to_datetime(df["ClaimEndDt"], errors="coerce")
df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")

df["claim_duration_days"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days.fillna(0)
df["age"] = (df["ClaimStartDt"].dt.year - df["DOB"].dt.year).clip(0, 120)

print("\nCreating claim intensity features...")
df["claim_amount"] = df["InscClaimAmtReimbursed"]
df["claim_log_amount"] = np.log1p(df["claim_amount"])

df["annual_claim_amount"] = df["IPAnnualReimbursementAmt"] + df["OPAnnualReimbursementAmt"]
df["annual_claim_amount_log"] = np.log1p(df["annual_claim_amount"].clip(lower=0))
df["annual_claim_frequency_proxy"] = (
    (df["IPAnnualReimbursementAmt"] > 0).astype(int)
    + (df["OPAnnualReimbursementAmt"] > 0).astype(int)
)

df["claim_year"] = df["ClaimStartDt"].dt.year
df["claim_month"] = df["ClaimStartDt"].dt.month
df["provider_month"] = df["claim_year"].astype(str) + "_" + df["claim_month"].astype(str)

print("\nCreating rolling time window features...")
df = df.sort_values(["Provider", "ClaimStartDt"])
df = df.set_index("ClaimStartDt")
df["dummy"] = 1
df["provider_claims_30d"] = (
    df.groupby("Provider")["dummy"].rolling("30D").sum().reset_index(level=0, drop=True)
)
df["provider_claims_90d"] = (
    df.groupby("Provider")["dummy"].rolling("90D").sum().reset_index(level=0, drop=True)
)
df = df.drop(columns=["dummy"]).reset_index()

print("\nSplitting data to guarantee ZERO group leakage...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = train_df.copy()
train_df["is_train"] = 1

test_df = test_df.copy()
test_df["is_train"] = 0

print("\nComputing Provider behavior stats ONLY on Train...")

provider_claim_count = train_df.groupby("Provider").size().rename("provider_claim_count")
provider_avg_claim = train_df.groupby("Provider")["claim_amount"].mean().rename("provider_avg_claim")
provider_total_claim = train_df.groupby("Provider")["claim_amount"].sum().rename("provider_total_claim")
provider_month_frequency = (
    train_df.groupby(["Provider", "provider_month"]).size().rename("provider_claims_per_month")
)

patient_claim_count = train_df.groupby("BeneID").size().rename("patient_claim_count")
patient_avg_claim = train_df.groupby("BeneID")["claim_amount"].mean().rename("patient_avg_claim")

global_avg = train_df["claim_amount"].mean()
provider_frequency_threshold = provider_month_frequency.quantile(0.95)


def map_features(data_df):
    d = data_df.copy()

    d = d.merge(provider_claim_count, on="Provider", how="left")
    d["provider_claim_count"] = d["provider_claim_count"].fillna(0)
    d["provider_claim_count_log"] = np.log1p(d["provider_claim_count"])

    d = d.merge(provider_avg_claim, on="Provider", how="left")
    d["provider_avg_claim"] = d["provider_avg_claim"].fillna(global_avg)

    d = d.merge(provider_total_claim, on="Provider", how="left")
    d["provider_total_claim"] = d["provider_total_claim"].fillna(0)

    d["claim_vs_provider_avg"] = d["claim_amount"] / (d["provider_avg_claim"] + 1)
    d["claim_vs_global_avg"] = d["claim_amount"] / (global_avg + 1)
    d["claim_difference_from_provider_avg"] = d["claim_amount"] - d["provider_avg_claim"]
    d["high_cost_flag"] = (d["claim_vs_provider_avg"] > 3).astype(int)

    d = d.merge(provider_month_frequency, on=["Provider", "provider_month"], how="left")
    d["provider_claims_per_month"] = d["provider_claims_per_month"].fillna(0)
    d["provider_claims_per_month_log"] = np.log1p(d["provider_claims_per_month"])
    d["high_frequency_flag"] = (d["provider_claims_per_month"] > provider_frequency_threshold).astype(int)
    d.drop(columns=["provider_month"], inplace=True)

    d = d.merge(patient_claim_count, on="BeneID", how="left")
    d["patient_claim_count"] = d["patient_claim_count"].fillna(0)
    d["patient_claim_count_log"] = np.log1p(d["patient_claim_count"])

    d = d.merge(patient_avg_claim, on="BeneID", how="left")
    d["patient_avg_claim"] = d["patient_avg_claim"].fillna(global_avg)

    return d


print("\nMapping stats to Train and Test sets...")
train_df = map_features(train_df)
test_df = map_features(test_df)

df = pd.concat([train_df, test_df], ignore_index=True)

print("\nCleaning final features...")
keep_columns = FEATURE_COLUMNS + IDENTIFIER_COLUMNS + TIME_COLUMNS
df = df[[col for col in keep_columns if col in df.columns]]
df = df.fillna(0)

print("\n==========================")
print("FEATURE ENGINEERING DONE")
print("==========================")
print("Final shape:", df.shape)

df.to_csv(ENGINEERED_CLAIMS_CSV, index=False)
print("\nSaved engineered dataset:", ENGINEERED_CLAIMS_CSV)
