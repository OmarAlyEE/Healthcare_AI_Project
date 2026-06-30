import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    BASE_DIR,
    CLEAN_CLAIMS_CSV,
    DATE_COLUMNS,
    ID_COLUMNS,
    PROCESSED_DIR,
    TRAIN_BENEFICIARY_CSV,
    TRAIN_INPATIENT_CSV,
    TRAIN_OUTPATIENT_CSV,
    TRAIN_PROVIDER_CSV,
)

PROCESSED_DIR.mkdir(exist_ok=True)

print("Base directory:")
print(BASE_DIR)

print("\nFiles:")
print(TRAIN_PROVIDER_CSV)
print(TRAIN_BENEFICIARY_CSV)
print(TRAIN_INPATIENT_CSV)
print(TRAIN_OUTPATIENT_CSV)

try:
    provider = pd.read_csv(TRAIN_PROVIDER_CSV)
    beneficiary = pd.read_csv(TRAIN_BENEFICIARY_CSV)
    inpatient = pd.read_csv(TRAIN_INPATIENT_CSV)
    outpatient = pd.read_csv(TRAIN_OUTPATIENT_CSV)

    print("\n[OK] Successfully loaded all files!")
    print("Provider:", provider.shape)
    print("Beneficiary:", beneficiary.shape)
    print("Inpatient:", inpatient.shape)
    print("Outpatient:", outpatient.shape)

except FileNotFoundError as e:
    print("\n[ERROR] File not found:")
    print(e)
    raise SystemExit(1)

print("\nCombining inpatient + outpatient...")

inpatient["claim_type"] = "inpatient"
outpatient["claim_type"] = "outpatient"

claims = pd.concat([inpatient, outpatient], ignore_index=True)
print("Claims shape:", claims.shape)

print("\nMerging beneficiary data...")
claims = claims.merge(beneficiary, on="BeneID", how="left")
print("After beneficiary merge:", claims.shape)

print("\nMerging provider data...")
claims = claims.merge(provider, on="Provider", how="left")

print("\nCreating medical indicators...")
diag_cols = [c for c in claims.columns if "ClmDiagnosisCode" in c]
proc_cols = [c for c in claims.columns if "ClmProcedureCode" in c]

claims["num_diagnoses"] = claims[diag_cols].notna().sum(axis=1)
claims["num_procedures"] = claims[proc_cols].notna().sum(axis=1)
claims["has_procedure"] = (claims["num_procedures"] > 0).astype(int)

print(claims["has_procedure"].value_counts())
print("After provider merge:", claims.shape)

claims["claim_amount"] = claims["InscClaimAmtReimbursed"]
claims["deductible_amount"] = claims["DeductibleAmtPaid"]

print("\nRemoving duplicates...")
duplicate_count = claims.duplicated().sum()
print("Duplicates:", duplicate_count)
claims = claims.drop_duplicates()
print("Shape after duplicates:", claims.shape)

print("\nMissing values before cleaning:")
print(claims.isnull().sum().sort_values(ascending=False).head(20))

print("\nHandling missing values...")
numeric_columns = claims.select_dtypes(include=np.number).columns
categorical_columns = claims.select_dtypes(exclude=np.number).columns

for col in numeric_columns:
    median_value = claims[col].median()
    if pd.isna(median_value):
        claims[col] = claims[col].fillna(0)
    else:
        claims[col] = claims[col].fillna(median_value)

for col in categorical_columns:
    claims[col] = claims[col].fillna("Missing")

print("\nConverting dates...")
for col in DATE_COLUMNS:
    if col in claims.columns:
        claims[col] = pd.to_datetime(claims[col], errors="coerce")

if "ClaimStartDt" in claims.columns:
    claims["claim_year"] = claims["ClaimStartDt"].dt.year
    claims["claim_month"] = claims["ClaimStartDt"].dt.month

for col in DATE_COLUMNS:
    if col in claims.columns:
        claims[col] = claims[col].fillna(pd.NaT)

print("\nChecking invalid costs...")
for col in ["InscClaimAmtReimbursed", "DeductibleAmtPaid"]:
    if col in claims.columns:
        negatives = (claims[col] < 0).sum()
        print(col, "negative values:", negatives)
        claims = claims[claims[col] >= 0]

print("\nCreating age feature...")
if "DOB" in claims.columns and "ClaimStartDt" in claims.columns:
    claims["age"] = claims["ClaimStartDt"].dt.year - claims["DOB"].dt.year
    claims = claims[(claims["age"] >= 0) & (claims["age"] <= 120)]

print("\nFrequency encoding categorical columns...")
physician_cols = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
encode_columns = [
    "Gender",
    "Race",
    "State",
    "County",
    "claim_type",
] + diag_cols + proc_cols + physician_cols

for col in encode_columns:
    if col in claims.columns:
        print("Frequency Encoding:", col)
        freq = claims[col].astype(str).value_counts(normalize=True)
        claims[col] = claims[col].astype(str).map(freq).fillna(0)

for col in ID_COLUMNS:
    if col in claims.columns:
        claims[col] = claims[col].astype(str)

claims = claims.fillna(0)

print("\nPreparing ML dataset...")
remove_columns = ["PotentialFraud"]
X = claims.drop(columns=remove_columns, errors="ignore")

print("\n========================")
print("FINAL CHECK")
print("========================")
print("Final shape:", X.shape)
print("Missing values:", X.isnull().sum().sum())
print("Duplicates:", X.duplicated().sum())

X.to_csv(CLEAN_CLAIMS_CSV, index=False)

print("\n[OK] Saved cleaned dataset:")
print(CLEAN_CLAIMS_CSV)
