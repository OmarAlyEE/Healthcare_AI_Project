from pathlib import Path

# Project root (parent of src/)
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# Raw training files
TRAIN_PROVIDER_CSV = RAW_DIR / "Train.csv"
TRAIN_BENEFICIARY_CSV = RAW_DIR / "Train_Beneficiarydata.csv"
TRAIN_INPATIENT_CSV = RAW_DIR / "Train_Inpatientdata.csv"
TRAIN_OUTPATIENT_CSV = RAW_DIR / "Train_Outpatientdata.csv"

# Processed outputs
CLEAN_CLAIMS_CSV = PROCESSED_DIR / "clean_claims.csv"
ENGINEERED_CLAIMS_CSV = PROCESSED_DIR / "engineered_claims.csv"
SCORED_CLAIMS_CSV = PROCESSED_DIR / "scored_claims.csv"
EXPLAINED_CLAIMS_CSV = PROCESSED_DIR / "explained_claims.csv"

# Visualization outputs
OUTPUT_DIR = BASE_DIR / "outputs"

# Model artifacts
ISOLATION_FOREST_PATH = MODEL_DIR / "isolation_forest.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

FEATURE_COLUMNS = [
    "claim_duration_days",
    "age",
    "claim_amount",
    "claim_log_amount",
    "annual_claim_amount",
    "annual_claim_amount_log",
    "annual_claim_frequency_proxy",
    "provider_claims_30d",
    "provider_claims_90d",
    "provider_claim_count",
    "provider_claim_count_log",
    "provider_avg_claim",
    "provider_total_claim",
    "claim_vs_provider_avg",
    "claim_vs_global_avg",
    "claim_difference_from_provider_avg",
    "high_cost_flag",
    "provider_claims_per_month",
    "provider_claims_per_month_log",
    "high_frequency_flag",
    "patient_claim_count",
    "patient_claim_count_log",
    "patient_avg_claim",
]

IDENTIFIER_COLUMNS = ["Provider", "ClaimID", "BeneID", "is_train"]

TIME_COLUMNS = ["ClaimStartDt", "claim_year", "claim_month"]

N_CLUSTERS = 4
KMEANS_SAMPLE_SIZE = 10_000

CLUSTERED_CLAIMS_CSV = PROCESSED_DIR / "clustered_claims.csv"


ID_COLUMNS = ["Provider", "BeneID", "ClaimID"]

DATE_COLUMNS = [
    "ClaimStartDt",
    "ClaimEndDt",
    "AdmissionDt",
    "DischargeDt",
    "DOB",
    "DOD",
]
