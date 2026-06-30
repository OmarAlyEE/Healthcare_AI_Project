import sys
from pathlib import Path

import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (
    BASE_DIR,
    TRAIN_BENEFICIARY_CSV,
    TRAIN_INPATIENT_CSV,
    TRAIN_OUTPATIENT_CSV,
    TRAIN_PROVIDER_CSV,
)

print("Base directory:", BASE_DIR)

print("\nLooking for files:")
print("Provider:", TRAIN_PROVIDER_CSV)
print("Beneficiary:", TRAIN_BENEFICIARY_CSV)
print("Inpatient:", TRAIN_INPATIENT_CSV)
print("Outpatient:", TRAIN_OUTPATIENT_CSV)

try:
    provider = pd.read_csv(TRAIN_PROVIDER_CSV)
    beneficiary = pd.read_csv(TRAIN_BENEFICIARY_CSV)
    inpatient = pd.read_csv(TRAIN_INPATIENT_CSV)
    outpatient = pd.read_csv(TRAIN_OUTPATIENT_CSV)

    print("\n[OK] Successfully loaded all files!")
    print("Provider shape:", provider.shape)
    print("Beneficiary shape:", beneficiary.shape)
    print("Inpatient shape:", inpatient.shape)
    print("Outpatient shape:", outpatient.shape)

except FileNotFoundError:
    print("\nFiles currently in data/raw folder:")
    raw_folder = BASE_DIR / "data" / "raw"
    if raw_folder.exists():
        csv_files = list(raw_folder.glob("*.csv"))
        if csv_files:
            for f in csv_files:
                print("   ", f.name)
        else:
            print("   (No CSV files found)")
    else:
        print("   data/raw folder does not exist!")
