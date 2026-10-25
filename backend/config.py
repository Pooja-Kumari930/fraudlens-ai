"""Central configuration (paths and constants)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PKG_DIR = DATA_DIR / "mendeley"          # the 12-file Mendeley package lives here
MODELS_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"

# files of the package that the application uses
FILES = {"train_time": "train_time_80.csv", "test_time": "test_time_20.csv",
         "train_random": "train_random_80.csv", "test_random": "test_random_20.csv",
         "harmonized": "harmonized_5000_temporally_matched.csv"}

PROCESSED_CSV = DATA_DIR / "transactions.csv"       # every transaction + risk score (written by training)
PIPELINE_PATH = MODELS_DIR / "fraud_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
RANDOM_STATE = 42
CV_FOLDS = 5

# Approximate fraud prevalence in the FULL public source datasets (from their public documentation).
# Used only for the illustrative "what would precision be at the real base rate" projection.
REAL_BASE_RATES = {"PaySim": 8213 / 6362620, "IEEE-CIS": 0.035, "ULB": 492 / 284807}
