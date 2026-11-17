"""Central configuration for the Customer Churn Prediction System.

Every path, column name, hyper-parameter search space and threshold used by the
project lives here so that nothing is hard-coded in the other modules.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
DATASET_MISSING_MESSAGE = (
    "Dataset not found. Please download the IBM Telco Customer Churn CSV "
    "and place it inside the data/ directory."
)

BEST_MODEL_FILENAME = "best_model.joblib"
METADATA_FILENAME = "model_metadata.json"
EVALUATION_FILENAME = "evaluation_results.json"
COMPARISON_FILENAME = "model_comparison.csv"

BEST_MODEL_PATH = MODELS_DIR / BEST_MODEL_FILENAME
METADATA_PATH = MODELS_DIR / METADATA_FILENAME
EVALUATION_PATH = REPORTS_DIR / EVALUATION_FILENAME
COMPARISON_PATH = REPORTS_DIR / COMPARISON_FILENAME

# Bump this number whenever the feature set or saved-artifact layout changes.
# A saved model with a different schema version is treated as incompatible and
# is retrained automatically.
SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- #
# Dataset schema (IBM Telco Customer Churn)
# --------------------------------------------------------------------------- #
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # 19 model inputs

# Values offered in the single-customer prediction form.
_YES_NO = ["No", "Yes"]
_YES_NO_NOINTERNET = ["No", "Yes", "No internet service"]
FEATURE_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": _YES_NO_NOINTERNET,
    "OnlineBackup": _YES_NO_NOINTERNET,
    "DeviceProtection": _YES_NO_NOINTERNET,
    "TechSupport": _YES_NO_NOINTERNET,
    "StreamingTV": _YES_NO_NOINTERNET,
    "StreamingMovies": _YES_NO_NOINTERNET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
INTERNET_ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

MIN_ROWS_FOR_TRAINING = 200  # refuse to train on tiny / broken files

# --------------------------------------------------------------------------- #
# Training settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITERATIONS = 12          # random-search candidates per model
DECISION_THRESHOLD = 0.50       # probability >= threshold -> predicted "Yes"

# The final model is chosen on cross-validated *training* scores (never on the
# test set): score = sum(weight * mean CV metric).  Accuracy is deliberately
# excluded because the classes are imbalanced (~27 % churn).
SELECTION_WEIGHTS = {"roc_auc": 0.5, "f1": 0.5}

# --------------------------------------------------------------------------- #
# Risk bands (model-based, configurable in the app sidebar)
# --------------------------------------------------------------------------- #
RISK_LOW_MAX = 0.30    # probability <  0.30           -> Low
RISK_HIGH_MIN = 0.60   # probability >= 0.60           -> High; in between -> Medium
