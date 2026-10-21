"""Central configuration: paths, column groups, validation rules, hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "heart.csv"
DATA_SOURCE_FILE = ROOT / "data" / "raw" / "DATA_SOURCE.txt"
DATA_CLEAN = ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
MLRUNS_DIR = ROOT / "mlruns"

EXPERIMENT_NAME = "heart-disease-risk"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 3
DECISION_THRESHOLD = 0.5

TARGET = "target"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
ENGINEERED_NUMERIC = ["hr_reserve_ratio", "chol_per_age"]
ENGINEERED_CATEGORICAL = ["high_bp"]
NULLABLE_FEATURES = ["ca", "thal"]

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak exercise ST segment (1 up, 2 flat, 3 down)",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thalassemia (3 normal, 6 fixed defect, 7 reversible defect)",
}
NUMERIC_RANGES = {"age": (1, 120), "trestbps": (50, 260), "chol": (50, 700),
                  "thalach": (40, 250), "oldpeak": (-3.0, 10.0), "ca": (0, 4)}
CATEGORICAL_VALUES = {"sex": {0, 1}, "cp": {1, 2, 3, 4}, "fbs": {0, 1}, "restecg": {0, 1, 2},
                      "exang": {0, 1}, "slope": {1, 2, 3}, "thal": {3, 6, 7}}
LABELS = {0: "no heart disease predicted", 1: "heart disease predicted"}

UCI_ID = 45
UCI_CLEVELAND_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
                     "processed.cleveland.data")
UCI_COLUMNS = FEATURES + [TARGET]
