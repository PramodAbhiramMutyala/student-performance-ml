"""
Configuration and constants for Student Performance Prediction & Academic Risk Analysis.
"""
from pathlib import Path
from typing import Dict, List, Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "student-mat.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "student_processed.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EDA_DIR = ARTIFACTS_DIR / "eda"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)

REGRESSOR_MODEL_PATH = MODELS_DIR / "best_regressor.joblib"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "risk_classifier.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Targets and feature exclusion
TARGET_COLUMN = "G3"
PERIOD_GRADE_COLUMNS = ["G1", "G2", "G3"]
EXCLUDED_FEATURES_EARLY_WARNING = ["G1", "G2", "G3"]

# Numerical and categorical feature definitions (30 early-warning inputs)
NUMERICAL_FEATURES: List[str] = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences"
]

CATEGORICAL_FEATURES: List[str] = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic"
]

ALL_FEATURE_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Allowed domains for categorical features
CATEGORICAL_VALUE_DOMAINS: Dict[str, List[str]] = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
    "romantic": ["yes", "no"]
}

# Allowed ranges for numerical features (min, max)
NUMERICAL_VALUE_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (15.0, 22.0),
    "Medu": (0.0, 4.0),
    "Fedu": (0.0, 4.0),
    "traveltime": (1.0, 4.0),
    "studytime": (1.0, 4.0),
    "failures": (0.0, 4.0),
    "famrel": (1.0, 5.0),
    "freetime": (1.0, 5.0),
    "goout": (1.0, 5.0),
    "Dalc": (1.0, 5.0),
    "Walc": (1.0, 5.0),
    "health": (1.0, 5.0),
    "absences": (0.0, 93.0)
}

# Academic Risk Categorization Settings (based on Portuguese 0-20 grading scale)
HIGH_RISK_THRESHOLD = 10.0   # Final score < 10: High Risk (Failing mark)
MEDIUM_RISK_THRESHOLD = 14.0 # Final score 10.0 to 13.99: Moderate Risk (Satisfactory mark)
# Final score >= 14.0: Low Risk (Good to Excellent mark)

RISK_LABELS = {
    "high": "High Academic Risk",
    "medium": "Moderate Academic Risk",
    "low": "Low Academic Risk"
}

RISK_RECOMMENDATIONS = {
    "High Academic Risk": [
        "Immediate remedial math tutoring and personalized intervention.",
        "Weekly parent-teacher academic review conferences.",
        "Strict absence monitoring and structured after-school study halls."
    ],
    "Moderate Academic Risk": [
        "Targeted study schedule with peer tutoring in difficult modules.",
        "Encourage reduction of excessive weekday social and screen time.",
        "Bi-weekly progress checks on homework and quiz scores."
    ],
    "Low Academic Risk": [
        "Advanced enrichment projects and STEM competitive preparation.",
        "Mentorship opportunities to guide peer study circles.",
        "Maintain current productive study balance and positive routines."
    ]
}


# Application Settings
class Settings(BaseSettings):
    APP_NAME: str = "Student Performance Prediction & Academic Risk Analysis"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'predictions.db'}"
    MAX_UPLOAD_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB
    MAX_BATCH_ROWS: int = 5000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
