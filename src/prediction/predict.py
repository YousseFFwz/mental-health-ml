import joblib
from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "mental_health_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def predict_student(student_data):
    model_data = load_model()

    model = model_data["model"]
    preprocessor = model_data["preprocessor"]
    feature_columns = model_data["feature_columns"]

    student_df = pd.DataFrame([student_data])

    student_df = student_df[feature_columns]

    student_processed = preprocessor.transform(student_df)

    prediction = model.predict(student_processed)[0]

    return prediction