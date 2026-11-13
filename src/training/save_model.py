import joblib
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "mental_health_model.joblib"


def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved successfully: {MODEL_PATH}")