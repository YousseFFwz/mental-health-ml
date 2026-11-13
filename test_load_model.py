import joblib
from pathlib import Path

MODEL_PATH = Path("models/mental_health_model.joblib")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print("Model:", model)