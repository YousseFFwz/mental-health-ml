import joblib
from pathlib import Path

from src.data.load_data import load_data

from src.preprocessing.preprocess import (
    split_features_target,
    identify_feature_types,
    build_preprocessor,
    split_data,
    transform_data
)

from src.training.train import get_models


# 1. Project paths
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "mental_health_model.joblib"


# 2. Load dataset
df = load_data()

# 3. Separate features and target
X, y = split_features_target(df)

# 4. Identify feature types
numerical_features, categorical_features = identify_feature_types(X)

# 5. Split dataset
X_train, X_test, y_train, y_test = split_data(X, y)

# 6. Build and fit preprocessor
preprocessor = build_preprocessor(
    numerical_features,
    categorical_features
)

X_train_processed, X_test_processed = transform_data(
    preprocessor,
    X_train,
    X_test
)

# 7. Train Logistic Regression
model = get_models()["Logistic Regression"]

model.fit(X_train_processed, y_train)

# 8. Save model and preprocessor together
model_data = {
    "model": model,
    "preprocessor": preprocessor,
    "feature_columns": X.columns.tolist()
}

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model_data, MODEL_PATH)

print("Model and preprocessor saved successfully!")
print(f"Saved at: {MODEL_PATH}")