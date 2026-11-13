from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


def get_models():
    """
    Return the classification models.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="mlogloss"
        )
    }

    return models



def train_models(models, X_train, y_train):
    """
    Train all classification models.
    """

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")

        if name == "XGBoost":
            model.fit(X_train, y_train_encoded)
        else:
            model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} trained successfully!")

    return trained_models, label_encoder