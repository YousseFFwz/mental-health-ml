import pandas as pd


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)




def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model.
    """

    y_pred = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, average="macro"
        ),
        "Recall": recall_score(
            y_test, y_pred, average="macro"
        ),
        "F1-Score": f1_score(
            y_test, y_pred, average="macro"
        )
    }

    return results




def compare_models(trained_models, X_test, y_test, label_encoder):

    results = []

    for name, model in trained_models.items():

        y_pred = model.predict(X_test)

        # Decode XGBoost predictions
        if name == "XGBoost":
            y_pred = label_encoder.inverse_transform(
                y_pred.astype(int)
            )

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, average="macro"
            ),
            "Recall": recall_score(
                y_test, y_pred, average="macro"
            ),
            "F1-Score": f1_score(
                y_test, y_pred, average="macro"
            )
        })

    return pd.DataFrame(results).sort_values(
        by="F1-Score",
        ascending=False
    )

