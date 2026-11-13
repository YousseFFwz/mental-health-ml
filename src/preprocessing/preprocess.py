from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


TARGET = "Mental_Health_Risk"

DROP_COLUMNS = ["Student_ID", TARGET]


def split_features_target(df):
    """
    Separate input features (X) from target (y).
    """
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    return X, y

def identify_feature_types(X):
    """
    Identify numerical and categorical columns.
    """
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numerical_features, categorical_features

def build_preprocessor(numerical_features, categorical_features):

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

def split_data(X, y):
    """
    Split data into training and testing sets.
    """
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def transform_data(preprocessor, X_train, X_test):
    """
    Fit the preprocessor on training data
    and transform both training and testing data.
    """
    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed