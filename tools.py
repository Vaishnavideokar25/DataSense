import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# -----------------------------------------
# DATASET PROFILING
# -----------------------------------------


def profile_dataset(df):

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "numeric_columns": list(df.select_dtypes(include=np.number).columns),
        "categorical_columns": list(df.select_dtypes(exclude=np.number).columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    return profile


# -----------------------------------------
# DESCRIPTIVE STATISTICS
# -----------------------------------------


def descriptive_statistics(df):

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return "No numerical columns found."

    return numeric_df.describe().round(2)


# -----------------------------------------
# CORRELATION ANALYSIS
# -----------------------------------------


def correlation_analysis(df):

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    correlation_matrix = numeric_df.corr()

    results = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            value = correlation_matrix.iloc[i, j]

            if pd.notna(value):

                results.append(
                    {
                        "Feature 1": columns[i],
                        "Feature 2": columns[j],
                        "Correlation": round(value, 3),
                    }
                )

    results = sorted(results, key=lambda x: abs(x["Correlation"]), reverse=True)

    return pd.DataFrame(results)


# -----------------------------------------
# MACHINE LEARNING MODEL
# -----------------------------------------


def train_model(df, target):

    if target not in df.columns:

        return {"error": "Target column not found."}

    data = df.dropna(subset=[target]).copy()

    X = data.drop(columns=[target])

    y = data[target]

    # Remove ID-like columns

    id_columns = [
        column
        for column in X.columns
        if column.lower() == "id" or column.lower().endswith("_id")
    ]

    X = X.drop(columns=id_columns, errors="ignore")

    numeric_features = list(X.select_dtypes(include=np.number).columns)

    categorical_features = list(X.select_dtypes(exclude=np.number).columns)

    # Numerical preprocessing

    numerical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    # Categorical preprocessing

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        [
            ("numerical", numerical_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # Decide classification vs regression

    is_classification = y.dtype == "object" or y.nunique() <= 10

    if is_classification:

        model = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )

    else:

        model = RandomForestRegressor(n_estimators=200, random_state=42)

    pipeline = Pipeline([("preprocessing", preprocessing), ("model", model)])

    # Train-test split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    # Classification metrics

    if is_classification:

        accuracy = accuracy_score(y_test, predictions)

        f1 = f1_score(y_test, predictions, average="weighted")

        return {
            "task": "Classification",
            "accuracy": round(accuracy, 3),
            "f1_score": round(f1, 3),
        }

    # Regression metrics

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    return {
        "task": "Regression",
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
    }
