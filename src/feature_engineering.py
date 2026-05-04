"""
Feature engineering pipeline for the Heart Disease dataset.

Builds a scikit-learn preprocessing pipeline with:
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- Train/test split with stratification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Feature categorization based on domain knowledge
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"


def get_feature_lists():
    """Return the numerical and categorical feature lists."""
    return NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor():
    """
    Build a scikit-learn ColumnTransformer for feature preprocessing.

    - Numerical features: StandardScaler (zero mean, unit variance)
    - Categorical features: OneHotEncoder (sparse=False, handle_unknown='ignore')

    Returns
    -------
    ColumnTransformer
        Fitted-ready preprocessing transformer.
    """
    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", sparse_output=False,
                                 handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and separate features from target.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with 'target' column.
    test_size : float
        Proportion for test split.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import preprocess

    df = preprocess(save=False)
    X_train, X_test, y_train, y_test = prepare_data(df)

    print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Train target dist: {y_train.value_counts().to_dict()}")
    print(f"Test target dist: {y_test.value_counts().to_dict()}")

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    print(f"\nTransformed train shape: {X_train_transformed.shape}")
    print(f"Transformed test shape: {X_test_transformed.shape}")
    print(f"Feature names: {preprocessor.get_feature_names_out()}")
