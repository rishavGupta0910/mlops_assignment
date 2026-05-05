"""
Data loader for the Heart Disease UCI dataset (Cleveland).

Downloads the dataset directly from the UCI ML Repository
using the ucimlrepo package (dataset ID: 45).

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATASET_ID = 45  # Heart Disease UCI


def load_cleveland_data():
    """
    Fetch the Cleveland heart disease dataset from UCI ML Repository.

    Returns a single DataFrame with 13 features + 1 target column ('num').
    Missing values are represented as NaN.

    Returns
    -------
    pd.DataFrame
        DataFrame with 14 columns (13 features + 'num' target).
        303 patient records.
    """
    heart_disease = fetch_ucirepo(id=DATASET_ID)

    X = heart_disease.data.features  # 13 features
    y = heart_disease.data.targets  # 'num' column

    df = pd.concat([X, y], axis=1)

    return df


if __name__ == "__main__":
    print("Fetching Cleveland heart disease data from UCI Repository...")
    df = load_cleveland_data()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution (num):\n{df['num'].value_counts().sort_index()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
