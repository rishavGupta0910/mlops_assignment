"""
Preprocessing module for the Heart Disease UCI dataset.

Handles:
- Missing value imputation
- Target binarization (0 vs 1-4 -> 0 vs 1)
- Saving cleaned dataset
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_cleveland_data


def handle_missing_values(df):
    """
    Impute missing values using median strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with potential NaN values.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df


def binarize_target(df, target_col="num"):
    """
    Convert the target variable from multi-class (0-4) to binary.
    0 = no heart disease, 1 = heart disease present (original values 1-4).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with target column.
    target_col : str
        Name of the target column.

    Returns
    -------
    pd.DataFrame
        DataFrame with binary target column.
    """
    df = df.copy()
    df["target"] = (df[target_col] > 0).astype(int)
    df.drop(columns=[target_col], inplace=True)
    return df


def preprocess(save=True, output_dir=None):
    """
    Full preprocessing pipeline:
    1. Fetch data from UCI Repository
    2. Handle missing values
    3. Binarize target variable
    4. Optionally save cleaned dataset

    Parameters
    ----------
    save : bool
        Whether to save the cleaned dataset to CSV.
    output_dir : str, optional
        Directory to save cleaned CSV. Defaults to data/processed/.

    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed DataFrame.
    """
    # Fetch from UCI
    df = load_cleveland_data()
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Handle missing values
    df = handle_missing_values(df)
    print(f"After imputation: {df.isnull().sum().sum()} missing values")

    # Binarize target
    df = binarize_target(df)
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")

    # Save
    if save:
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "data", "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "cleveland_cleaned.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")

    return df


if __name__ == "__main__":
    df = preprocess()
    print(f"\nFinal shape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nDescribe:\n{df.describe()}")
