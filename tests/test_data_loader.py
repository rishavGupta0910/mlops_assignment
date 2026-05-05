"""Tests for data_loader module."""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_cleveland_data


class TestDataLoader:
    """Test suite for the data loading functionality."""

    def test_load_returns_dataframe(self):
        df = load_cleveland_data()
        assert isinstance(df, pd.DataFrame)

    def test_load_correct_shape(self):
        df = load_cleveland_data()
        assert df.shape[0] == 303
        assert df.shape[1] == 14

    def test_load_expected_columns(self):
        df = load_cleveland_data()
        expected = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_target_column_values(self):
        df = load_cleveland_data()
        assert df["num"].isin([0, 1, 2, 3, 4]).all()

    def test_no_empty_dataframe(self):
        df = load_cleveland_data()
        assert len(df) > 0
