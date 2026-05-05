"""Tests for preprocessing module."""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import preprocess, handle_missing_values, binarize_target


class TestPreprocessing:
    """Test suite for data preprocessing."""

    def test_preprocess_returns_dataframe(self):
        df = preprocess(save=False)
        assert isinstance(df, pd.DataFrame)

    def test_no_missing_values_after_preprocess(self):
        df = preprocess(save=False)
        assert df.isnull().sum().sum() == 0

    def test_target_column_exists(self):
        df = preprocess(save=False)
        assert "target" in df.columns

    def test_target_is_binary(self):
        df = preprocess(save=False)
        assert set(df["target"].unique()).issubset({0, 1})

    def test_original_target_removed(self):
        df = preprocess(save=False)
        assert "num" not in df.columns

    def test_handle_missing_values(self):
        # Create test data with missing values
        test_df = pd.DataFrame(
            {
                "a": [1.0, 2.0, None, 4.0],
                "b": [10.0, None, 30.0, 40.0],
            }
        )
        result = handle_missing_values(test_df)
        assert result.isnull().sum().sum() == 0
        # Median of [1, 2, 4] = 2.0
        assert result["a"].iloc[2] == 2.0

    def test_binarize_target(self):
        test_df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "num": [0, 1, 2, 3, 4],
            }
        )
        result = binarize_target(test_df)
        assert "target" in result.columns
        assert "num" not in result.columns
        assert result["target"].tolist() == [0, 1, 1, 1, 1]

    def test_correct_number_of_features(self):
        df = preprocess(save=False)
        # 13 features + 1 target
        assert df.shape[1] == 14
