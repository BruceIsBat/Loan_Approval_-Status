"""Feature engineering for the employee loan approval pipeline."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a minimal feature set for loan approval modeling.

    The project is intentionally lightweight and production-safe: it does basic
    imputations, creates a debt-to-income feature when relevant, and scales
    numeric columns for downstream ML models.
    """
    working_df = df.copy()

    numeric_cols = working_df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        working_df[numeric_cols] = working_df[numeric_cols].fillna(
            working_df[numeric_cols].median()
        )

    if "loan_amount" in working_df.columns and "salary" in working_df.columns:
        salary = working_df["salary"].replace(0, pd.NA)
        working_df["debt_to_income"] = working_df["loan_amount"] / salary

    for column in ("applicant_id", "name"):
        if column in working_df.columns:
            working_df = working_df.drop(columns=[column])

    numeric_cols = working_df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        working_df[numeric_cols] = scaler.fit_transform(working_df[numeric_cols])

    return working_df
