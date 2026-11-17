"""Feature preprocessing: stateless cleaning + the scikit-learn preprocessor.

Two clearly separated steps:

1. ``normalize_features`` - *stateless* type fixes (e.g. TotalCharges -> number).
   It learns nothing from the data, so applying it before the train/test split
   cannot leak information.
2. ``build_preprocessor`` - a ``ColumnTransformer`` (imputation, scaling and
   one-hot encoding).  It is placed *inside* the model Pipeline, so it is fitted
   on training data only (and on training folds only during cross-validation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

_SENIOR_MAP = {
    "yes": 1, "no": 0, "true": 1, "false": 0,
    "1": 1, "0": 0, "1.0": 1, "0.0": 0,
}


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with correct dtypes for every model input column.

    - strips stray whitespace from text columns
    - converts TotalCharges (blank strings in the raw file) to numbers
    - maps SeniorCitizen (0/1 or Yes/No) to 0/1
    - a customer with tenure 0 has not been billed yet, so a missing
      TotalCharges for them is set to 0 (a business rule, not a statistic)
    Anything that still cannot be parsed becomes NaN and is imputed later
    inside the pipeline.
    """
    out = df.copy()

    for col in config.CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
            out[col] = out[col].astype(object)

    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = (
            out["SeniorCitizen"].astype(str).str.strip().str.lower().map(_SENIOR_MAP)
        )

    if {"tenure", "TotalCharges"}.issubset(out.columns):
        new_customer = (out["tenure"] == 0) & out["TotalCharges"].isna()
        out.loc[new_customer, "TotalCharges"] = 0.0

    return out


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise and return only the 19 model input columns, in a fixed order."""
    return normalize_features(df)[config.FEATURE_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    """Create the (unfitted) preprocessing step used by every model.

    * numeric columns     -> median imputation -> standard scaling
    * categorical columns -> most-frequent imputation -> one-hot encoding
      (``handle_unknown='ignore'`` so unseen categories never crash prediction)
    customerID is never passed in, so it cannot become a feature.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def compute_baselines(features: pd.DataFrame) -> dict:
    """'Typical customer' values (median / mode) from the TRAINING features.

    Used only to explain individual predictions (see ``predict.explain_customer``).
    """
    baselines: dict = {}
    for col in config.NUMERIC_FEATURES:
        baselines[col] = float(np.nanmedian(features[col].astype(float)))
    for col in config.CATEGORICAL_FEATURES:
        baselines[col] = str(features[col].mode(dropna=True).iloc[0])
    return baselines
