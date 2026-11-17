"""Dataset loading, validation and cleaning.

The project never fabricates data: it reads the CSV supplied by the user
(either ``data/WA_Fn-UseC_-Telco-Customer-Churn.csv`` or an uploaded file) and
stops with a clear message when the file is missing or malformed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

import config
from src.preprocessing import normalize_features

logger = logging.getLogger(__name__)

# customerID is optional - it is never used as a feature.
REQUIRED_COLUMNS = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the Telco CSV is not in the data/ folder."""


class DatasetValidationError(ValueError):
    """Raised when a CSV exists but does not look like the Telco dataset."""


def dataset_exists(path: Optional[Path] = None) -> bool:
    """True if the dataset file exists (default: data/WA_Fn-UseC_-Telco-Customer-Churn.csv)."""
    return Path(path if path is not None else config.DATASET_PATH).is_file()


def read_csv(source) -> pd.DataFrame:
    """Read a CSV from a path or file-like object with friendly errors."""
    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError("The CSV file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetValidationError(
            f"The file could not be read as a CSV ({exc})."
        ) from exc
    if df.empty:
        raise DatasetValidationError("The CSV file has headers but no rows.")
    return df


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate and clean the raw Telco frame.

    Returns ``(clean_df, notes)`` where ``notes`` documents every data-quality
    action taken (shown in the EDA page).  Cleaning here is stateless, so doing
    it before the train/test split does not leak information.
    """
    notes: list[str] = []
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(
            "The CSV is missing required column(s): " + ", ".join(missing)
            + ". Expected the IBM Telco Customer Churn dataset."
        )

    # --- target column -------------------------------------------------- #
    n_before = len(df)
    df = df.dropna(subset=[config.TARGET_COLUMN])
    if len(df) < n_before:
        notes.append(f"Dropped {n_before - len(df)} row(s) with a missing Churn label.")
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(str).str.strip().str.title()
    bad_labels = sorted(set(df[config.TARGET_COLUMN]) - {"Yes", "No"})
    if bad_labels:
        raise DatasetValidationError(
            f"Churn column must contain only 'Yes'/'No' but also has: {bad_labels[:5]}"
        )

    # --- exact duplicate rows -------------------------------------------- #
    n_before = len(df)
    df = df.drop_duplicates()
    if len(df) < n_before:
        notes.append(f"Removed {n_before - len(df)} exact duplicate row(s).")

    # --- TotalCharges (blank strings in the raw file) -------------------- #
    unparsable = int(pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum())
    df = normalize_features(df)
    still_missing = int(df["TotalCharges"].isna().sum())
    if unparsable:
        notes.append(
            f"TotalCharges had {unparsable} blank/non-numeric value(s); it was converted "
            f"to a numeric column. {unparsable - still_missing} belonged to customers with "
            f"tenure 0 and were set to 0; {still_missing} remain missing and are imputed "
            f"(median, fitted on training data only)."
        )

    if len(df) < config.MIN_ROWS_FOR_TRAINING:
        raise DatasetValidationError(
            f"Only {len(df)} usable rows found; at least "
            f"{config.MIN_ROWS_FOR_TRAINING} are needed."
        )
    if df[config.TARGET_COLUMN].nunique() < 2:
        raise DatasetValidationError("Churn column contains only one class.")

    return df.reset_index(drop=True), notes


def load_dataset_with_notes(
    source: Union[str, Path, object, None] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load + validate + clean the dataset. ``source=None`` uses the default path."""
    if source is None:
        if not dataset_exists():
            raise DatasetNotFoundError(config.DATASET_MISSING_MESSAGE)
        source = Path(config.DATASET_PATH)
    raw = read_csv(source)
    df, notes = clean_dataset(raw)
    logger.info("Loaded dataset: %d rows x %d columns", *df.shape)
    return df, notes


def load_dataset(source=None) -> pd.DataFrame:
    """Convenience wrapper returning only the cleaned DataFrame."""
    return load_dataset_with_notes(source)[0]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). X excludes customerID and Churn; y is 1 = churned, 0 = retained."""
    X = df[config.FEATURE_COLUMNS].copy()
    y = (df[config.TARGET_COLUMN] == "Yes").astype(int)
    return X, y


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Small summary dictionary used for logging and the EDA page."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "churn_rate": float((df[config.TARGET_COLUMN] == "Yes").mean()),
    }
