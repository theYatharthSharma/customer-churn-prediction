"""Loading the saved model, scoring customers and explaining predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

import config
from src.preprocessing import select_features


class ModelNotTrainedError(RuntimeError):
    """Raised when no usable trained model is available."""


class PredictionInputError(ValueError):
    """Raised when data passed for prediction is unusable."""


# --------------------------------------------------------------------------- #
# Model artifacts
# --------------------------------------------------------------------------- #
def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def model_status(models_dir: Path = config.MODELS_DIR) -> tuple[str, str]:
    """Return ('ok' | 'missing' | 'incompatible', human-readable reason)."""
    models_dir = Path(models_dir)
    model_path = models_dir / config.BEST_MODEL_FILENAME
    meta_path = models_dir / config.METADATA_FILENAME
    if not model_path.is_file() or not meta_path.is_file():
        return "missing", "No trained model was found."
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "incompatible", "The model metadata file is unreadable."
    if meta.get("schema_version") != config.SCHEMA_VERSION:
        return "incompatible", "The saved model was built with a different project version."
    if meta.get("feature_columns") != config.FEATURE_COLUMNS:
        return "incompatible", "The saved model uses a different set of input features."
    saved = meta.get("versions", {})
    for lib, current in (("scikit-learn", sklearn.__version__), ("xgboost", xgboost.__version__)):
        if saved.get(lib) and _major_minor(saved[lib]) != _major_minor(current):
            return "incompatible", (
                f"The model was saved with {lib} {saved[lib]} but {current} is installed.")
    return "ok", "A compatible trained model is available."


def load_artifacts(models_dir: Path = config.MODELS_DIR):
    """Load (pipeline, metadata). Raises ModelNotTrainedError if unusable."""
    status, reason = model_status(models_dir)
    if status != "ok":
        raise ModelNotTrainedError(reason)
    models_dir = Path(models_dir)
    try:
        pipeline = joblib.load(models_dir / config.BEST_MODEL_FILENAME)
        metadata = json.loads((models_dir / config.METADATA_FILENAME).read_text(encoding="utf-8"))
    except Exception as exc:  # corrupted file, unpickling error, ...
        raise ModelNotTrainedError(f"The saved model could not be loaded ({exc}).") from exc
    return pipeline, metadata


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def risk_category(probability: float,
                  low_max: float = config.RISK_LOW_MAX,
                  high_min: float = config.RISK_HIGH_MIN) -> str:
    """Map a churn probability to a model-based risk band."""
    if probability >= high_min:
        return "High"
    if probability >= low_max:
        return "Medium"
    return "Low"


def validate_prediction_input(df: pd.DataFrame, metadata: Optional[dict] = None
                              ) -> tuple[pd.DataFrame, list[str]]:
    """Check an uploaded frame and return (normalised frame, warnings).

    The target column ``Churn`` is not required (and is ignored if present).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise PredictionInputError(
            "Missing required column(s): " + ", ".join(missing)
            + ". Download the CSV template to see the expected layout.")
    if df.empty:
        raise PredictionInputError("The uploaded file contains no rows.")

    warnings_list: list[str] = []
    features = select_features(df)

    for col in config.NUMERIC_FEATURES:
        n_bad = int(features[col].isna().sum())
        if n_bad:
            warnings_list.append(
                f"{n_bad} row(s) have a missing/invalid '{col}'; the training median is used.")

    known = (metadata or {}).get("categorical_options", {})
    for col in config.CATEGORICAL_FEATURES:
        n_missing = int(features[col].isna().sum())
        if n_missing:
            warnings_list.append(
                f"{n_missing} row(s) have a missing '{col}'; the most frequent training value is used.")
        if col in known:
            unseen = sorted(set(features[col].dropna().astype(str)) - set(known[col]))
            if unseen:
                warnings_list.append(
                    f"'{col}' contains value(s) not seen in training: {unseen[:5]}. "
                    "They are encoded as 'unknown' (all-zero) for that column.")
    return df, warnings_list


def predict_dataframe(pipeline, df: pd.DataFrame,
                      threshold: float = config.DECISION_THRESHOLD,
                      low_max: float = config.RISK_LOW_MAX,
                      high_min: float = config.RISK_HIGH_MIN) -> pd.DataFrame:
    """Score every row using the saved pipeline (preprocessing is part of it).

    Returns the original columns (minus Churn) plus
    ``Churn_Probability``, ``Predicted_Churn`` and ``Risk_Category``.
    """
    features = select_features(df)
    proba = pipeline.predict_proba(features)[:, 1]
    out = df.drop(columns=[config.TARGET_COLUMN], errors="ignore").copy()
    out["Churn_Probability"] = np.round(proba, 4)
    out["Predicted_Churn"] = np.where(proba >= threshold, "Yes", "No")
    out["Risk_Category"] = [risk_category(p, low_max, high_min) for p in proba]
    return out


def harmonize_service_inputs(inputs: dict) -> tuple[dict, list[str]]:
    """Make form answers consistent with how the Telco dataset encodes services."""
    fixed, notes = dict(inputs), []
    if fixed["PhoneService"] == "No" and fixed["MultipleLines"] != "No phone service":
        fixed["MultipleLines"] = "No phone service"
        notes.append("Phone service is 'No', so Multiple Lines was set to 'No phone service'.")
    if fixed["PhoneService"] == "Yes" and fixed["MultipleLines"] == "No phone service":
        fixed["MultipleLines"] = "No"
        notes.append("Phone service is 'Yes', so Multiple Lines was set to 'No'.")
    if fixed["InternetService"] == "No":
        changed = [c for c in config.INTERNET_ADDON_COLUMNS if fixed[c] != "No internet service"]
        for col in changed:
            fixed[col] = "No internet service"
        if changed:
            notes.append("Internet service is 'No', so internet add-ons were set to 'No internet service'.")
    else:
        changed = [c for c in config.INTERNET_ADDON_COLUMNS if fixed[c] == "No internet service"]
        for col in changed:
            fixed[col] = "No"
        if changed:
            notes.append("Customer has internet service, so 'No internet service' add-ons were set to 'No'.")
    return fixed, notes


# --------------------------------------------------------------------------- #
# Individual explanation (LOCAL) - model-agnostic what-if sensitivity
# --------------------------------------------------------------------------- #
def explain_customer(pipeline, customer: pd.DataFrame, baselines: dict, top_n: int = 8) -> pd.DataFrame:
    """Explain ONE prediction by swapping each feature for a 'typical customer' value.

    For every input feature we replace the customer's value with the training
    median/mode and measure how the predicted probability changes.  A positive
    change means "this customer's value pushes the predicted churn probability
    UP relative to a typical value".  This describes the model's behaviour for
    this customer; it does not prove that the feature causes churn, and the
    effects of correlated features are not additive.
    """
    customer = select_features(customer).iloc[[0]].reset_index(drop=True)
    base_prob = float(pipeline.predict_proba(customer)[0, 1])

    variants, cols = [], []
    for col in config.FEATURE_COLUMNS:
        if customer.at[0, col] == baselines[col]:
            continue  # already typical -> no effect to measure
        variant = customer.copy()
        variant[col] = baselines[col]
        variants.append(variant)
        cols.append(col)
    if not variants:
        return pd.DataFrame(columns=["Feature", "Customer value", "Typical value", "Effect on probability (pp)"])

    probs = pipeline.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    table = pd.DataFrame({
        "Feature": cols,
        "Customer value": [str(customer.at[0, c]) for c in cols],
        "Typical value": [str(round(baselines[c], 2)) if isinstance(baselines[c], float)
                          else str(baselines[c]) for c in cols],
        "Effect on probability (pp)": np.round((base_prob - probs) * 100, 2),
    })
    order = table["Effect on probability (pp)"].abs().sort_values(ascending=False).index
    return table.loc[order].head(top_n).reset_index(drop=True)


def retention_suggestions(inputs: dict, risk: str) -> list[tuple[str, str]]:
    """Rule-based retention ideas (business heuristics, not model output)."""
    tips: list[tuple[str, str]] = []
    has_internet = inputs["InternetService"] != "No"

    if inputs["Contract"] == "Month-to-month":
        tips.append(("Offer a longer contract",
                     "Month-to-month customers can leave at any time. Offer a discount or "
                     "perk for moving to a one- or two-year plan."))
    if inputs["PaymentMethod"] == "Electronic check":
        tips.append(("Encourage automatic payments",
                     "Offer a small bill credit for switching to bank transfer or card "
                     "auto-pay, which reduces payment friction."))
    if has_internet and inputs["TechSupport"] == "No":
        tips.append(("Offer a tech-support trial",
                     "Provide a free or discounted tech-support period so problems are "
                     "solved before they cause frustration."))
    if has_internet and inputs["OnlineSecurity"] == "No":
        tips.append(("Bundle security / backup services",
                     "Add-on services such as online security can be offered as a "
                     "low-cost bundle to deepen the relationship."))
    if inputs["InternetService"] == "Fiber optic" and float(inputs["MonthlyCharges"]) >= 70:
        tips.append(("Review price and service quality",
                     "Fiber customers with high monthly charges may be price-sensitive. "
                     "Check for service issues and consider a price-lock or bundle offer."))
    if float(inputs["tenure"]) <= 12:
        tips.append(("Strengthen onboarding",
                     "Newer customers have had less time to build loyalty. Schedule "
                     "check-in calls and usage tips during the first year."))

    if not tips:
        tips.append(("Keep engaging", "No specific friction points were found in the inputs. "
                     "Continue regular service and loyalty communications."))
    if risk == "High":
        tips.insert(0, ("Prioritise personal outreach",
                        "This customer is in the model's high-risk band: consider a "
                        "proactive call from a retention specialist."))
    return tips
