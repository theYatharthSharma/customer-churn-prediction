"""Automated checks for the Customer Churn Prediction System.

Run with:  python -m pytest -q
The tests use a small schema-compatible frame from tests/fixtures.py (test-only).
"""
import io

import pandas as pd
import pytest

import config
from src import data_loader, predict, train
from src.preprocessing import build_preprocessor, normalize_features, select_features
from tests.fixtures import make_schema_frame


@pytest.fixture(scope="module")
def raw_frame():
    return make_schema_frame(600, seed=1)


@pytest.fixture(scope="module")
def clean_frame(raw_frame):
    return data_loader.clean_dataset(raw_frame)[0]


@pytest.fixture(scope="module")
def trained(tmp_path_factory, clean_frame):
    base = tmp_path_factory.mktemp("artifacts")
    results = train.train_all_models(clean_frame, models_dir=base / "models",
                                     reports_dir=base / "reports", quick=True)
    return base, results


# ---- data loading --------------------------------------------------------- #
def test_missing_dataset_message(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATASET_PATH", tmp_path / "nope.csv")
    with pytest.raises(data_loader.DatasetNotFoundError, match="Dataset not found"):
        data_loader.load_dataset_with_notes(None)


def test_missing_columns_rejected(raw_frame):
    with pytest.raises(data_loader.DatasetValidationError, match="Contract"):
        data_loader.clean_dataset(raw_frame.drop(columns=["Contract"]))


def test_total_charges_converted(raw_frame):
    df, notes = data_loader.clean_dataset(raw_frame)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0
    assert any("TotalCharges" in n for n in notes)


def test_csv_round_trip(raw_frame):
    buffer = io.StringIO(raw_frame.to_csv(index=False))
    df, _ = data_loader.load_dataset_with_notes(buffer)
    assert len(df) == len(raw_frame)


def test_customer_id_not_a_feature(clean_frame):
    X, y = data_loader.split_features_target(clean_frame)
    assert config.ID_COLUMN not in X.columns and config.TARGET_COLUMN not in X.columns
    assert set(y.unique()) <= {0, 1}


# ---- preprocessing -------------------------------------------------------- #
def test_preprocessor_handles_unseen_category(clean_frame):
    X, _ = data_loader.split_features_target(clean_frame)
    pre = build_preprocessor().fit(X)
    odd = X.iloc[[0]].copy()
    odd["Contract"] = "Five year"  # never seen
    assert pre.transform(odd).shape[0] == 1


def test_senior_citizen_yes_no_mapping():
    df = normalize_features(pd.DataFrame({"SeniorCitizen": ["Yes", "No", 1, 0]}))
    assert df["SeniorCitizen"].tolist() == [1, 0, 1, 0]


# ---- training, saving, loading ---------------------------------------------- #
def test_training_outputs(trained):
    base, results = trained
    assert set(results["models"]) == {"Logistic Regression", "Decision Tree",
                                      "Random Forest", "XGBoost"}
    assert results["selected_model"] in results["models"]
    for name in ("best_model.joblib", "model_metadata.json"):
        assert (base / "models" / name).is_file()
    for name in ("evaluation_results.json", "model_comparison.csv", "roc_curves.png",
                 "confusion_matrices.png", "feature_importance.png"):
        assert (base / "reports" / name).is_file()


def test_saved_model_reusable(trained, clean_frame):
    base, _ = trained
    assert predict.model_status(base / "models")[0] == "ok"
    pipeline, meta = predict.load_artifacts(base / "models")
    scored = predict.predict_dataframe(pipeline, clean_frame.head(20))
    assert scored["Churn_Probability"].between(0, 1).all()
    assert set(scored["Risk_Category"]) <= {"Low", "Medium", "High"}
    assert "Churn" not in scored.columns


def test_missing_model_reported(tmp_path):
    assert predict.model_status(tmp_path)[0] == "missing"
    with pytest.raises(predict.ModelNotTrainedError):
        predict.load_artifacts(tmp_path)


# ---- prediction ------------------------------------------------------------- #
def test_batch_validation_errors(trained, clean_frame):
    base, _ = trained
    _, meta = predict.load_artifacts(base / "models")
    with pytest.raises(predict.PredictionInputError, match="tenure"):
        predict.validate_prediction_input(clean_frame.drop(columns=["tenure"]), meta)
    ok, warns = predict.validate_prediction_input(clean_frame.drop(columns=["Churn"]).head(5), meta)
    assert len(ok) == 5 and isinstance(warns, list)


def test_single_prediction_and_explanation(trained, clean_frame):
    base, _ = trained
    pipeline, meta = predict.load_artifacts(base / "models")
    row = clean_frame.drop(columns=["Churn"]).head(1)
    table = predict.explain_customer(pipeline, row, meta["baselines"])
    assert {"Feature", "Effect on probability (pp)"} <= set(table.columns)


def test_risk_bands_are_configurable():
    assert predict.risk_category(0.1, 0.3, 0.6) == "Low"
    assert predict.risk_category(0.4, 0.3, 0.6) == "Medium"
    assert predict.risk_category(0.9, 0.3, 0.6) == "High"
    assert predict.risk_category(0.4, 0.1, 0.3) == "High"


def test_form_consistency_rules():
    inputs = {c: v[0] for c, v in config.FEATURE_OPTIONS.items()}
    inputs.update(PhoneService="No", MultipleLines="Yes", InternetService="No", OnlineSecurity="Yes")
    fixed, notes = predict.harmonize_service_inputs(inputs)
    assert fixed["MultipleLines"] == "No phone service"
    assert fixed["OnlineSecurity"] == "No internet service" and len(notes) == 2
