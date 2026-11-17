"""Model training pipeline.

Steps: split (stratified) -> for each model: Pipeline(preprocessor + classifier)
-> randomised hyper-parameter search with stratified cross-validation on the
TRAINING set only -> evaluate on the untouched test set -> pick the final model
using cross-validated scores -> save model, metadata and reports.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config
from src import evaluate
from src.data_loader import split_features_target
from src.preprocessing import build_preprocessor, compute_baselines

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, float], None]

CV_SCORING = {"roc_auc": "roc_auc", "f1": "f1", "recall": "recall", "precision": "precision"}


def _to_builtin(obj):
    """Recursively convert numpy types so the object is JSON serialisable."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_model_definitions(scale_pos_weight: float, quick: bool = False) -> dict:
    """Return {name: (classifier, param_grid)}.

    Class imbalance is handled *inside* each model (class weights /
    scale_pos_weight), so no resampling is needed and no information can leak
    from the test set.  ``quick=True`` shrinks the models for fast unit tests.
    """
    trees = 40 if quick else None
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=config.RANDOM_STATE),
            {"classifier__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=config.RANDOM_STATE),
            {
                "classifier__max_depth": [3, 4, 5, 6, 8, 10],
                "classifier__min_samples_leaf": [5, 10, 20, 50],
                "classifier__criterion": ["gini", "entropy"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1,
                                   random_state=config.RANDOM_STATE),
            {
                "classifier__n_estimators": [trees] if quick else [200, 300, 400],
                "classifier__max_depth": [6, 8, 12, None],
                "classifier__min_samples_leaf": [1, 3, 5, 10],
                "classifier__max_features": ["sqrt", "log2"],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                scale_pos_weight=scale_pos_weight, n_jobs=1,
                random_state=config.RANDOM_STATE,
            ),
            {
                "classifier__n_estimators": [trees] if quick else [200, 300, 400],
                "classifier__max_depth": [3, 4, 5],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.7, 1.0],
                "classifier__min_child_weight": [1, 3, 5],
            },
        ),
    }


def _grid_size(grid: dict) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _selection_score(cv_scores: dict) -> float:
    return float(sum(w * cv_scores[m] for m, w in config.SELECTION_WEIGHTS.items()))


def _selection_text(best: str, models: dict) -> str:
    weights = " + ".join(f"{w:.0%} x CV {m.upper().replace('_', '-')}"
                         for m, w in config.SELECTION_WEIGHTS.items())
    ranked = sorted(models.items(), key=lambda kv: kv[1]["selection_score"], reverse=True)
    others = ", ".join(f"{n} ({r['selection_score']:.3f})" for n, r in ranked if n != best)
    b = models[best]
    return (
        f"{best} was selected because it had the highest cross-validated selection score "
        f"({b['selection_score']:.3f} = {weights}), measured on the training folds only. "
        f"Accuracy was not used because only about a quarter of customers churn, so a "
        f"model can look accurate while missing churners. ROC-AUC measures ranking quality "
        f"across all thresholds and F1 balances precision with recall on the churn class. "
        f"Other models: {others}. Test-set results are reported for honesty but were not used "
        f"to choose the model."
    )


def train_all_models(
    df: pd.DataFrame,
    progress_callback: Optional[ProgressCallback] = None,
    models_dir: Path = config.MODELS_DIR,
    reports_dir: Path = config.REPORTS_DIR,
    quick: bool = False,
) -> dict:
    """Train, evaluate, select and save everything. Returns the evaluation results."""
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    def report(message: str, fraction: float) -> None:
        logger.info(message)
        if progress_callback:
            progress_callback(message, fraction)

    # ---- 1. split BEFORE any fitting (prevents leakage) -------------------- #
    X, y = split_features_target(df)
    if y.nunique() < 2:
        raise ValueError("Training needs both churned and retained customers.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    report(f"Split data: {len(X_train)} training rows, {len(X_test)} test rows.", 0.02)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    definitions = get_model_definitions(scale_pos_weight, quick=quick)
    cv = StratifiedKFold(n_splits=3 if quick else config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)

    model_results: dict = {}
    fitted: dict = {}
    n_models = len(definitions)

    for idx, (name, (classifier, grid)) in enumerate(definitions.items()):
        report(f"Tuning and cross-validating {name} ({idx + 1}/{n_models})...",
               0.05 + 0.85 * idx / n_models)
        # Preprocessing lives inside the pipeline -> refitted on every CV training fold.
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", classifier)])
        n_iter = min(2 if quick else config.SEARCH_ITERATIONS, _grid_size(grid))
        search = RandomizedSearchCV(
            pipeline, grid, n_iter=n_iter, scoring=CV_SCORING, refit="roc_auc",
            cv=cv, n_jobs=-1, random_state=config.RANDOM_STATE, error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)

        best_i = search.best_index_
        cv_scores = {m: float(search.cv_results_[f"mean_test_{m}"][best_i]) for m in CV_SCORING}
        cv_scores["roc_auc_std"] = float(search.cv_results_["std_test_roc_auc"][best_i])

        best_pipeline = search.best_estimator_
        result = evaluate.evaluate_model(best_pipeline, X_test, y_test, config.DECISION_THRESHOLD)
        result["best_params"] = {k.replace("classifier__", ""): v
                                 for k, v in search.best_params_.items()}
        result["cv"] = cv_scores
        result["selection_score"] = _selection_score(cv_scores)
        result["feature_importance"] = evaluate.extract_feature_importance(best_pipeline)
        model_results[name] = result
        fitted[name] = best_pipeline

    # ---- 2. choose the final model from CROSS-VALIDATION scores ------------ #
    best_name = max(model_results, key=lambda n: model_results[n]["selection_score"])
    rationale = _selection_text(best_name, model_results)
    report(f"Selected model: {best_name}. Saving artifacts...", 0.93)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": best_name,
        "selection_rationale": rationale,
        "selection_weights": config.SELECTION_WEIGHTS,
        "decision_threshold": config.DECISION_THRESHOLD,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_churn_rate": float(y_test.mean()),
        "models": _to_builtin(model_results),
    }

    # ---- 3. persist model, metadata and reports --------------------------- #
    joblib.dump(fitted[best_name], models_dir / config.BEST_MODEL_FILENAME)

    metadata = {
        "schema_version": config.SCHEMA_VERSION,
        "trained_at": results["generated_at"],
        "selected_model": best_name,
        "best_params": results["models"][best_name]["best_params"],
        "selection_rationale": rationale,
        "decision_threshold": config.DECISION_THRESHOLD,
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "cv_folds": cv.get_n_splits(),
        "feature_columns": config.FEATURE_COLUMNS,
        "categorical_options": {
            c: sorted(map(str, X_train[c].dropna().unique())) for c in config.CATEGORICAL_FEATURES
        },
        "baselines": compute_baselines(X_train),
        "dataset": {
            "rows": int(len(df)),
            "churn_rate": float(y.mean()),
            "fingerprint": hashlib.sha256(
                pd.util.hash_pandas_object(df, index=False).values.tobytes()
            ).hexdigest()[:16],
        },
        "test_metrics": results["models"][best_name]["test"],
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (models_dir / config.METADATA_FILENAME).write_text(
        json.dumps(_to_builtin(metadata), indent=2), encoding="utf-8")
    (reports_dir / config.EVALUATION_FILENAME).write_text(
        json.dumps(results), encoding="utf-8")
    evaluate.build_comparison_table(results["models"]).to_csv(
        reports_dir / config.COMPARISON_FILENAME, index=False)
    evaluate.save_report_figures(results, reports_dir)

    report("Training complete.", 1.0)
    return results


def print_progress(message: str, fraction: float) -> None:
    """Simple console progress callback used by run.py."""
    print(f"[{fraction:5.0%}] {message}", flush=True)
