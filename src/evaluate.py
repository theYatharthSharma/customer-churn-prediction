"""Model evaluation: metrics, curves, feature importance and report figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved to files
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config

CLASS_NAMES = ["Retained", "Churned"]
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def evaluate_model(pipeline, X_test, y_test, threshold: float = config.DECISION_THRESHOLD) -> dict:
    """Score a fitted pipeline on the held-out test set.

    Returns metrics, confusion matrix, classification report and the points of
    the ROC and precision-recall curves (all computed from real predictions).
    """
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_test, proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, proba)

    return {
        "test": {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
        },
        # rows = actual [Retained, Churned], columns = predicted [Retained, Churned]
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision_curve.tolist(), "recall": recall_curve.tolist()},
    }


def extract_feature_importance(pipeline, top_n: int = 30) -> dict:
    """GLOBAL feature importance of a fitted pipeline.

    * Logistic Regression -> signed coefficients (log-odds per scaled / one-hot unit)
    * Tree models         -> impurity / gain based ``feature_importances_``
    These describe what the model relies on overall - not what causes churn.
    """
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    clf = pipeline.named_steps["classifier"]

    if hasattr(clf, "coef_"):
        signed = np.asarray(clf.coef_).ravel()
        importance = np.abs(signed)
        kind = "Logistic Regression coefficient (positive = raises predicted churn odds)"
    elif hasattr(clf, "feature_importances_"):
        importance = np.asarray(clf.feature_importances_, dtype=float)
        signed = None
        kind = "Tree-based feature importance (share of the model's splits/gain)"
    else:  # pragma: no cover - defensive
        return {"type": "Not available for this model", "items": []}

    order = np.argsort(importance)[::-1][:top_n]
    items = []
    for i in order:
        item = {"feature": str(names[i]), "importance": float(importance[i])}
        if signed is not None:
            item["coefficient"] = float(signed[i])
        items.append(item)
    return {"type": kind, "items": items}


def build_comparison_table(model_results: dict) -> "pd.DataFrame":  # noqa: F821
    """One row per model with test metrics, CV scores and the selection score."""
    import pandas as pd

    rows = []
    for name, res in model_results.items():
        row = {"Model": name}
        row.update({
            "Accuracy": res["test"]["accuracy"],
            "Precision": res["test"]["precision"],
            "Recall": res["test"]["recall"],
            "F1-score": res["test"]["f1"],
            "ROC-AUC": res["test"]["roc_auc"],
            "PR-AUC": res["test"]["pr_auc"],
            "CV ROC-AUC": res["cv"]["roc_auc"],
            "CV F1": res["cv"]["f1"],
            "Selection score": res["selection_score"],
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Selection score", ascending=False).reset_index(drop=True)


def save_report_figures(results: dict, reports_dir: Path) -> list[Path]:
    """Save static PNG figures (matplotlib / seaborn) into the reports folder."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models = results["models"]
    best = results["selected_model"]
    sns.set_theme(style="whitegrid")
    saved: list[Path] = []

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, res in models.items():
        ax.plot(res["roc_curve"]["fpr"], res["roc_curve"]["tpr"],
                lw=3 if name == best else 1.6,
                label=f"{name} (AUC = {res['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax.legend(loc="lower right")
    saved.append(_save(fig, reports_dir / "roc_curves.png"))

    # Precision-recall curves
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, res in models.items():
        ax.plot(res["pr_curve"]["recall"], res["pr_curve"]["precision"],
                lw=3 if name == best else 1.6,
                label=f"{name} (AP = {res['test']['pr_auc']:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)")
    ax.legend(loc="upper right")
    saved.append(_save(fig, reports_dir / "precision_recall_curves.png"))

    # Confusion matrices
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, (name, res) in zip(axes, models.items()):
        sns.heatmap(np.array(res["confusion_matrix"]), annot=True, fmt="d", cmap="Blues",
                    cbar=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set(title=name, xlabel="Predicted", ylabel="Actual")
    saved.append(_save(fig, reports_dir / "confusion_matrices.png"))

    # Model comparison
    table = build_comparison_table(models).set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    table.plot(kind="bar", ax=ax, colormap="Blues_r", edgecolor="black", width=0.8)
    ax.set(ylim=(0, 1), ylabel="Score", title="Model comparison (test set)")
    ax.tick_params(axis="x", rotation=15)
    saved.append(_save(fig, reports_dir / "model_comparison.png"))

    # Feature importance of the selected model
    items = models[best]["feature_importance"]["items"][:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([i["feature"] for i in items], [i["importance"] for i in items], color="#1E6FD9")
    ax.set(title=f"Top features - {best} (global importance)", xlabel="Importance")
    saved.append(_save(fig, reports_dir / "feature_importance.png"))
    return saved


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
