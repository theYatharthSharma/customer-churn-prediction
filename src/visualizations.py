"""Plotly figure builders used by the Streamlit dashboard.

Every function receives real data (the loaded dataset or saved evaluation
results) and returns a Plotly figure - nothing is hard-coded.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config

NAVY, BLUE, LIGHT, ORANGE = "#0B1F3A", "#1E6FD9", "#EAF1FB", "#F26B38"
CHURN_COLORS = {"No": BLUE, "Yes": ORANGE}
RISK_COLORS = {"Low": "#2E9E6B", "Medium": "#F2B138", "High": "#D64545"}
MODEL_COLORS = ["#1E6FD9", "#7FB3F5", "#0B1F3A", "#F26B38"]


def _style(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": NAVY}},
        height=height, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    fig.update_xaxes(gridcolor="#E6ECF5")
    fig.update_yaxes(gridcolor="#E6ECF5")
    return fig


# --------------------------------------------------------------------------- #
# Dataset charts
# --------------------------------------------------------------------------- #
def churn_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df[config.TARGET_COLUMN].value_counts().reindex(["No", "Yes"]).fillna(0)
    fig = go.Figure(go.Pie(
        labels=["Retained", "Churned"], values=counts.values, hole=0.55,
        marker=dict(colors=[BLUE, ORANGE]), textinfo="label+percent"))
    return _style(fig, "Churn distribution")


def churn_rate_by_category(df: pd.DataFrame, column: str, title: str | None = None) -> go.Figure:
    """Bar chart of churn rate (%) per category; hover shows customer counts."""
    grouped = (df.groupby(column)[config.TARGET_COLUMN]
               .agg(customers="size", churned=lambda s: int((s == "Yes").sum())).reset_index())
    grouped["churn_rate"] = grouped["churned"] / grouped["customers"] * 100
    grouped = grouped.sort_values("churn_rate", ascending=False)
    fig = go.Figure(go.Bar(
        x=grouped[column].astype(str), y=grouped["churn_rate"], marker_color=BLUE,
        text=[f"{v:.1f}%" for v in grouped["churn_rate"]], textposition="outside",
        customdata=grouped[["customers", "churned"]].values,
        hovertemplate="%{x}<br>Churn rate: %{y:.1f}%<br>Customers: %{customdata[0]}"
                      "<br>Churned: %{customdata[1]}<extra></extra>"))
    fig.update_yaxes(title="Churn rate (%)", range=[0, max(grouped["churn_rate"].max() * 1.2, 5)])
    return _style(fig, title or f"Churn rate by {column}")


def histogram_by_churn(df: pd.DataFrame, column: str, title: str | None = None, bins: int = 30) -> go.Figure:
    fig = px.histogram(df, x=column, color=config.TARGET_COLUMN, nbins=bins, barmode="overlay",
                       opacity=0.7, color_discrete_map=CHURN_COLORS)
    return _style(fig, title or f"{column} distribution")


def box_by_churn(df: pd.DataFrame, column: str) -> go.Figure:
    fig = px.box(df, x=config.TARGET_COLUMN, y=column, color=config.TARGET_COLUMN,
                 color_discrete_map=CHURN_COLORS)
    fig.update_layout(showlegend=False)
    return _style(fig, f"{column} by churn status")


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric = df[config.NUMERIC_FEATURES].copy()
    numeric["Churn (1 = Yes)"] = (df[config.TARGET_COLUMN] == "Yes").astype(int)
    corr = numeric.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    aspect="auto")
    return _style(fig, "Correlation of numeric features (Pearson)", height=420)


# --------------------------------------------------------------------------- #
# Model charts
# --------------------------------------------------------------------------- #
def model_comparison_bar(models: dict) -> go.Figure:
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
               ("f1", "F1-score"), ("roc_auc", "ROC-AUC")]
    fig = go.Figure()
    for (key, label), color in zip(metrics, ["#9DBFEA", "#7FB3F5", "#1E6FD9", NAVY, ORANGE]):
        fig.add_bar(name=label, x=list(models), y=[m["test"][key] for m in models.values()],
                    marker_color=color)
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1])
    return _style(fig, "Test-set metrics by model", height=400)


def roc_figure(models: dict, selected: str) -> go.Figure:
    fig = go.Figure()
    for (name, res), color in zip(models.items(), MODEL_COLORS):
        fig.add_scatter(x=res["roc_curve"]["fpr"], y=res["roc_curve"]["tpr"], mode="lines",
                        name=f"{name} (AUC {res['test']['roc_auc']:.3f})",
                        line=dict(color=color, width=4 if name == selected else 2))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                    line=dict(color="grey", dash="dash"))
    fig.update_xaxes(title="False positive rate")
    fig.update_yaxes(title="True positive rate")
    return _style(fig, "ROC curves (test set)", height=450)


def precision_recall_figure(models: dict, selected: str, positive_rate: float) -> go.Figure:
    fig = go.Figure()
    for (name, res), color in zip(models.items(), MODEL_COLORS):
        fig.add_scatter(x=res["pr_curve"]["recall"], y=res["pr_curve"]["precision"], mode="lines",
                        name=f"{name} (AP {res['test']['pr_auc']:.3f})",
                        line=dict(color=color, width=4 if name == selected else 2))
    fig.add_hline(y=positive_rate, line_dash="dash", line_color="grey",
                  annotation_text="No-skill baseline (churn rate)")
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return _style(fig, "Precision-recall curves (test set)", height=450)


def confusion_matrix_figure(cm: list, title: str) -> go.Figure:
    labels = ["Retained", "Churned"]
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Customers"))
    fig.update_coloraxes(showscale=False)
    return _style(fig, title, height=330)


def feature_importance_bar(importance: dict, top_n: int = 15, title: str = "") -> go.Figure:
    items = importance["items"][:top_n][::-1]
    signed = bool(items) and "coefficient" in items[0]
    if signed:
        values = [i["coefficient"] for i in items]
        colors = [ORANGE if v > 0 else BLUE for v in values]
    else:
        values = [i["importance"] for i in items]
        colors = BLUE
    fig = go.Figure(go.Bar(x=values, y=[i["feature"] for i in items], orientation="h",
                           marker_color=colors))
    fig.update_xaxes(title="Coefficient (orange = raises churn odds)" if signed else "Importance")
    return _style(fig, title or "Global feature importance", height=max(360, 26 * len(items) + 120))


# --------------------------------------------------------------------------- #
# Prediction charts
# --------------------------------------------------------------------------- #
def probability_gauge(probability: float, low_max: float, high_min: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, number={"suffix": "%"},
        title={"text": "Churn probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": NAVY},
               "steps": [{"range": [0, low_max * 100], "color": "#BFE6D2"},
                         {"range": [low_max * 100, high_min * 100], "color": "#FBE3B0"},
                         {"range": [high_min * 100, 100], "color": "#F5B8B8"}]}))
    return _style(fig, "", height=280)


def local_effect_bar(table: pd.DataFrame) -> go.Figure:
    data = table.iloc[::-1]
    values = data["Effect on probability (pp)"]
    fig = go.Figure(go.Bar(
        x=values, y=[f"{f} = {v}" for f, v in zip(data["Feature"], data["Customer value"])],
        orientation="h", marker_color=[ORANGE if v > 0 else BLUE for v in values]))
    fig.update_xaxes(title="Change in churn probability vs. a typical value (percentage points)")
    return _style(fig, "What pushes this prediction up or down", height=max(300, 40 * len(data) + 120))


def risk_distribution(scored: pd.DataFrame) -> go.Figure:
    counts = scored["Risk_Category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                           marker_color=[RISK_COLORS[r] for r in counts.index],
                           text=counts.values.astype(int), textposition="outside"))
    fig.update_yaxes(title="Customers")
    return _style(fig, "Customers by risk band")
