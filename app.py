"""Customer Churn Prediction System - Streamlit dashboard.

Run with:  python run.py          (recommended)
      or:  streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import config
from src import data_loader, predict, train
from src import visualizations as viz

st.set_page_config(page_title="Customer Churn Prediction System", page_icon="📉", layout="wide")

PAGES = [
    "Dashboard Overview",
    "Exploratory Data Analysis",
    "Model Performance",
    "Customer Churn Prediction",
    "Batch Prediction",
]

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
try:
    _NEW_WIDTH_API = tuple(int(p) for p in st.__version__.split(".")[:2]) >= (1, 50)
except ValueError:  # pragma: no cover
    _NEW_WIDTH_API = True


def _stretch() -> dict:
    """Full-width kwargs that work on both older and newer Streamlit versions."""
    return {"width": "stretch"} if _NEW_WIDTH_API else {"use_container_width": True}


def show_table(data, **kwargs) -> None:
    st.dataframe(data, hide_index=kwargs.pop("hide_index", True), **_stretch(), **kwargs)


def show_plot(fig, key: str) -> None:
    st.plotly_chart(fig, key=key, **_stretch())


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .hero {background: linear-gradient(135deg, #0B1F3A 0%, #1E6FD9 100%);
               padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;}
        .hero h1 {color: #FFFFFF; margin: 0; font-size: 2rem;}
        .hero p {color: #D6E4FA; margin: .3rem 0 0 0;}
        div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #D6E4FA;
               border-left: 5px solid #1E6FD9; border-radius: 10px; padding: .7rem 1rem;
               box-shadow: 0 1px 3px rgba(11,31,58,.08);}
        h2, h3 {color: #0B1F3A;}
        .badge {display:inline-block; padding:.35rem .9rem; border-radius:999px;
                color:#fff; font-weight:700; font-size:1.05rem;}
        .badge-Low {background:#2E9E6B;} .badge-Medium {background:#D99A12;}
        .badge-High {background:#D64545;}
        .note {background:#EAF1FB; border-left:4px solid #1E6FD9; padding:.7rem 1rem;
               border-radius:6px; color:#0B1F3A; font-size:.92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached loaders (nothing is re-read or retrained on every rerun)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading dataset...")
def load_default_dataset(path: str, modified: float):
    return data_loader.load_dataset_with_notes(path)


@st.cache_data(show_spinner="Reading uploaded file...")
def load_uploaded_dataset(content: bytes):
    import io
    return data_loader.load_dataset_with_notes(io.BytesIO(content))


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(modified: float):
    return predict.load_artifacts()


@st.cache_data(show_spinner=False)
def load_evaluation(modified: float) -> dict:
    return json.loads(config.EVALUATION_PATH.read_text(encoding="utf-8"))


def resolve_dataset():
    """Return (df, notes, source_label) or (None, [], None). Sidebar holds the uploader."""
    with st.sidebar:
        st.markdown("### Data source")
        uploaded = st.file_uploader("Upload your own Telco CSV (optional)", type=["csv"],
                                    key="dataset_upload")
    if uploaded is not None:
        try:
            df, notes = load_uploaded_dataset(uploaded.getvalue())
            return df, notes, f"uploaded file '{uploaded.name}'"
        except data_loader.DatasetValidationError as exc:
            st.sidebar.error(f"Uploaded file rejected: {exc}")
    if data_loader.dataset_exists():
        try:
            df, notes = load_default_dataset(str(config.DATASET_PATH),
                                             config.DATASET_PATH.stat().st_mtime)
            return df, notes, f"data/{config.DATASET_FILENAME}"
        except data_loader.DatasetValidationError as exc:
            st.error(f"The dataset in data/ is not valid: {exc}")
    return None, [], None


def run_training(df: pd.DataFrame) -> bool:
    """Train all models with a progress bar. Returns True on success."""
    bar = st.progress(0.0, text="Starting training...")

    def update(message: str, fraction: float) -> None:
        bar.progress(min(max(fraction, 0.0), 1.0), text=message)

    try:
        train.train_all_models(df, progress_callback=update)
    except Exception as exc:  # show a readable message instead of a stack trace
        bar.empty()
        st.error(f"Training failed: {exc}")
        return False
    bar.empty()
    st.cache_resource.clear()
    st.cache_data.clear()
    return True


def ensure_model(df) -> bool:
    """Make sure a compatible model exists; train one (once) if needed."""
    status, reason = predict.model_status()
    if status == "ok":
        return True
    if df is None:
        return False
    st.info(f"{reason} Training the models now - this runs only once and takes a few minutes.")
    return run_training(df)


def require_dataset(df) -> bool:
    if df is None:
        st.error(config.DATASET_MISSING_MESSAGE)
        st.markdown(f"Expected path: `data/{config.DATASET_FILENAME}`  \n"
                    "Alternatively, upload the CSV from the sidebar.")
        return False
    return True


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Dashboard Overview")
    churned = int((df[config.TARGET_COLUMN] == "Yes").sum())
    total = len(df)

    c = st.columns(3)
    c[0].metric("Total customers", f"{total:,}")
    c[1].metric("Churned customers", f"{churned:,}")
    c[2].metric("Retained customers", f"{total - churned:,}")
    c = st.columns(3)
    c[0].metric("Overall churn rate", f"{churned / total:.1%}")
    c[1].metric("Monthly revenue", f"${df['MonthlyCharges'].sum():,.0f}")
    c[2].metric("Average monthly charges", f"${df['MonthlyCharges'].mean():,.2f}")
    lost = df.loc[df[config.TARGET_COLUMN] == "Yes", "MonthlyCharges"].sum()
    st.caption(f"Monthly revenue belonging to churned customers: ${lost:,.0f} "
               f"({lost / df['MonthlyCharges'].sum():.1%} of total).")

    left, right = st.columns(2)
    with left:
        show_plot(viz.churn_distribution(df), "dash_dist")
        show_plot(viz.churn_rate_by_category(df, "PaymentMethod", "Churn rate by payment method"), "dash_pay")
        show_plot(viz.histogram_by_churn(df, "tenure", "Tenure distribution (months)"), "dash_tenure")
    with right:
        show_plot(viz.churn_rate_by_category(df, "Contract", "Churn rate by contract type"), "dash_contract")
        show_plot(viz.churn_rate_by_category(df, "InternetService", "Churn rate by internet service"), "dash_internet")
        show_plot(viz.histogram_by_churn(df, "MonthlyCharges", "Monthly charges distribution"), "dash_monthly")


def page_eda(df: pd.DataFrame, notes: list[str]) -> None:
    st.subheader("Exploratory Data Analysis")
    tabs = st.tabs(["Preview", "Structure & quality", "Statistics", "Numerical", "Categorical", "Correlation"])

    with tabs[0]:
        c = st.columns(3)
        c[0].metric("Rows", f"{df.shape[0]:,}")
        c[1].metric("Columns", df.shape[1])
        c[2].metric("Missing cells", int(df.isna().sum().sum()))
        n = st.slider("Rows to preview", 5, 100, 10, key="eda_rows")
        show_table(df.head(n))

    with tabs[1]:
        info = pd.DataFrame({
            "Column": df.columns, "Data type": df.dtypes.astype(str).values,
            "Non-null": df.notna().sum().values, "Missing": df.isna().sum().values,
            "Missing %": (df.isna().mean() * 100).round(2).values, "Unique values": df.nunique().values})
        show_table(info)
        st.markdown("**Data-quality actions applied while loading**")
        for line in notes or ["No data-quality problems were found."]:
            st.markdown(f"- {line}")

    with tabs[2]:
        st.markdown("**Numerical features**")
        show_table(df[config.NUMERIC_FEATURES].describe().T.reset_index().rename(columns={"index": "Feature"}))
        st.markdown("**Categorical features**")
        cat_cols = config.CATEGORICAL_FEATURES + [config.TARGET_COLUMN]
        show_table(df[cat_cols].describe().T.reset_index().rename(columns={"index": "Feature"}))

    with tabs[3]:
        col = st.selectbox("Numerical feature", config.NUMERIC_FEATURES, index=1, key="eda_num")
        a, b = st.columns(2)
        with a:
            show_plot(viz.histogram_by_churn(df, col), f"eda_hist_{col}")
        with b:
            show_plot(viz.box_by_churn(df, col), f"eda_box_{col}")
        summary = df.groupby(config.TARGET_COLUMN)[col].describe().reset_index()
        show_table(summary)

    with tabs[4]:
        col = st.selectbox("Categorical feature", config.CATEGORICAL_FEATURES, key="eda_cat")
        a, b = st.columns(2)
        with a:
            show_plot(viz.churn_rate_by_category(df, col), f"eda_cat_{col}")
        with b:
            table = pd.crosstab(df[col], df[config.TARGET_COLUMN]).reset_index()
            table["Churn rate %"] = (table.get("Yes", 0) / (table.get("Yes", 0) + table.get("No", 0)) * 100).round(1)
            show_table(table)

    with tabs[5]:
        show_plot(viz.correlation_heatmap(df), "eda_corr")
        note("Correlation only measures linear association between numeric columns. "
             "Categorical features are examined in the Categorical tab, and correlation "
             "does not imply that one variable causes another.")


def page_model_performance(df, evaluation: dict, metadata: dict) -> None:
    st.subheader("Model Performance")
    models, best = evaluation["models"], evaluation["selected_model"]

    st.success(f"Selected final model: **{best}**  -  trained {metadata['trained_at']} UTC")
    with st.expander("Why was this model selected?", expanded=True):
        st.write(evaluation["selection_rationale"])
    st.caption(f"Metrics below come from {evaluation['test_rows']:,} held-out test customers "
               f"({evaluation['test_churn_rate']:.1%} churn) that were never used for training or tuning. "
               f"Decision threshold: {evaluation['decision_threshold']:.2f}.")

    table = pd.DataFrame([{
        "Model": ("✅ " if n == best else "") + n,
        "Accuracy": m["test"]["accuracy"], "Precision": m["test"]["precision"],
        "Recall": m["test"]["recall"], "F1-score": m["test"]["f1"],
        "ROC-AUC": m["test"]["roc_auc"], "PR-AUC": m["test"]["pr_auc"],
        "CV ROC-AUC": m["cv"]["roc_auc"], "CV F1": m["cv"]["f1"],
        "Selection score": m["selection_score"]} for n, m in models.items()])
    fmt = {c: st.column_config.NumberColumn(format="%.3f") for c in table.columns if c != "Model"}
    show_table(table.sort_values("Selection score", ascending=False), column_config=fmt)

    tabs = st.tabs(["Comparison", "ROC curves", "Precision-Recall", "Confusion matrices",
                    "Feature importance", "Details"])
    with tabs[0]:
        show_plot(viz.model_comparison_bar(models), "perf_bar")
    with tabs[1]:
        show_plot(viz.roc_figure(models, best), "perf_roc")
    with tabs[2]:
        show_plot(viz.precision_recall_figure(models, best, evaluation["test_churn_rate"]), "perf_pr")
    with tabs[3]:
        cols = st.columns(2)
        for i, (name, m) in enumerate(models.items()):
            with cols[i % 2]:
                show_plot(viz.confusion_matrix_figure(m["confusion_matrix"], name), f"perf_cm_{i}")
        st.caption("Rows = actual class, columns = predicted class.")
    with tabs[4]:
        choice = st.selectbox("Model", list(models), index=list(models).index(best), key="perf_fi")
        imp = models[choice]["feature_importance"]
        show_plot(viz.feature_importance_bar(imp, title=f"Global feature importance - {choice}"), f"perf_fi_{choice}")
        note(f"<b>Global importance</b> ({imp['type']}). It summarises what the model relies on "
             "across all customers. It does not show why one particular customer is predicted "
             "to churn, and it does not prove causes.")
    with tabs[5]:
        choice = st.selectbox("Model", list(models), index=list(models).index(best), key="perf_detail")
        m = models[choice]
        st.markdown("**Best hyper-parameters (found with stratified cross-validation)**")
        st.json(m["best_params"])
        st.markdown("**Classification report (test set)**")
        report = pd.DataFrame(m["classification_report"]).T.round(3).reset_index().rename(columns={"index": ""})
        show_table(report)

    with st.expander("Retrain models"):
        st.write("Retrains all four models on the dataset currently loaded in the app and "
                 "replaces the saved model.")
        if st.button("Retrain now", key="retrain_btn"):
            if df is None:
                st.error(config.DATASET_MISSING_MESSAGE)
            elif run_training(df):
                st.rerun()


def _input_form(options: dict) -> dict | None:
    """Render the customer form; return the answers when submitted."""
    with st.form("customer_form"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", options["gender"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", options["Partner"])
            dependents = st.selectbox("Dependents", options["Dependents"])
            st.markdown("**Account**")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", options["Contract"])
            paperless = st.selectbox("Paperless Billing", options["PaperlessBilling"])
            payment = st.selectbox("Payment Method", options["PaymentMethod"])
        with b:
            st.markdown("**Phone & internet**")
            phone = st.selectbox("Phone Service", options["PhoneService"])
            lines = st.selectbox("Multiple Lines", options["MultipleLines"])
            internet = st.selectbox("Internet Service", options["InternetService"], index=1)
            security = st.selectbox("Online Security", options["OnlineSecurity"])
            backup = st.selectbox("Online Backup", options["OnlineBackup"])
        with c:
            st.markdown("**Add-ons & billing**")
            protection = st.selectbox("Device Protection", options["DeviceProtection"])
            support = st.selectbox("Tech Support", options["TechSupport"])
            tv = st.selectbox("Streaming TV", options["StreamingTV"])
            movies = st.selectbox("Streaming Movies", options["StreamingMovies"])
            monthly = st.number_input("Monthly Charges ($)", 0.0, 500.0, 70.0, 0.5)
            total = st.number_input("Total Charges ($)", 0.0, 100000.0, 840.0, 10.0,
                                    help="Roughly tenure x monthly charges for most customers.")
        submitted = st.form_submit_button("Predict churn")
    if not submitted:
        return None
    return {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total}


def page_prediction(pipeline, metadata: dict, evaluation: dict | None, bands: tuple) -> None:
    st.subheader("Customer Churn Prediction")
    low_max, high_min = bands
    submitted = _input_form(config.FEATURE_OPTIONS)
    if submitted is not None:
        inputs, notes = predict.harmonize_service_inputs(submitted)
        st.session_state["last_inputs"], st.session_state["last_notes"] = inputs, notes
    inputs = st.session_state.get("last_inputs")
    if inputs is None:
        st.info("Fill in the customer details above and press **Predict churn**.")
        return

    for message in st.session_state.get("last_notes", []):
        st.warning(message)

    customer = pd.DataFrame([inputs])[config.FEATURE_COLUMNS]
    result = predict.predict_dataframe(pipeline, customer, metadata["decision_threshold"], low_max, high_min).iloc[0]
    probability, risk = float(result["Churn_Probability"]), result["Risk_Category"]

    st.markdown("---")
    left, right = st.columns([1, 1])
    with left:
        verdict = "Likely to CHURN" if result["Predicted_Churn"] == "Yes" else "Likely to STAY"
        st.metric("Predicted status", verdict)
        st.metric("Churn probability", f"{probability:.1%}")
        st.markdown(f'Risk category: <span class="badge badge-{risk}">{risk} risk</span>', unsafe_allow_html=True)
        st.caption(f"Model-based risk bands: Low < {low_max:.0%}, Medium {low_max:.0%}-{high_min:.0%}, "
                   f"High >= {high_min:.0%}. They describe the model's estimate, not a guaranteed outcome. "
                   f"Model: {metadata['selected_model']}.")
    with right:
        show_plot(viz.probability_gauge(probability, low_max, high_min), "pred_gauge")

    st.markdown("### Why did the model predict this?")
    effects = predict.explain_customer(pipeline, customer, metadata["baselines"])
    if effects.empty:
        st.write("This customer matches the typical customer on every feature.")
    else:
        show_plot(viz.local_effect_bar(effects), "pred_local")
        show_table(effects)
    note("<b>Individual explanation:</b> each bar shows how the predicted probability changes when "
         "that feature is replaced by a typical value from the training data. Positive bars push this "
         "customer's predicted churn probability up. These are properties of the model's behaviour, "
         "not proof that a feature causes churn, and effects of related features overlap.")

    if evaluation is not None:
        with st.expander("Global feature importance of the model (all customers)"):
            imp = evaluation["models"][metadata["selected_model"]]["feature_importance"]
            show_plot(viz.feature_importance_bar(imp, 12), "pred_global")
            st.caption("Global importance describes the model overall, not this individual customer.")

    st.markdown("### Retention suggestions")
    st.caption("Rule-based ideas derived from the customer's inputs; they are business heuristics, not model output.")
    for title, text in predict.retention_suggestions(inputs, risk):
        st.markdown(f"- **{title}.** {text}")


def page_batch(pipeline, metadata: dict, bands: tuple) -> None:
    st.subheader("Batch Prediction")
    low_max, high_min = bands
    st.write("Upload a CSV with one customer per row. The `Churn` column is **not** required.")
    template = pd.DataFrame(columns=[config.ID_COLUMN] + config.FEATURE_COLUMNS).to_csv(index=False)
    st.download_button("Download CSV template (headers only)", template, "batch_template.csv", "text/csv")

    uploaded = st.file_uploader("Customer CSV", type=["csv"], key="batch_upload")
    if uploaded is None:
        return
    try:
        raw = data_loader.read_csv(uploaded)
        raw, warnings_list = predict.validate_prediction_input(raw, metadata)
        scored = predict.predict_dataframe(pipeline, raw, metadata["decision_threshold"], low_max, high_min)
    except (data_loader.DatasetValidationError, predict.PredictionInputError) as exc:
        st.error(str(exc))
        return

    if config.TARGET_COLUMN in raw.columns:
        st.info("The uploaded file contains a 'Churn' column; it was ignored.")
    for message in warnings_list:
        st.warning(message)

    c = st.columns(4)
    c[0].metric("Customers scored", f"{len(scored):,}")
    c[1].metric("Predicted to churn", f"{(scored['Predicted_Churn'] == 'Yes').sum():,}")
    c[2].metric("High-risk customers", f"{(scored['Risk_Category'] == 'High').sum():,}")
    c[3].metric("Average probability", f"{scored['Churn_Probability'].mean():.1%}")
    show_plot(viz.risk_distribution(scored), "batch_risk")
    show_table(scored, column_config={"Churn_Probability": st.column_config.ProgressColumn(
        "Churn_Probability", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("Download predictions (CSV)", scored.to_csv(index=False).encode("utf-8"),
                       "churn_predictions.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()
    st.markdown('<div class="hero"><h1>Customer Churn Prediction System</h1>'
                '<p>Predict which telecom customers are likely to leave, understand why, '
                'and compare machine-learning models.</p></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Go to", PAGES, label_visibility="collapsed")
        st.markdown("### Risk band thresholds")
        bands = st.slider("Low / Medium / High cut-offs (probability)", 0.0, 1.0,
                          (config.RISK_LOW_MAX, config.RISK_HIGH_MIN), 0.05)

    df, notes, source = resolve_dataset()
    model_ready = ensure_model(df)

    with st.sidebar:
        st.markdown("---")
        st.caption(f"Dataset: {source}" if source else "Dataset: not loaded")
        st.caption("Model: ready" if model_ready else "Model: not trained")

    if df is None and not model_ready:
        require_dataset(df)
        return

    if page in ("Dashboard Overview", "Exploratory Data Analysis"):
        if require_dataset(df):
            if page == "Dashboard Overview":
                page_dashboard(df)
            else:
                page_eda(df, notes)
        return

    if not model_ready:
        st.error("No trained model is available.")
        return
    pipeline, metadata = load_model(config.METADATA_PATH.stat().st_mtime)
    evaluation = load_evaluation(config.EVALUATION_PATH.stat().st_mtime) if config.EVALUATION_PATH.is_file() else None

    if page == "Model Performance":
        if evaluation is None:
            st.warning("Evaluation results were not found. Use the retrain option to regenerate them.")
        else:
            page_model_performance(df, evaluation, metadata)
    elif page == "Customer Churn Prediction":
        page_prediction(pipeline, metadata, evaluation, bands)
    else:
        page_batch(pipeline, metadata, bands)


main()
