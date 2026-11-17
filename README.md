# Customer Churn Prediction System

A complete supervised-machine-learning project that predicts whether a telecom customer is likely to
leave (**churn**), explains the prediction, compares four algorithms and presents everything in an
interactive Streamlit dashboard.

---

## 1. Quick start (Windows 11 / PowerShell / Python 3.12)

1. Extract the ZIP and download the dataset from Kaggle:
   <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
2. Place the file here (exact name):

   ```
   data/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```
3. Open the folder in VS Code, open a terminal (PowerShell) and run:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py
```

`run.py` checks the dataset, trains the models the first time (a few minutes; progress is printed),
saves them, and opens the dashboard at <http://localhost:8501>.
On every later start the saved model is reused - **nothing is retrained** unless you ask.

Other commands:

```powershell
python run.py --retrain        # force retraining
python run.py --train-only     # train and save, do not start the app
python run.py --port 8502      # different port
streamlit run app.py           # start the app directly (it trains automatically if needed)
pip install -r requirements-dev.txt; python -m pytest -q    # run the automated tests
```

If the dataset is missing the app shows:
*"Dataset not found. Please download the IBM Telco Customer Churn CSV and place it inside the data/ directory."*
You can also upload a Telco CSV from the app sidebar instead of using the `data/` folder.

---

## 2. Introduction and problem statement

Acquiring a new customer usually costs far more than keeping an existing one, so telecom companies
want to know **which customers are about to leave** while there is still time to act. Given a
customer's demographics, services, contract and billing information, the task is to predict
whether the customer will churn and to estimate how likely that is.

## 3. Objectives

- Build a reproducible binary-classification pipeline on the IBM Telco Customer Churn dataset.
- Compare Logistic Regression, Decision Tree, Random Forest and XGBoost fairly.
- Choose the final model with suitable metrics (not accuracy alone).
- Output churn probability, a Low/Medium/High risk band and the factors behind each prediction.
- Provide an interactive dashboard with EDA, model comparison, single and batch prediction.

## 4. Dataset

IBM Telco Customer Churn (Kaggle, ~7,043 rows, 21 columns). Target: `Churn` (Yes/No).

| Group | Columns |
|---|---|
| Identifier (never used as a feature) | `customerID` |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Subscription | `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Contract & billing | `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |

Known quirk: `TotalCharges` contains blank strings for customers with tenure 0. The project converts the
column to numbers, sets those new-customer blanks to 0 and imputes anything else inside the pipeline.
The project never generates synthetic customer data; every statistic and prediction comes from the CSV you provide.

## 5. Technologies

Python 3.12 · scikit-learn · XGBoost · pandas · NumPy · Matplotlib · Seaborn · Plotly · Streamlit · Joblib.
Optional libraries were **deliberately left out**: imbalanced-learn (class weights are enough) and SHAP
(see section 9).

## 6. Machine-learning algorithms

| Algorithm | Idea | Imbalance handling |
|---|---|---|
| Logistic Regression | Linear model estimating the log-odds of churn; easy to interpret via coefficients. | `class_weight="balanced"` |
| Decision Tree | Learns a sequence of if/else splits; very readable but can overfit (depth is tuned). | `class_weight="balanced"` |
| Random Forest | Many decorrelated trees voting together; robust and low-variance. | `class_weight="balanced_subsample"` |
| XGBoost | Gradient-boosted trees where each tree corrects the previous ones. | `scale_pos_weight` = negatives / positives (from training data) |

## 7. System architecture

```
customer-churn-prediction/
├── data/                  put the Kaggle CSV here
├── models/                best_model.joblib, model_metadata.json (created by training)
├── reports/               evaluation_results.json, model_comparison.csv, PNG figures (created by training)
├── src/
│   ├── data_loader.py     load, validate, clean the CSV; friendly errors
│   ├── preprocessing.py   stateless fixes + ColumnTransformer (impute / scale / one-hot)
│   ├── train.py           split, tune (CV), evaluate, select, save
│   ├── evaluate.py        metrics, curves, feature importance, report figures
│   ├── predict.py         load model, score customers, explain, risk bands, suggestions
│   └── visualizations.py  Plotly charts for the dashboard
├── tests/                 pytest checks (test-only schema fixture, not app data)
├── app.py                 Streamlit dashboard
├── config.py              paths, columns, thresholds, search settings
├── run.py                 one-command launcher
└── requirements.txt
```

Flow: `CSV → data_loader (validate/clean) → stratified split → Pipeline[ColumnTransformer → model]
→ RandomizedSearchCV (stratified 5-fold CV) → test-set evaluation → model selection → joblib + reports → Streamlit`.

## 8. Application features

| Page | What it provides |
|---|---|
| Dashboard Overview | Total/churned/retained customers, churn %, monthly revenue, average charges; churn by contract, payment method, internet service; tenure and charge distributions. |
| Exploratory Data Analysis | Preview, dimensions, data types, missing values, descriptive statistics, numeric and categorical analysis, correlation heat-map, data-cleaning log. |
| Model Performance | Comparison table, ROC and precision-recall curves, confusion matrices, feature importance, best hyper-parameters, classification reports, the selected model and why, retrain button. |
| Customer Churn Prediction | Form with all 19 inputs → predicted status, probability, gauge, risk band, individual explanation, retention suggestions. |
| Batch Prediction | Upload CSV (no `Churn` column needed), validation, probabilities and risk bands, download results. |

The sidebar lets you upload your own CSV and adjust the risk-band thresholds
(defaults: Low < 30 %, Medium 30-60 %, High ≥ 60 %; edit `config.py` to change the defaults).

## 9. Evaluation methodology

- **Split first:** a stratified 80/20 train-test split with `random_state=42`, done before anything is fitted.
- **No leakage:** imputation, scaling and encoding live *inside* a scikit-learn `Pipeline`, so they are fitted on
  training data only (and on training folds only during cross-validation). No resampling (SMOTE) is used.
- **Tuning:** `RandomizedSearchCV` with stratified 5-fold CV on the training set, refitted on ROC-AUC.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, classification report, on the untouched test set.
- **Model selection:** the final model has the highest *cross-validated* selection score
  (50 % ROC-AUC + 50 % F1 of the churn class), computed on training folds only. Accuracy is not used because
  only about a quarter of customers churn. Test-set numbers are reported but not used to choose the model.
- **Saved artifacts:** the complete pipeline (`models/best_model.joblib`), metadata (library versions, parameters,
  feature list, category lists, "typical customer" values) and evaluation reports.

### Explainability

- **Global importance** (Model Performance page): Logistic Regression coefficients or tree importances - what the model relies on overall.
- **Individual explanation** (Prediction page): for each feature, the customer's value is swapped for a typical
  training value and the change in predicted probability is shown. It works for *every* model and avoids
  fragile dependencies, which is why SHAP was not used.
- These explain **the model's behaviour, not causes of churn**; associated factors are never presented as causal.
- Retention suggestions are simple business rules based on the inputs, labelled as heuristics.

## 10. Limitations

- The dataset is a single, anonymised snapshot of one fictional telecom company; results may not transfer to other companies or time periods.
- Churn labels are historical; the model finds associations, not causes, and customer behaviour may drift.
- Risk bands are model-based probability ranges, not guaranteed outcomes. Probabilities from class-weighted models are
  not perfectly calibrated.
- The individual explanation is an approximation; effects of correlated features overlap and are not additive.
- No cost-sensitive threshold tuning: the default decision threshold is 0.50.
- `SeniorCitizen`, `gender` and `Partner/Dependents` are demographic attributes; before any real-world use, review
  fairness and legal implications.

## 11. Future improvements

- Probability calibration and threshold optimisation using retention-campaign cost vs. benefit.
- SHAP explanations, partial-dependence plots and fairness analysis.
- Experiment tracking (MLflow), scheduled retraining and drift monitoring.
- Survival analysis for "when" a customer churns; REST API (FastAPI) and Docker deployment.

---
*Educational project. Predictions are estimates that support - not replace - business judgement.*
