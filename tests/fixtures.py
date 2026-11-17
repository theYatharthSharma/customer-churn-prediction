"""TEST-ONLY helper: builds a small random frame that has the Telco *schema*.

It exists so unit tests can run without the real Kaggle file.  It is never used
by the application, never written to data/, and its results mean nothing
about real customers.
"""
import numpy as np
import pandas as pd


def make_schema_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(19, 115, n), 2)
    yn = lambda p=0.5: rng.choice(["Yes", "No"], n, p=[p, 1 - p])  # noqa: E731

    def addon():
        vals = yn(0.4).astype(object)
        vals[internet == "No"] = "No internet service"
        return vals

    lines = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n))
    total = np.round(tenure * monthly, 2).astype(object)
    total[tenure == 0] = " "  # the real file has blank strings here

    logit = (-0.5 + 1.2 * (contract == "Month-to-month") - 0.03 * tenure
             + 0.01 * monthly + 0.4 * (internet == "Fiber optic"))
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    return pd.DataFrame({
        "customerID": [f"TEST-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": yn(), "Dependents": yn(0.3), "tenure": tenure,
        "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
        "OnlineSecurity": addon(), "OnlineBackup": addon(), "DeviceProtection": addon(),
        "TechSupport": addon(), "StreamingTV": addon(), "StreamingMovies": addon(),
        "Contract": contract, "PaperlessBilling": yn(0.6),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": monthly, "TotalCharges": total, "Churn": churn,
    })
