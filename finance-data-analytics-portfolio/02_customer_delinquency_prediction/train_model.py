"""Baseline late-payment classifier trained on synthetic data only."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)
n = 1200
age = rng.integers(21, 61, n)
income = np.clip(rng.normal(52000, 19000, n), 12000, 150000).round().astype(int)
amount = np.clip(rng.normal(220000, 100000, n), 20000, 700000).round().astype(int)
tenure = rng.choice([6, 12, 18, 24, 36, 48, 60], n)
rate = np.clip(rng.normal(15.5, 3.8, n), 8, 29).round(2)
loan_type = rng.choice(["Personal", "Consumer Durable", "Two-Wheeler", "Education"], n)
employment = rng.choice(["Salaried", "Self-employed"], n)
logit = -3.0 + (amount / (income + 1)) * 0.08 + (employment == "Self-employed") * 0.4 + (rate > 20) * 0.35 + rng.normal(0, 0.55, n)
prob = 1 / (1 + np.exp(-logit))
late = rng.binomial(1, np.clip(prob, 0.02, 0.85))
df = pd.DataFrame({"age": age, "monthly_income_inr": income, "loan_amount_inr": amount, "tenure_months": tenure, "annual_interest_rate_pct": rate, "loan_type": loan_type, "employment_type": employment, "late_payment_flag": late})
df.to_csv(ROOT / "synthetic_customer_loans.csv", index=False)
target = "late_payment_flag"
features = [c for c in df.columns if c != target]
numeric = ["age", "monthly_income_inr", "loan_amount_inr", "tenure_months", "annual_interest_rate_pct"]
categorical = ["loan_type", "employment_type"]
preprocess = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
model = Pipeline([("preprocess", preprocess), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.25, random_state=42, stratify=df[target])
model.fit(X_train, y_train)
pred = model.predict(X_test); scores = model.predict_proba(X_test)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"Precision (late): {precision_score(y_test, pred, zero_division=0):.3f}")
print(f"Recall (late): {recall_score(y_test, pred, zero_division=0):.3f}")
print(f"F1 (late): {f1_score(y_test, pred, zero_division=0):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, scores):.3f}")
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["On-time", "Late"])
plt.title("Confusion Matrix — Synthetic Data"); plt.tight_layout(); plt.savefig(ROOT / "confusion_matrix.png", dpi=160)
pd.DataFrame({"actual_late_payment_flag": y_test, "predicted_late_payment_flag": pred, "predicted_probability": scores}).to_csv(ROOT / "sample_predictions.csv", index=False)
print("Saved synthetic_customer_loans.csv, confusion_matrix.png and sample_predictions.csv")
