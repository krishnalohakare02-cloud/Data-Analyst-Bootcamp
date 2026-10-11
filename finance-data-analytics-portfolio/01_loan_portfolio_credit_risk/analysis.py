"""Explore a generated synthetic loan portfolio."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)
n = 1200
income = np.clip(rng.normal(52000, 19000, n), 12000, 150000).round().astype(int)
amount = np.clip(rng.normal(220000, 100000, n), 20000, 700000).round().astype(int)
loan_type = rng.choice(["Personal", "Consumer Durable", "Two-Wheeler", "Education"], n)
approved_prob = np.clip(0.45 + income / 250000 - amount / 1200000, 0.08, 0.95)
approved = rng.binomial(1, approved_prob)
late_prob = np.clip(0.08 + amount / 2500000 + (income < 30000) * 0.12, 0.02, 0.65)
late = rng.binomial(1, late_prob)
df = pd.DataFrame({"customer_id": [f"S{i:05d}" for i in range(1, n + 1)], "monthly_income_inr": income, "loan_amount_inr": amount, "loan_type": loan_type, "approved": approved, "late_payment_flag": late})
df.to_csv(ROOT / "synthetic_loan_portfolio.csv", index=False)
summary = df.groupby("loan_type").agg(applications=("customer_id", "count"), approval_rate_pct=("approved", lambda x: round(100*x.mean(), 2)), late_payment_rate_pct=("late_payment_flag", lambda x: round(100*x.mean(), 2)), avg_loan_amount_inr=("loan_amount_inr", "mean")).reset_index()
summary["avg_loan_amount_inr"] = summary["avg_loan_amount_inr"].round(0).astype(int)
summary.to_csv(ROOT / "loan_type_summary.csv", index=False)
print("Overall applications:", len(df))
print("Approval rate (%):", round(100*df.approved.mean(), 2))
print("Late-payment flag rate (%):", round(100*df.late_payment_flag.mean(), 2))
print(summary.to_string(index=False))
ax = summary.plot(x="loan_type", y="late_payment_rate_pct", kind="bar", legend=False, title="Synthetic Late-Payment Rate by Loan Type")
ax.set_xlabel("Loan type"); ax.set_ylabel("Late-payment rate (%)")
plt.tight_layout(); plt.savefig(ROOT / "late_payment_by_loan_type.png", dpi=160)
print("Saved synthetic_loan_portfolio.csv, loan_type_summary.csv and chart.")
