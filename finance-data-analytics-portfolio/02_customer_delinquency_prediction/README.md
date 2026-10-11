# Customer Repayment & Delinquency Prediction

**Tools:** Python, pandas, scikit-learn, logistic regression

## Objective
Train a baseline classifier to predict a synthetic late-payment flag from sample loan attributes.

## Run
```bash
python -m pip install -r requirements.txt
python 02_customer_delinquency_prediction/train_model.py
```

The script generates synthetic data, splits it into train/test sets, preprocesses numeric and categorical columns, trains logistic regression, and reports accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

## Important limitation
This is a learning demonstration using artificial data. Metrics are not evidence of real-world lending performance and must not be used to approve or reject applicants. Real credit models require robust validation, explainability, privacy controls, and fairness testing.
