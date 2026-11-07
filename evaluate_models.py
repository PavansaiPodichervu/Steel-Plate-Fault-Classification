
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Load test data
test_data = pd.read_csv("data/test_data.csv")

X_test = test_data.drop(columns=["Fault_Type"])
y_test = test_data["Fault_Type"]

# Load saved feature names
feature_names = joblib.load("model/feature_names.joblib")
X_test = X_test[feature_names]

results = []

model_names = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "svm"
]

# Evaluate all models
for name in model_names:

    model = joblib.load(f"model/{name}.joblib")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(
            y_test,
            y_prob,
            labels=model.classes_,
            multi_class="ovr",
            average="macro"
        ),
        "Precision": precision_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    results.append(metrics)

# Save comparison
results_df = pd.DataFrame(results)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

print("\nResults saved to model_comparison.csv")
