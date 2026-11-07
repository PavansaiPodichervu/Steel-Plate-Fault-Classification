
import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Load training data
train_data = pd.read_csv("data/train_data.csv")

X_train = train_data.drop(columns=["Fault_Type"])
y_train = train_data["Fault_Type"]

# Create model directory
os.makedirs("model", exist_ok=True)

# Define models
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    "decision_tree": DecisionTreeClassifier(
        random_state=42
    ),

    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "naive_bayes": GaussianNB(),

    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", CalibratedClassifierCV(
            estimator=SVC(),
            method="sigmoid",
            cv=3
        ))
    ])
}

# Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)

    path = f"model/{name}.joblib"
    joblib.dump(model, path)

    print(f"{name} trained and saved.")

# Save feature names for validation
joblib.dump(
    list(X_train.columns),
    "model/feature_names.joblib"
)

print("\nAll five models trained successfully.")
