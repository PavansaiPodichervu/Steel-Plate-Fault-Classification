
import os
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# Create data directory
os.makedirs("data", exist_ok=True)

# Download Steel Plates Faults dataset
steel_plates_faults = fetch_ucirepo(id=198)

X = steel_plates_faults.data.features.copy()
y = steel_plates_faults.data.targets.copy()

# Convert one-hot targets into a single class label
if y.sum(axis=1).eq(1).all():
    y["Fault_Type"] = y.idxmax(axis=1)
else:
    raise ValueError("Unexpected target encoding in dataset.")

# Combine features and target
dataset = X.copy()
dataset["Fault_Type"] = y["Fault_Type"]

# Save complete dataset
dataset.to_csv("data/steel_plates_faults.csv", index=False)

# Split features and target
X = dataset.drop(columns=["Fault_Type"])
y = dataset["Fault_Type"]

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save training data
train_data = X_train.copy()
train_data["Fault_Type"] = y_train
train_data.to_csv("data/train_data.csv", index=False)

# Save testing data
test_data = X_test.copy()
test_data["Fault_Type"] = y_test
test_data.to_csv("data/test_data.csv", index=False)

print("Preprocessing completed successfully!")
print("Complete dataset:", dataset.shape)
print("Training dataset:", train_data.shape)
print("Testing dataset:", test_data.shape)
print("\nClass distribution:")
print(dataset["Fault_Type"].value_counts())
