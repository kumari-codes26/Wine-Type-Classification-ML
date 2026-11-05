
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

# 1. Create output folders
os.makedirs("model", exist_ok=True)

# 2. Load combined dataset
df = pd.read_csv("wine_quality_combined.csv")

X = df.drop(columns=["Wine_Type"])
y = df["Wine_Type"]

# 3. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Define five classification models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": GaussianNB(),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}

results = []

# 5. Train and evaluate each model
for name, model in models.items():

    print(f"\nTraining: {name}")

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "ROC_AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "MCC": mcc,
        "Training_Time_Seconds": training_time
    })

    # Save trained model
    filename = name.lower().replace(" ", "_")
    joblib.dump(model, f"model/{filename}.joblib")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(
        cm,
        index=["Actual_Red", "Actual_White"],
        columns=["Predicted_Red", "Predicted_White"]
    ).to_csv(f"model/{filename}_confusion_matrix.csv")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

# 6. Save model comparison results
comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    by="F1_Score",
    ascending=False
)

comparison.to_csv("model_comparison.csv", index=False)

# 7. Save test data for Streamlit application
test_data = X_test.copy()
test_data["Wine_Type"] = y_test
test_data.to_csv("test_data.csv", index=False)

# 8. Display final results
print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

print("\nBest Model:")
print(comparison.iloc[0]["Model"])

print("\nAll models trained and saved successfully!")
