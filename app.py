import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Wine Type Classification",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Type Classification System")

st.write(
    "A machine learning application that classifies wine "
    "as Red Wine or White Wine using physicochemical properties."
)

# --------------------------------------------------
# Feature columns
# --------------------------------------------------

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality"
]

# --------------------------------------------------
# Available models
# --------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors": "model/k-nearest_neighbors.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

# --------------------------------------------------
# Load selected model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# Model selection
# --------------------------------------------------

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model",
    list(MODEL_FILES.keys())
)

model_path = MODEL_FILES[selected_model]

if os.path.exists(model_path):
    model = load_model(model_path)
    st.sidebar.success("Model loaded successfully")
else:
    st.error(f"Model file not found: {model_path}")
    st.stop()

st.subheader(f"Selected Model: {selected_model}")

# --------------------------------------------------
# Upload dataset
# --------------------------------------------------

st.header("Upload Wine Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset Preview")
        st.dataframe(df.head())

        # Check required columns
        missing_columns = [
            col for col in FEATURES if col not in df.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {missing_columns}"
            )
            st.stop()

        # Extract input features
        X = df[FEATURES].copy()

        # Convert features to numeric
        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isnull().any().any():
            st.error(
                "The dataset contains missing or non-numeric "
                "values in the required feature columns."
            )
            st.stop()

        # --------------------------------------------------
        # Make predictions
        # --------------------------------------------------

        predictions = model.predict(X)

        result_df = df.copy()

        result_df["Predicted_Wine_Type"] = [
            "Red Wine" if value == 0 else "White Wine"
            for value in predictions
        ]

        st.subheader("Classification Results")

        st.dataframe(result_df.head(20))

        # Prediction distribution
        prediction_counts = pd.Series(
            result_df["Predicted_Wine_Type"]
        ).value_counts()

        st.subheader("Prediction Distribution")
        st.bar_chart(prediction_counts)

        # Download predictions
        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="wine_predictions.csv",
            mime="text/csv"
        )

        # --------------------------------------------------
        # Evaluate model if actual target is available
        # --------------------------------------------------

        if "Wine_Type" in df.columns:

            st.header("Model Evaluation Metrics")

            y_true = pd.to_numeric(
                df["Wine_Type"], errors="coerce"
            )

            if y_true.isnull().any():
                st.error("Wine_Type contains invalid values.")
                st.stop()

            if not set(y_true.unique()).issubset({0, 1}):
                st.error("Wine_Type must contain only 0 and 1.")
                st.stop()

            accuracy = accuracy_score(y_true, predictions)
            precision = precision_score(
                y_true, predictions, zero_division=0
            )
            recall = recall_score(
                y_true, predictions, zero_division=0
            )
            f1 = f1_score(
                y_true, predictions, zero_division=0
            )
            mcc = matthews_corrcoef(y_true, predictions)

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("Precision", f"{precision:.4f}")
            col3.metric("Recall", f"{recall:.4f}")

            col4, col5 = st.columns(2)

            col4.metric("F1 Score", f"{f1:.4f}")
            col5.metric("MCC", f"{mcc:.4f}")

            # ROC-AUC
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)[:, 1]
                auc = roc_auc_score(y_true, probabilities)
                st.metric("ROC-AUC", f"{auc:.4f}")

            # Confusion matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_true, predictions, labels=[0, 1]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Red", "Actual White"],
                columns=["Predicted Red", "Predicted White"]
            )

            st.dataframe(cm_df)

        else:
            st.info(
                "Upload a CSV containing the Wine_Type column "
                "to calculate evaluation metrics."
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --------------------------------------------------
# Model comparison
# --------------------------------------------------

st.header("Comparison of All Five Models")

if os.path.exists("model_comparison.csv"):

    comparison_df = pd.read_csv("model_comparison.csv")

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.bar_chart(
        comparison_df.set_index("Model")["Accuracy"]
    )

else:
    st.warning("Model comparison file not found.")