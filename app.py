
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Steel Plate Fault Classification",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Steel Plate Fault Classification")
st.caption("Machine Learning Model Evaluation Dashboard")

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_DIR = "model"

MODEL_NAMES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm"
}

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    models = {}

    for display_name, filename in MODEL_NAMES.items():

        model_path = os.path.join(
            MODEL_DIR,
            filename + ".joblib"
        )

        if os.path.exists(model_path):
            models[display_name] = joblib.load(model_path)

    feature_path = os.path.join(
        MODEL_DIR,
        "feature_names.joblib"
    )

    if os.path.exists(feature_path):
        features = joblib.load(feature_path)
    else:
        features = None

    return models, features


models, feature_names = load_models()

if not models:
    st.error("No trained models found in the model folder.")
    st.info("Run: python run_project.py")
    st.stop()

if feature_names is None:
    st.error("Feature information is missing.")
    st.info("Run: python run_project.py")
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Model Configuration")

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Select a trained model and upload test samples "
    "to evaluate its classification performance."
)

st.sidebar.markdown(
    "**Dataset:** Steel Plates Faults"
)

st.sidebar.markdown(
    "**Models:** 5"
)

# =========================================================
# MODEL COMPARISON
# =========================================================

st.header("📊 Model Comparison")

comparison_path = "model_comparison.csv"

if os.path.exists(comparison_path):

    comparison = pd.read_csv(comparison_path)

    display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "K-Nearest Neighbors",
        "naive_bayes": "Naive Bayes",
        "svm": "Support Vector Machine"
    }

    comparison["Model"] = comparison["Model"].replace(
        display_names
    )

    # Highlight best model by macro F1
    best_model = comparison.loc[
        comparison["F1 Score"].idxmax(),
        "Model"
    ]

    st.success(
        f"Best model based on Macro F1 Score: {best_model}"
    )

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "MCC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Performance Comparison")

    chart_metric = st.selectbox(
        "Select metric for comparison",
        [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]
    )

    chart_data = comparison.set_index("Model")[
        [chart_metric]
    ]

    st.bar_chart(chart_data)

else:

    st.warning(
        "Model comparison is unavailable. "
        "Run evaluate_models.py first."
    )

# =========================================================
# DATASET UPLOAD
# =========================================================

st.divider()

st.header("📂 Test Dataset Evaluation")

st.write(
    "Upload a CSV containing the input features and "
    "actual Fault_Type labels."
)

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_uploaded_data(test_data, model):

    X_test = test_data[feature_names]
    y_test = test_data["Fault_Type"]

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    classes = model.classes_

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Multiclass AUC
    if set(classes).issubset(set(y_test)):

        auc = roc_auc_score(
            y_test,
            y_prob,
            labels=classes,
            multi_class="ovr",
            average="macro"
        )

    else:
        auc = None

    return {
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "y_pred": y_pred,
        "y_test": y_test,
        "classes": classes
    }

# =========================================================
# PROCESS UPLOADED CSV
# =========================================================

if uploaded_file is not None:

    try:

        test_data = pd.read_csv(uploaded_file)

        # ---------------------------------------------
        # VALIDATION
        # ---------------------------------------------

        if test_data.empty:

            st.error("Uploaded CSV is empty.")

        elif "Fault_Type" not in test_data.columns:

            st.error(
                "CSV must contain the Fault_Type column "
                "to calculate evaluation metrics."
            )

        else:

            missing_features = [
                feature for feature in feature_names
                if feature not in test_data.columns
            ]

            if missing_features:

                st.error(
                    "Missing required features: "
                    + ", ".join(missing_features)
                )

            elif test_data["Fault_Type"].isnull().any():

                st.error(
                    "Fault_Type contains missing values."
                )

            elif test_data[feature_names].isnull().any().any():

                st.error(
                    "Test data contains missing feature values."
                )

            else:

                # -----------------------------------------
                # UPLOAD SUCCESS
                # -----------------------------------------

                st.success(
                    f"Dataset uploaded successfully! "
                    f"Samples: {len(test_data)}"
                )

                with st.expander("Preview Uploaded Dataset"):

                    st.dataframe(
                        test_data.head(),
                        use_container_width=True
                    )

                # -----------------------------------------
                # MODEL EVALUATION
                # -----------------------------------------

                if st.button(
                    "Evaluate Selected Model",
                    type="primary",
                    use_container_width=True
                ):

                    model = models[selected_model_name]

                    results = evaluate_uploaded_data(
                        test_data,
                        model
                    )

                    # -------------------------------------
                    # METRICS
                    # -------------------------------------

                    st.divider()

                    st.header(
                        f"📈 Evaluation Results: "
                        f"{selected_model_name}"
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Accuracy",
                        f"{results['accuracy']:.4f}"
                    )

                    col2.metric(
                        "AUC",
                        (
                            f"{results['auc']:.4f}"
                            if results["auc"] is not None
                            else "N/A"
                        )
                    )

                    col3.metric(
                        "Precision",
                        f"{results['precision']:.4f}"
                    )

                    col4, col5, col6 = st.columns(3)

                    col4.metric(
                        "Recall",
                        f"{results['recall']:.4f}"
                    )

                    col5.metric(
                        "F1 Score",
                        f"{results['f1']:.4f}"
                    )

                    col6.metric(
                        "MCC",
                        f"{results['mcc']:.4f}"
                    )

                    # -------------------------------------
                    # CONFUSION MATRIX
                    # -------------------------------------

                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(
                        results["y_test"],
                        results["y_pred"],
                        labels=results["classes"]
                    )

                    fig, ax = plt.subplots(
                        figsize=(10, 7)
                    )

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=results["classes"],
                        yticklabels=results["classes"],
                        ax=ax
                    )

                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("Actual Label")
                    ax.set_title("Confusion Matrix")

                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    plt.tight_layout()

                    st.pyplot(fig)

                    plt.close(fig)

                    # -------------------------------------
                    # CLASSIFICATION REPORT
                    # -------------------------------------

                    st.subheader("Classification Report")

                    report = classification_report(
                        results["y_test"],
                        results["y_pred"],
                        zero_division=0,
                        output_dict=True
                    )

                    report_df = pd.DataFrame(
                        report
                    ).transpose()

                    st.dataframe(
                        report_df.round(4),
                        use_container_width=True
                    )

                    # -------------------------------------
                    # DOWNLOAD PREDICTIONS
                    # -------------------------------------

                    st.subheader("Download Predictions")

                    predictions = test_data.copy()

                    predictions["Predicted_Fault"] = (
                        results["y_pred"]
                    )

                    csv_data = predictions.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_data,
                        file_name="steel_fault_predictions.csv",
                        mime="text/csv"
                    )

    except Exception as error:

        st.error(
            f"Unable to process uploaded CSV: {error}"
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Steel Plate Fault Classification | "
    "Machine Learning Assignment | "
    "UCI Steel Plates Faults Dataset"
)
