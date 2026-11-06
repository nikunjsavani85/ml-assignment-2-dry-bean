"""Streamlit app to read the saved Dry Bean classifiers on an uploaded labelled CSV.
The models are loaded from model/*.joblib and are never retrained here
"""

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# This functions are cached to avoid reloading the model/metadata on every rerun of the app.
@st.cache_data
def load_metadata():
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))

# This function is cached to avoid reloading the model on every rerun of the app.
@st.cache_resource
def load_model(filename):
    # Only the trusted files shipped in the repository are ever loaded.
    return joblib.load(MODEL_DIR / filename)

# This function uses for validation of the uploaded CSV file and returns the feature frame and target series
def validate(data, features, target, class_names):
    if data.empty:
        raise ValueError("The uploaded file has no rows.")
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    expected = features + [target]
    missing = sorted(set(expected) - set(data.columns))
    extra = sorted(set(data.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"Missing columns: {missing or 'none'}. Unexpected columns: {extra or 'none'}.")
    if data[expected].isna().any().any():
        raise ValueError("Empty cells are not allowed in the feature or target columns.")
    # Validate that all feature values are numeric and finite, and that all target values are known classes.
    try:
        X = data[features].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError(f"All feature values must be numeric ({error}).") from None
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("All feature values must be finite numbers.")
    y = data[target].astype(str).str.strip()
    unknown = sorted(set(y) - set(class_names))
    # The uploaded CSV may contain unknown class labels, which is not allowed
    if unknown:
        raise ValueError(f"Unknown class labels in '{target}': {unknown}.")
    return X, y


def evaluate(model, X, y):
    class_names = model.classes_
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, labels=class_names, average="macro", zero_division=0
    )
    # read AUC only if all classes are present in the uploaded data,otherwise it will raise an error
    auc = np.nan
    if set(y.unique()) == set(class_names):
        auc = roc_auc_score(y, probabilities, labels=class_names, multi_class="ovr", average="macro")
    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": matthews_corrcoef(y, predictions),
    }
    return metrics, predictions

# Streamlit app starts here using the above functions to load metadata, validate uploaded CSV, load model, and evaluate it.
st.set_page_config(page_title="Dry Bean classifier evaluation", page_icon=":material/eco:", layout="wide")

metadata = load_metadata()
features = metadata["features"]
target = metadata["target"]
class_names = metadata["classes"]
model_files = {name: spec["filename"] for name, spec in metadata["models"].items()}

st.title("Dry Bean classifier evaluation")
st.caption(
    "Upload a labelled test CSV to evaluate five pre-trained classifiers on the UCI Dry Bean dataset. "
    "The saved models are only used for prediction; uploaded data is never used for training."
)

# Sidebar for file upload and model selection
with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Labelled test CSV", type="csv")
    st.download_button(
        "Download sample test_data.csv",
        data=SAMPLE_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    model_name = st.selectbox("Model", list(model_files), index=list(model_files).index(metadata["selected_model"]))
    with st.expander("Expected CSV format"):
        st.write(f"{len(features)} numeric feature columns plus the target column `{target}`:")
        st.code(", ".join(features + [target]), language=None)
        st.write("Valid classes: " + ", ".join(class_names))

# Main app logic
if uploaded is None:
    st.info("Upload a labelled CSV in the sidebar to see the results. You can download the sample file there.", icon=":material/upload_file:")
    st.stop()

try:
    data = pd.read_csv(uploaded)
    X, y = validate(data, features, target, class_names)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"Could not use the uploaded file: {error}", icon=":material/error:")
    st.stop()

st.success(f"Loaded {len(X):,} rows with {len(features)} features.", icon=":material/check_circle:")

model = load_model(model_files[model_name])
metrics, predictions = evaluate(model, X, y)
# Display metrics, confusion matrix, and classification report
st.subheader(f"Metrics: {model_name}")
with st.container(horizontal=True):
    for name in METRIC_NAMES:
        value = metrics[name]
        st.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}", border=True)
# Warn the user if AUC is unavailable due to missing classes in the uploaded data
if np.isnan(metrics["AUC"]):
    st.warning("AUC is unavailable because the uploaded data does not contain every trained class.")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    matrix = confusion_matrix(y, predictions, labels=model.classes_)
    cells = pd.DataFrame(matrix, index=model.classes_, columns=model.classes_).stack().reset_index()
    cells.columns = ["Actual", "Predicted", "Count"]
    base = alt.Chart(cells).encode(
        x=alt.X("Predicted:N", sort=list(model.classes_)),
        y=alt.Y("Actual:N", sort=list(model.classes_)),
    )
    # The confusion matrix is displayed as a heatmap with counts and tooltips.
    heatmap = base.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")), tooltip=["Actual", "Predicted", "Count"])
    labels = base.mark_text().encode(
        text="Count:Q",
        color=alt.condition(alt.datum.Count > matrix.max() / 2, alt.value("white"), alt.value("black")),
    )
    st.altair_chart(heatmap + labels, height=420, alt=f"Confusion matrix for {model_name}")
with right:
    st.subheader("Classification report")
    report = classification_report(
        y, predictions, labels=model.classes_, target_names=list(model.classes_), output_dict=True, zero_division=0
    )
    report_table = pd.DataFrame(report).T
    report_table.loc["accuracy"] = [np.nan, np.nan, report["accuracy"], report_table.loc["macro avg", "support"]]
    st.dataframe(report_table.round(4), height=420, alt=f"Classification report for {model_name}")

st.subheader("All models on this file")
rows = []
for name, filename in model_files.items():
    model_metrics, _ = evaluate(load_model(filename), X, y)
    rows.append({"Model": name, **model_metrics})
comparison = pd.DataFrame(rows).set_index("Model").round(4)
st.dataframe(comparison.style.highlight_max(axis=0, color="#cde4f7"), alt="Metrics for all five models")
