"""Streamlit app: evaluate the saved Dry Bean classifiers on an uploaded labelled CSV.

The models are loaded from model/*.joblib and are never retrained here.
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


@st.cache_data
def load_metadata():
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@st.cache_resource
def load_model(filename):
    # Only the trusted files shipped in the repository are ever loaded.
    return joblib.load(MODEL_DIR / filename)


def validate(data, features, target, class_names):
    """Return (feature frame, target series) or raise ValueError with a readable message."""
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
    try:
        X = data[features].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError(f"All feature values must be numeric ({error}).") from None
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("All feature values must be finite numbers.")
    y = data[target].astype(str).str.strip()
    unknown = sorted(set(y) - set(class_names))
    if unknown:
        raise ValueError(f"Unknown class labels in '{target}': {unknown}.")
    return X, y


def evaluate(model, X, y):
    """Same conventions as the notebook: macro P/R/F1, macro one-vs-rest AUC, multiclass MCC."""
    class_names = model.classes_
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, labels=class_names, average="macro", zero_division=0
    )
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

st.subheader(f"Metrics: {model_name}")
with st.container(horizontal=True):
    for name in METRIC_NAMES:
        value = metrics[name]
        st.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}", border=True)
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
