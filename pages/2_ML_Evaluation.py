"""Read-only presentation of the frozen historical ADAS ML evaluation."""

from hashlib import sha256
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "M05_D11_NESTED_EVALUATION_20261025.json"

EXPECTED_SHA = (
    "beb6949ce2f8cf183b9042672b7333cc29de49d732c4539d818be8f38ddab3f8"
)

st.set_page_config(
    page_title="ADAS | ML Evaluation",
    page_icon="??",
    layout="wide",
)

st.title("ML Evaluation")
st.caption(
    "Frozen nested-evaluation results from historical 2024 "
    "two-wheeler IMU recordings."
)

st.warning(
    "These results measure agreement with historical "
    "operator-generated annotations, not independently "
    "validated road-safety ground truth. "
    "They do not establish production ADAS accuracy."
)


@st.cache_data(show_spinner=False)
def load_frozen_report():
    raw = REPORT.read_bytes()

    if sha256(raw).hexdigest() != EXPECTED_SHA:
        raise ValueError("Frozen report integrity verification failed")

    return json.loads(raw)


try:
    data = load_frozen_report()
except (OSError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"Cannot load verified evaluation evidence: {exc}")
    st.stop()

if data.get("status") != "PASS":
    st.error("Frozen evaluation status is not PASS.")
    st.stop()

comparisons = data.get("comparisons", {})
populations = data.get("population_sizes", {})
classes = data.get("class_order", [])

required = {
    f"{population}/{representation}"
    for population in ("primary", "sensitivity")
    for representation in (
        "accelerometer_only",
        "accelerometer_plus_relative_angles",
        "accelerometer_plus_raw_angles",
    )
}

if not required.issubset(comparisons):
    st.error("Required frozen evaluation comparisons are missing.")
    st.stop()

with st.sidebar:
    st.header("Evaluation controls")
    population = st.selectbox(
        "Evaluation population",
        ("primary", "sensitivity"),
    )

st.info(
    f"Population: {population} | "
    f"Frozen evaluation windows: {populations[population]:,} | "
    "No new training or predictions are performed."
)

rows = []

for key in sorted(required):
    result = comparisons[key]
    evaluation = result["report"]

    rows.append({
        "Population": result["population"],
        "Representation": result["representation"],
        "Macro F1": float(evaluation["macro_f1"]),
        "Balanced accuracy": float(evaluation["balanced_accuracy"]),
        "Evaluated windows": int(evaluation["observed_count"]),
        "Status": evaluation["status"],
    })

frame = pd.DataFrame(rows)
selected = frame[frame["Population"] == population].copy()

st.subheader("Feature representation comparison")

display = selected[
    ["Representation", "Macro F1", "Balanced accuracy"]
].set_index("Representation")

st.bar_chart(display)
st.dataframe(
    selected,
    hide_index=True,
)

best_row = selected.loc[selected["Macro F1"].idxmax()]

st.caption(
    "Highest observed macro F1 in this comparison: "
    f"{best_row['Representation']} "
    f"({best_row['Macro F1']:.3f}). "
    "This is a descriptive comparison, not a deployment approval."
)

st.subheader("Detailed evaluation")

representation = st.selectbox(
    "Feature representation",
    (
        "accelerometer_only",
        "accelerometer_plus_relative_angles",
        "accelerometer_plus_raw_angles",
    ),
)

key = f"{population}/{representation}"
detail = comparisons[key]["report"]

a, b, c = st.columns(3)
a.metric("Macro F1", f"{detail['macro_f1']:.3f}")
b.metric(
    "Balanced accuracy",
    f"{detail['balanced_accuracy']:.3f}",
)
c.metric(
    "Evaluated windows",
    f"{detail['observed_count']:,}",
)

st.subheader("Confusion matrix")

matrix = detail.get("confusion_matrix", [])

if (
    isinstance(matrix, list)
    and len(matrix) == len(classes)
    and all(
        isinstance(row, list) and len(row) == len(classes)
        for row in matrix
    )
):
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"Actual: {label}" for label in classes],
        columns=[f"Predicted: {label}" for label in classes],
    )
    st.dataframe(matrix_frame)
    st.caption(
        "Rows represent historical operator annotations; "
        "columns represent frozen out-of-fold predictions."
    )
else:
    st.info("Confusion matrix cannot be displayed as a square table.")
    with st.expander("Raw confusion matrix"):
        st.json(matrix)

st.subheader("Class-wise evaluation")

class_metrics = detail.get("class_metrics", [])
if class_metrics:
    st.dataframe(
        pd.DataFrame(class_metrics),
        hide_index=True,
    )
else:
    st.info("No class-wise results included.")

st.subheader("Outer-fold evaluation")

fold_metrics = detail.get("fold_metrics", [])
if fold_metrics:
    st.dataframe(
        pd.DataFrame(fold_metrics),
        hide_index=True,
    )
else:
    st.info("No fold-level results included.")

with st.expander("Methodology and evidence provenance"):
    st.write("Evaluation status:", data["status"])
    st.write("Purpose:", data.get("purpose"))
    st.write("Evidence date (UTC):", data.get("date_utc"))
    st.write("Evaluation population:", population)
    st.write("Comparison:", key)
    st.code(EXPECTED_SHA, language=None)
    st.caption(
        "SHA-256 identifies the frozen JSON evidence. "
        "The dashboard does not modify this file."
    )

st.divider()

st.caption(
    "Historical offline ML evaluation only. "
    "No live inference, hardware validation, independently "
    "verified driving ground truth, or certified safety behavior."
)
