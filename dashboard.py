"""Historical two-wheeler IMU data exploration dashboard.

Offline visualization only. No live hardware connection or
independently validated road-safety inference is claimed.
"""

from collections import Counter
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend" / "src"))

from adas.data.loader import load_recording
from adas.reporting.quality import build_quality_report

DATASET = ROOT / "Dataset" / "Own Bike"
DEMO_DATASET = ROOT / "demo_data"

LABELS = (
    "normal_driving",
    "rash_driving",
    "normal_turn",
    "rash_turn",
    "NAN",
)

CHANNELS = {
    "AccX": "acc_x",
    "AccY": "acc_y",
    "AccZ": "acc_z",
    "GyroX": "gyro_x",
    "GyroY": "gyro_y",
    "GyroZ": "gyro_z",
}

st.set_page_config(
    page_title="ADAS | Historical IMU Explorer",
    page_icon="🏍️",
    layout="wide",
)


@st.cache_data(show_spinner="Loading historical recording...")
def load_selected(path_string: str, file_size: int, modified_ns: int):
    """Cache a recording by its known path and file metadata."""
    return load_recording(Path(path_string))


def recording_options():
    """Discover synthetic and optional historical recordings."""
    result = []
    for directory in (DEMO_DATASET, DATASET):
        if directory.is_dir():
            result.extend(
                path for path in directory.iterdir()
                if path.is_file()
                and path.name.upper().startswith("MAIN")
                and path.suffix.upper() == ".CSV"
                and path.stem[4:].isdigit()
            )
    return sorted(
        result,
        key=lambda path: (
            0 if path.parent == DEMO_DATASET else 1,
            int(path.stem[4:]),
        ),
    )


st.title("ADAS ? IMU Recording Explorer")
st.caption(
    "Explore historical two-wheeler IMU recordings "
    "or a clearly labeled synthetic demonstration."
)

st.info(
    "Offline software demonstration only. Historical labels "
    "are operator-generated and lack independent ground-truth "
    "validation; synthetic labels are artificial. "
    "This is not a live or certified ADAS system."
)

files = recording_options()

if not files:
    st.error(
        "No recordings found. The public demo requires "
        "demo_data/MAIN900.CSV; optional historical recordings "
        "belong in Dataset/Own Bike."
    )
    st.stop()

with st.sidebar:
    st.header("Recording controls")

    selected = st.selectbox(
        "Choose recording",
        files,
        format_func=lambda path: (
            ("SYNTHETIC DEMO ? " if path.parent == DEMO_DATASET
             else "HISTORICAL 2024 ? ") + path.name
        ),
    )

    threshold = st.number_input(
        "Timestamp gap observation threshold (ms)",
        min_value=1.0,
        value=1000.0,
        step=100.0,
    )

    maximum_points = st.slider(
        "Maximum plotted points",
        min_value=300,
        max_value=5000,
        value=2000,
        step=100,
    )

    st.caption(
        "Sampling for visualization does not modify the "
        "underlying dataset or quality calculations."
    )

if selected.parent == DEMO_DATASET:
    st.warning(
        "SYNTHETIC DEMONSTRATION: These measurements and "
        "labels are artificial. They are not 2024 field data "
        "or evidence of real driving behavior."
    )

file_stat = selected.stat()
records, file_report = load_selected(
    str(selected),
    file_stat.st_size,
    file_stat.st_mtime_ns,
)

quality = build_quality_report(
    records,
    [file_report],
    gap_threshold_ms=threshold,
)

summary = quality["summary"]
timestamps = quality["files"][0]["timestamps"]

st.subheader(f"Recording: {selected.name}")

a, b, c, d = st.columns(4)
a.metric("Valid measurements", f"{file_report.valid:,}")
b.metric("Header rows", file_report.headers)
c.metric("Rejected rows", file_report.invalid)
d.metric("Annotated measurements", f"{summary['labeled']:,}")

if not records:
    st.warning(
        "This recording contains no accepted sensor measurements. "
        "Its file-quality information is still shown below."
    )
else:
    stride = max(1, (len(records) + maximum_points - 1) // maximum_points)

    sampled = records[::stride]

    frame = pd.DataFrame(
        [
            {
                "Sample index": index,
                "Timestamp (s)": record.timestamp_ms / 1000.0,
                "AccX": record.acc_x,
                "AccY": record.acc_y,
                "AccZ": record.acc_z,
                "GyroX": record.gyro_x,
                "GyroY": record.gyro_y,
                "GyroZ": record.gyro_z,
                "Annotation": record.annotation,
                "Source line": record.source_line,
            }
            for index, record in enumerate(records[::stride])
        ]
    )

    frame["Sample index"] = [
        index * stride for index in range(len(sampled))
    ]

    st.subheader("Accelerometer measurements")
    st.caption(
        "Nominal acceleration units: g. Charts use source-order "
        "sample indices rather than assuming evenly spaced time."
    )

    st.line_chart(
        frame,
        x="Sample index",
        y=["AccX", "AccY", "AccZ"],
    )

    st.subheader("Gyroscope-derived angles")
    st.caption(
        "Historical GyroX/Y/Z are gyroscope-derived integrated "
        "angles, nominally in degrees, not angular rates. "
        "Synthetic measurements are artificially generated."
    )

    st.line_chart(
        frame,
        x="Sample index",
        y=["GyroX", "GyroY", "GyroZ"],
    )

    st.subheader("Annotation distribution")

    counts = Counter(record.annotation for record in records)

    label_frame = pd.DataFrame(
        {
            "Annotation": LABELS,
            "Measurements": [counts.get(label, 0) for label in LABELS],
        }
    ).set_index("Annotation")

    st.bar_chart(label_frame)

    st.caption(
        "'NAN' represents unannotated measurements, "
        "not normal driving."
    )

    st.subheader("Sampled recording data")
    st.dataframe(frame, width="stretch", hide_index=True)

st.subheader("Timestamp quality")

q1, q2, q3 = st.columns(3)
q1.metric("Duplicate timestamps", timestamps["duplicate"])
q2.metric("Decreasing timestamps", timestamps["decreasing"])
q3.metric(
    "Intervals above threshold",
    timestamps["gaps_over_threshold"],
)

st.caption(quality["interpretation"])

with st.expander("Detailed recording quality report"):
    st.json(quality)

st.divider()
if selected.parent == DEMO_DATASET:
    st.caption(
        "DATA PROVENANCE: Synthetic software-test recording. "
        "Measurements and labels are artificial, "
        "not historical field observations."
    )
else:
    st.caption(
        "DATA PROVENANCE: Historical 2024 Arduino/MPU6050 "
        "recording. Hardware re-verification is unavailable. "
        "Operator labels are not independently validated."
    )
