"""Offline analysis of historical ADAS recordings.

Uses existing tested backend functions. No hardware or model inference.
"""

from collections import Counter
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from adas.data.loader import load_recording
from adas.signal.segmentation import segment_records
from adas.signal.adaptive_segmentation import split_timing_stalls
from adas.features.windowing import build_windows
from adas.features.extraction import extract_labeled_windows
from adas.features.d4_representations import extract_d4_representations

DATASET = ROOT / "Dataset" / "Own Bike"
DEMO_DATASET = ROOT / "demo_data"

st.set_page_config(
    page_title="ADAS | Recording Analysis",
    page_icon="??",
    layout="wide",
)

st.title("Recording Analysis")
st.caption(
    "Offline segmentation and feature extraction "
    "from historical or synthetic IMU recordings."
)

st.warning(
    "Historical annotations originate from operator inputs; "
    "synthetic labels are artificial. Feature extraction "
    "does not verify driving behavior or road safety."
)

files = []
for directory in (DEMO_DATASET, DATASET):
    if directory.is_dir():
        files.extend(
            p for p in directory.iterdir()
            if p.is_file()
            and p.name.upper().startswith("MAIN")
            and p.suffix.upper() == ".CSV"
            and p.stem[4:].isdigit()
        )

files.sort(
    key=lambda p: (
        0 if p.parent == DEMO_DATASET else 1,
        int(p.stem[4:]),
    )
)

if not files:
    st.error("No demo or historical MAIN*.CSV recordings found.")
    st.stop()

with st.sidebar:
    st.header("Analysis configuration")

    default_index = next(
        (i for i, p in enumerate(files) if p.name == "MAIN32.CSV"),
        0,
    )

    selected = st.selectbox(
        "Recording",
        files,
        index=default_index,
        format_func=lambda p: (
            ("SYNTHETIC DEMO ? " if p.parent == DEMO_DATASET
             else "HISTORICAL 2024 ? ") + p.name
        ),
    )

    gap_ms = st.number_input(
        "Segment gap threshold (ms)",
        min_value=1.0,
        value=1000.0,
    )

    ratio = st.number_input(
        "Timing-stall interval ratio",
        min_value=1.0,
        value=3.0,
    )

    window_ms = st.number_input(
        "Window duration (ms)",
        min_value=100.0,
        value=2000.0,
        step=100.0,
    )

    stride_ms = st.number_input(
        "Window stride (ms)",
        min_value=100.0,
        value=1000.0,
        step=100.0,
    )

    min_samples = st.number_input(
        "Minimum samples per window",
        min_value=2,
        value=20,
        step=1,
    )

st.info(
    "Windows remain inside individual timing stretches. "
    "Only pure-labeled windows are eligible for supervised "
    "feature extraction."
)

if selected.parent == DEMO_DATASET:
    st.warning(
        "SYNTHETIC DEMONSTRATION: This recording was generated "
        "for software testing. Its annotations do not represent "
        "real-world motorcycle events."
    )

if st.button("Analyze recording", type="primary"):
    try:
        records, file_report = load_recording(selected)

        segments = segment_records(
            records,
            gap_threshold_ms=gap_ms,
        )

        stretches = split_timing_stalls(
            segments,
            interval_ratio=ratio,
        )

        windows = build_windows(
            stretches,
            window_ms=window_ms,
            stride_ms=stride_ms,
            min_samples=int(min_samples),
        )

        features = extract_labeled_windows(windows)

    except (ValueError, OSError, RuntimeError) as exc:
        st.error(f"Analysis could not complete: {exc}")
        st.stop()

    st.subheader(f"Results ? {selected.name}")

    a, b, c, d = st.columns(4)
    a.metric("Valid samples", f"{file_report.valid:,}")
    b.metric("Segments", len(segments))
    c.metric("Timing stretches", len(stretches))
    d.metric("Feature windows", len(windows))

    st.subheader("Window eligibility")

    statuses = Counter(window.status for window in windows)

    eligibility = pd.DataFrame(
        [
            {"Status": name, "Windows": statuses.get(name, 0)}
            for name in (
                "PURE_LABELED",
                "UNANNOTATED",
                "MIXED_WITH_UNANNOTATED",
                "MIXED_CLASSES",
                "TOO_FEW_SAMPLES",
            )
        ]
    ).set_index("Status")

    st.bar_chart(eligibility)
    st.metric("Extracted labeled feature rows", len(features))

    st.caption(
        "Window counts describe annotation consistency, "
        "not independently verified driving behavior."
    )

    st.subheader("Segment boundary analysis")

    boundary_counts = Counter(
        segment.start_reason for segment in segments
    )

    if boundary_counts:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Boundary": key, "Segments": count}
                    for key, count in sorted(boundary_counts.items())
                ]
            ),
            hide_index=True,
        )
    else:
        st.info("No nonempty segments in this recording.")

    st.subheader("Extracted statistical features")

    if features:
        feature_table = pd.DataFrame(
            [
                {
                    "Source": row.source_file,
                    "Segment": row.parent_segment_id,
                    "Stretch": row.stretch_id,
                    "Window": row.window_index,
                    "Start (ms)": row.start_ms,
                    "End (ms)": row.end_ms,
                    "Samples": row.sample_count,
                    "Operator label": row.label,
                    **{
                        f"Feature {i + 1:02d}": value
                        for i, value in enumerate(row.values)
                    },
                }
                for row in features[:50]
            ]
        )

        st.dataframe(
            feature_table,
            hide_index=True,
        )

        st.caption(
            "Showing up to 50 extracted rows. Each row has "
            "30 raw-unit statistics from the six sensor channels."
        )

        st.subheader("D4 feature representations")

        pure_windows = [
            window for window in windows
            if window.status == "PURE_LABELED"
        ]

        representation_rows = []

        for window in pure_windows[:5]:
            for representation in extract_d4_representations(window):
                representation_rows.append(
                    {
                        "Representation": representation.representation,
                        "Operator label": representation.label,
                        "Feature count": len(representation.values),
                        "Recording group": representation.group,
                    }
                )

        if representation_rows:
            st.dataframe(
                pd.DataFrame(representation_rows),
                hide_index=True,
            )

        st.caption(
            "D4 table is a preview of up to five eligible windows. "
            "It does not train or evaluate a model."
        )
    else:
        st.info(
            "No eligible pure-labeled feature rows were extracted "
            "under the selected settings."
        )

    st.success("Offline recording analysis completed.")

st.divider()
st.caption(
    "Research and portfolio demonstration only. "
    "No live hardware, independently validated driving labels, "
    "or production safety decisions."
)
