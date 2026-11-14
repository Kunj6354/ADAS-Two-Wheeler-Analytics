# ADAS for Two-Wheeler Drivers

**Offline IMU analytics and machine-learning evaluation — a software-only research portfolio project.**

This repository contains a reproducible Streamlit application for exploring historical two-wheeler inertial measurements, analyzing recording quality, extracting statistical features, and reviewing frozen machine-learning evaluation results.

The application is **not a live driver-assistance system**, collision detector, deployed prediction service, or safety-certified product.

## What the application does

| Page | Function |
|---|---|
| IMU Recording Explorer | Explore acceleration, gyro-derived angles, annotations, and timestamp quality |
| Recording Analysis | Segment recordings, identify timing stretches, construct feature windows, and inspect statistical representations |
| ML Evaluation | Review frozen nested-evaluation results, feature representations, class-level metrics, and sensitivity analysis |

The application supports two distinct data sources:

- **Synthetic demonstration:** Included in the repository as `demo_data/MAIN900.CSV`. It contains 600 artificial measurements and three artificial annotation categories.
- **Historical recordings:** Original 2024 two-wheeler IMU recordings, used locally for historical analysis. These recordings are not included in the public release candidate.

Synthetic demonstration data is never presented as independently observed road behavior.

## Quick start

Requires Python 3.12.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location).Path "backend\src")
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

### Linux / macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/backend/src"
.venv/bin/python -m streamlit run dashboard.py
```

The bundled synthetic recording is available automatically. No Arduino device, wearable, or original historical CSV is required to explore the demonstration.

## Reproducible software tests

From the repository root, with the virtual environment installed and `PYTHONPATH` pointing to `backend/src`:

```bash
python -m pytest backend/tests -q
```

The verified local baseline on Python 3.12 is:

- **455 passing tests**, including three Streamlit application integration tests
- **60 passing subtests**
- Three Streamlit pages passing isolated application acceptance without historical recordings
- Clean Python 3.12 dependency installation and application testing

These checks verify software behavior; they do not validate real-world road-safety effectiveness.

## Synthetic demonstration acceptance

The generated demonstration recording provides:

| Measurement | Result |
|---|---:|
| Synthetic samples | 600 |
| Segments | 1 |
| Timing stretches | 1 |
| Feature windows | 28 |
| Extracted eligible feature rows | 26 |

Each eligible feature row contains 30 raw-unit statistical features derived from six sensor channels.

## Machine-learning evaluation

The ML Evaluation page displays historical, frozen evaluation evidence. It does not train or deploy a prediction model when the application runs.

Selected results for the primary evaluation population:

| Feature representation | Macro F1 | Balanced accuracy |
|---|---:|---:|
| Accelerometer only | 0.352 | 0.351 |
| Accelerometer + relative angles | 0.399 | 0.401 |
| Accelerometer + raw angles | 0.411 | 0.403 |

The primary population contains 3,478 evaluated windows. A separate sensitivity population contains 3,578 evaluated windows.

These metrics reflect the specified retrospective evaluation protocol, not independently verified detection accuracy on real-world driving hazards.

## Data and interpretation limitations

- Historical annotation categories came from operator-button inputs and are **not independently verified driving ground truth**.
- `NAN` identifies unannotated measurements; it must not be interpreted as normal driving.
- GyroX/Y/Z are stored as gyro-derived integrated angle channels, nominally in degrees, not raw angular rates.
- Timestamp irregularities are reported as measurement-quality observations, not automatic proof of missing samples.
- Processing and feature extraction are offline; experimental signal-processing components must not be interpreted as production-validated.
- No original hardware is available for repeat field validation.
- There is no validated live sensor connection, production inference service, rider-warning capability, or safety certification.
- The synthetic recording is intended for application demonstration and testing only; it is not included in historical ML evaluation metrics.

## Repository layout

```text
dashboard.py              Streamlit recording explorer
pages/                    Recording Analysis and ML Evaluation
backend/src/adas/         Data loading, signal processing, features, evaluation
backend/tests/            Unit and application integration tests
demo_data/                Clearly labeled synthetic demonstration CSV
docs/                     Frozen evaluation evidence and technical documentation
requirements.txt          Python dependencies
```

## Project history and attribution

The original ADAS for Two-Wheeler Drivers project was developed as an academic project by **Mihir Vangani and Kunj Patel**, under the supervision of **Prof. (Dr.) Manish Chaturvedi** at the Institute of Infrastructure Technology Research and Management (IITRAM), Ahmedabad.

The original academic project description covered a broader hardware and wearable-sensor concept. The present repository demonstration focuses on the subsequently developed **offline software analytics and evaluation workflow**. Historical academic ambitions are not claims of currently validated product functionality.

## Intended use

Software engineering, signal-processing research, reproducibility studies, and portfolio demonstration.

**Not intended for real-time rider assistance or safety-critical decision-making.**

## Public evaluation evidence boundary

This public portfolio release contains a reduced, frozen aggregate evaluation report. It preserves the published comparison metrics, confusion matrices, class-level and fold-level results. It excludes individual prediction records, recording-level metrics, and historical recordings.

The original evaluation was performed using historical operator-generated annotations whose real-world ground-truth accuracy was not independently verified. The public report supports inspection of reported aggregate findings, not independent replay of the original training or prediction process.
