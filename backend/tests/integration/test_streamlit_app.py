"""Streamlit application regression acceptance."""

from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[3]


def healthy(app):
    assert len(app.exception) == 0
    assert len(app.error) == 0


def selector(app, name):
    found = [x for x in app.selectbox if x.label == name]
    assert len(found) == 1
    return found[0]


def values(app):
    return [str(x.value) for x in app.metric]


def test_dashboard():
    app = AppTest.from_file(
        str(ROOT / "dashboard.py"), default_timeout=30
    ).run()
    healthy(app)
    assert values(app)[:4] == ["600", "1", "0", "600"]
    assert any(
        "SYNTHETIC DEMO" in str(option)
        for option in selector(app, "Choose recording").options
    )


def test_recording_analysis():
    app = AppTest.from_file(
        str(ROOT / "pages/1_Recording_Analysis.py"),
        default_timeout=60,
    ).run()
    healthy(app)
    box = selector(app, "Recording")
    positions = [
        i for i, option in enumerate(box.options)
        if "SYNTHETIC DEMO" in str(option)
    ]
    assert len(positions) == 1
    box.select_index(positions[0])
    app.run()
    healthy(app)

    buttons = [
        b for b in app.button
        if b.label == "Analyze recording"
    ]
    assert len(buttons) == 1
    buttons[0].click()
    app.run()
    healthy(app)
    assert values(app) == ["600", "1", "1", "28", "26"]


def test_frozen_ml_evaluation():
    app = AppTest.from_file(
        str(ROOT / "pages/2_ML_Evaluation.py"),
        default_timeout=30,
    ).run()
    healthy(app)
    assert values(app)[:3] == ["0.352", "0.351", "3,478"]

    selector(app, "Evaluation population").select_index(1)
    app.run()
    healthy(app)
    assert values(app)[:3] == ["0.352", "0.354", "3,578"]

    selector(app, "Feature representation").select_index(2)
    app.run()
    healthy(app)
    assert values(app)[:3] == ["0.405", "0.407", "3,578"]

    selector(app, "Evaluation population").select_index(0)
    app.run()
    healthy(app)
    assert values(app)[:3] == ["0.411", "0.403", "3,478"]
