from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def main() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    assert not app.exception, app.exception
    calculate = next(button for button in app.button if button.label == "Calculate model estimate")
    calculate.click().run()
    assert not app.exception, app.exception
    assert any(metric.label == "Estimated probability" for metric in app.metric)
    assert any("threshold" in metric.label.lower() for metric in app.metric)

    unknown_app = AppTest.from_file(str(APP), default_timeout=30)
    unknown_app.run()
    chronic_liver = next(
        box for box in unknown_app.selectbox if box.label == "Chronic liver disease"
    )
    chronic_liver.select("Unknown / unavailable")
    next(
        button for button in unknown_app.button if button.label == "Calculate model estimate"
    ).click().run()
    assert not unknown_app.exception, unknown_app.exception
    assert any("prediction withheld" in error.value.lower() for error in unknown_app.error)
    assert not unknown_app.metric
    print(
        "PASS | page rendered | calculation completed | result widgets present | "
        "unknown binary status withheld"
    )


if __name__ == "__main__":
    main()
