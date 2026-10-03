# Public deployment checklist

- [x] Frozen XGBoost model bundled.
- [x] No patient-level development data bundled.
- [x] Exact Python dependencies pinned.
- [x] Model checksum verified at application start.
- [x] Research-use and non-clinical-use warnings displayed.
- [x] Unknown binary status is not coded as absence.
- [x] Local prediction and SHAP explanation tested.
- [x] Streamlit automated page test passed.
- [x] Public GitHub repository created and pushed.
- [x] Streamlit Community Cloud app deployed.
- [x] Public URL tested in a signed-out browser session.

Public app: https://sofa2-trajectory-predictor-gzhp3dybsebvv3g9lcomrr.streamlit.app/

Public source: https://github.com/weien405/sofa2-trajectory-predictor

Signed-out verification completed on 2026-10-03. The app loaded and returned a
model estimate from the default example inputs.
