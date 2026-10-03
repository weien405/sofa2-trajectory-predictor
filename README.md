# SOFA-2 unfavorable trajectory predictor

This Streamlit research prototype serves the frozen XGBoost model from the
MIMIC-IV landmark-cohort analysis. It estimates unfavorable SOFA-2 trajectory
membership (Class 3 persistent-high or Class 4 worsening) versus
non-unfavorable membership (Class 1 low/stable or Class 2 improving).

## Live application

The public research prototype is available at:

https://sofa2-trajectory-predictor-gzhp3dybsebvv3g9lcomrr.streamlit.app/

Source repository:

https://github.com/weien405/sofa2-trajectory-predictor

## Run locally

1. Open this folder.
2. Double-click `run_local.bat`, or run:

   ```powershell
   py -m pip install -r requirements.txt
   py -m streamlit run streamlit_app.py
   ```

3. Open the local URL shown in the terminal (normally `http://localhost:8501`).

## Deployment

The application is deployed from the `main` branch with `streamlit_app.py` as
the entrypoint. The `model` directory must remain beside the entrypoint file.

The exact package versions used for verification are pinned in
`requirements.txt`. No patient-level development data are included in this app.

## Frozen model facts

- Development/landmark cohort: N=637; unfavorable N=129.
- Training set: N=445; unfavorable N=90.
- Held-out test set: N=192; unfavorable N=39.
- Selected algorithm: XGBoost.
- Held-out AUROC: 0.783 (95% CI 0.705–0.851).
- Held-out AUPRC: 0.440.
- Held-out Brier score: 0.140.
- Study-derived threshold: 0.2003.

Unknown binary inputs are not interpreted as absence. The app withholds a
prediction until all four binary inputs are confirmed as Yes or No. Missing
continuous inputs are handled by the preprocessing already frozen inside the
model pipeline.

## Intended use

Research demonstration only. The model is not a mortality model, has not
undergone prospective clinical validation, and must not be used to make or
withhold treatment decisions.
