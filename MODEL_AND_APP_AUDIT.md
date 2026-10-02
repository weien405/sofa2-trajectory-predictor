# Model and app audit

- App type: Streamlit single-patient research predictor.
- Target: Class 3 (persistent-high) plus Class 4 (worsening) versus Class 1
  (low/stable) plus Class 2 (improving).
- Frozen selected model: XGBoost.
- Model SHA-256: `e9619b16d19bcffa1432f39d8e7608806c9eccdf88c5603f63f519f201e96d1d`.
- Predictors: the eight predictors frozen after training-only Boruta selection.
- Model retrained: NO.
- Data resplit: NO.
- Patient-level development data bundled with app: NO.
- Unknown binary input automatically coded as absence: NO — prediction is
  withheld until binary status is confirmed.
- Output: probability of unfavorable trajectory membership plus the
  study-derived threshold signal.
- Mortality predicted: NO.
- Individual four-class assignment predicted: NO.
- Local SHAP interpretation: descriptive model explanation only; not causal.
- Clinical-use authorization: NO — research demonstration only.
