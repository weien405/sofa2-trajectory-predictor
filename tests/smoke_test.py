from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_utils import (  # noqa: E402
    EXPECTED_MODEL_SHA256,
    MODEL_PATH,
    file_sha256,
    load_artifacts,
    local_shap_values,
    predict_probability,
)


def main() -> None:
    model, manifest = load_artifacts()
    assert file_sha256(MODEL_PATH) == EXPECTED_MODEL_SHA256
    assert manifest["cohort_n"] == 637
    assert manifest["selected_model_training_only"] == "XGBoost"
    assert manifest["boruta_selected_n"] == 8

    reference = {
        "chronic_liver_disease": 0,
        "vasopressor_any_0_24h": 0,
        "invasive_mechanical_ventilation_0_24h": 0,
        "hematocrit": 33.1,
        "BUN": 20.0,
        "RRT_any_0_24h": 0,
        "heart_rate": 102.0,
        "systolic_bp": 122.5,
    }
    probability = predict_probability(model, reference)
    assert np.isfinite(probability) and 0 <= probability <= 1
    explanation = local_shap_values(model, reference)
    assert len(explanation) == 8
    assert explanation["shap_value"].notna().all()

    missing = dict(reference)
    missing.update(
        {
            "BUN": None,
            "heart_rate": None,
        }
    )
    missing_probability = predict_probability(model, missing)
    assert np.isfinite(missing_probability) and 0 <= missing_probability <= 1
    print(
        "PASS | frozen model loaded | reference probability "
        f"{probability:.6f} | missing-input probability {missing_probability:.6f}"
    )


if __name__ == "__main__":
    main()
