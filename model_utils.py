from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
MODEL_PATH = MODEL_DIR / "FINAL_SELECTED_MODEL.pkl"
MANIFEST_PATH = MODEL_DIR / "ANALYSIS_MANIFEST.json"
EXPECTED_MODEL_SHA256 = "e9619b16d19bcffa1432f39d8e7608806c9eccdf88c5603f63f519f201e96d1d"

FEATURE_LABELS = {
    "chronic_liver_disease": "Chronic liver disease",
    "vasopressor_any_0_24h": "Vasopressor use (0–24 h)",
    "invasive_mechanical_ventilation_0_24h": "Invasive mechanical ventilation (0–24 h)",
    "hematocrit": "Hematocrit",
    "BUN": "Blood urea nitrogen",
    "RRT_any_0_24h": "Renal replacement therapy (0–24 h)",
    "heart_rate": "Heart rate",
    "systolic_bp": "Systolic blood pressure",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("The frozen model bundle is incomplete.")
    if file_sha256(MODEL_PATH) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Frozen model checksum mismatch.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    with MODEL_PATH.open("rb") as stream:
        model = pickle.load(stream)
    features = list(manifest["boruta_selected_features"])
    if features != list(FEATURE_LABELS):
        raise RuntimeError("Predictor order differs from the frozen analysis manifest.")
    return model, manifest


def make_input_frame(values: dict[str, float | int | None]) -> pd.DataFrame:
    row = {
        feature: np.nan if values.get(feature) is None else values[feature]
        for feature in FEATURE_LABELS
    }
    return pd.DataFrame([row], columns=list(FEATURE_LABELS))


def predict_probability(model: Any, values: dict[str, float | int | None]) -> float:
    frame = make_input_frame(values)
    probability = float(model.predict_proba(frame)[0, 1])
    if not 0.0 <= probability <= 1.0:
        raise RuntimeError("Model returned an invalid probability.")
    return probability


def local_shap_values(model: Any, values: dict[str, float | int | None]) -> pd.DataFrame:
    frame = make_input_frame(values)
    transformed = model.named_steps["preprocess"].transform(frame)
    estimator = model.named_steps["model"]
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    values_1d = shap_values.reshape(-1)
    if values_1d.size != len(FEATURE_LABELS):
        raise RuntimeError("Unexpected SHAP output shape.")
    result = pd.DataFrame(
        {
            "feature": list(FEATURE_LABELS),
            "display_name": list(FEATURE_LABELS.values()),
            "shap_value": values_1d,
        }
    )
    result["absolute_shap"] = result["shap_value"].abs()
    result["direction"] = np.where(
        result["shap_value"] >= 0,
        "Toward unfavorable",
        "Away from unfavorable",
    )
    return result.sort_values("absolute_shap", ascending=False).reset_index(drop=True)

