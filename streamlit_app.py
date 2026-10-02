from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from model_utils import FEATURE_LABELS, load_artifacts, local_shap_values, predict_probability


st.set_page_config(
    page_title="SOFA-2 Trajectory Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: #f6f8fb;}
    .block-container {max-width: 1180px; padding-top: 2.0rem; padding-bottom: 3rem;}
    h1, h2, h3 {color: #17243a; letter-spacing: -0.02em;}
    .hero {
        padding: 1.35rem 1.55rem;
        background: linear-gradient(120deg, #0b3158 0%, #125d86 100%);
        border-radius: 16px;
        color: white;
        box-shadow: 0 8px 24px rgba(18, 65, 102, 0.14);
        margin-bottom: 1.15rem;
    }
    .hero h1 {color: white; font-size: 2.05rem; margin: 0 0 0.35rem 0;}
    .hero p {color: #e6f2f7; margin: 0; font-size: 1.02rem;}
    .eyebrow {font-size: 0.76rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: #96d7e7;}
    .result-card {
        background: white;
        border: 1px solid #dce4ec;
        border-left: 6px solid #12718d;
        border-radius: 14px;
        padding: 1.2rem 1.35rem;
        box-shadow: 0 4px 16px rgba(26, 47, 73, 0.07);
    }
    .probability {font-size: 2.55rem; font-weight: 760; line-height: 1; color: #0b5372;}
    .status-chip {
        display: inline-block; border-radius: 999px; padding: 0.28rem 0.72rem;
        font-size: 0.86rem; font-weight: 650; background: #e7f3f6; color: #0b5372;
    }
    .fine-print {font-size: 0.86rem; color: #5c6877;}
    div[data-testid="stForm"] {background: white; border: 1px solid #dce4ec; border-radius: 14px; padding: 1rem 1.15rem;}
    div[data-testid="stMetric"] {background: white; border: 1px solid #e1e7ee; border-radius: 11px; padding: 0.75rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return load_artifacts()


def binary_value(label: str, key: str, help_text: str) -> int | None:
    choice = st.selectbox(
        label,
        ["No", "Yes", "Unknown / unavailable"],
        index=0,
        key=key,
        help=help_text,
    )
    return {"No": 0, "Yes": 1, "Unknown / unavailable": None}[choice]


def numeric_value(
    label: str,
    key: str,
    default: float,
    minimum: float,
    maximum: float,
    step: float,
    unit: str,
    help_text: str,
) -> float | None:
    missing = st.checkbox(f"{label} unavailable", key=f"{key}_missing")
    value = st.number_input(
        f"{label} ({unit})",
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=step,
        key=key,
        disabled=missing,
        help=help_text,
    )
    return None if missing else float(value)


try:
    model, manifest = get_model_bundle()
except Exception as exc:
    st.error(f"The frozen model could not be loaded: {exc}")
    st.stop()


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Research prototype · MIMIC-IV development cohort</div>
      <h1>Early prediction of unfavorable SOFA-2 trajectory membership</h1>
      <p>Estimate the probability of subsequently following a persistent-high or worsening SOFA-2 trajectory.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "For research demonstration only. This internally evaluated model is not a clinical decision rule, "
    "does not predict mortality, and should not be used to direct individual patient care."
)

with st.sidebar:
    st.header("Model definition")
    st.markdown(
        "**Unfavorable:** Class 3 (persistent-high) or Class 4 (worsening)\n\n"
        "**Non-unfavorable:** Class 1 (low/stable) or Class 2 (improving)"
    )
    st.divider()
    st.subheader("Frozen analysis")
    st.markdown(
        f"- Algorithm: **{manifest['selected_model_training_only']}**\n"
        f"- Landmark cohort: **N={manifest['cohort_n']}**\n"
        f"- Training set: **N={manifest['train_n']}**\n"
        f"- Held-out test set: **N={manifest['heldout_test_n']}**\n"
        f"- Selected predictors: **{manifest['boruta_selected_n']}**"
    )
    metrics = manifest["heldout_metrics_selected_model"]
    auc_ci = manifest["heldout_AUROC_CI"]
    st.caption(
        f"Held-out AUROC {metrics['AUROC']:.3f} "
        f"(95% CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f}); "
        f"AUPRC {metrics['AUPRC']:.3f}; Brier score {metrics['Brier']:.3f}."
    )
    st.divider()
    st.caption("Version 1.0 · frozen model bundle")

st.subheader("Enter early ICU information")
st.caption(
    "Vital signs use the earliest valid value during 0–6 h; laboratory measurements use the earliest "
    "valid value during 0–24 h; treatment variables cover 0–24 h."
)

with st.form("prediction_form", clear_on_submit=False):
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Baseline condition and early treatments")
        chronic_liver = binary_value(
            "Chronic liver disease",
            "chronic_liver_disease",
            "Documented chronic liver disease at baseline. Do not code unavailable status as No.",
        )
        vasopressor = binary_value(
            "Vasopressor use during 0–24 h",
            "vasopressor",
            "Any documented vasopressor exposure during the first 24 hours.",
        )
        ventilation = binary_value(
            "Invasive mechanical ventilation during 0–24 h",
            "ventilation",
            "Any invasive mechanical ventilation during the first 24 hours.",
        )
        rrt = binary_value(
            "Renal replacement therapy during 0–24 h",
            "rrt",
            "Any renal replacement therapy during the first 24 hours.",
        )

    with right:
        st.markdown("#### Early physiologic and laboratory measurements")
        heart_rate = numeric_value(
            "Heart rate", "heart_rate", 102.0, 20.0, 250.0, 1.0, "beats/min",
            "Earliest valid heart rate during 0–6 hours.",
        )
        systolic_bp = numeric_value(
            "Systolic blood pressure", "systolic_bp", 122.5, 30.0, 300.0, 1.0, "mmHg",
            "Earliest valid systolic blood pressure during 0–6 hours.",
        )
        bun = numeric_value(
            "Blood urea nitrogen", "bun", 20.0, 1.0, 250.0, 1.0, "mg/dL",
            "Earliest valid blood urea nitrogen measurement during 0–24 hours.",
        )
        hematocrit = numeric_value(
            "Hematocrit", "hematocrit", 33.1, 5.0, 70.0, 0.1, "%",
            "Earliest valid hematocrit measurement during 0–24 hours.",
        )

    submitted = st.form_submit_button(
        "Calculate model estimate",
        type="primary",
        width="stretch",
    )

if submitted:
    input_values = {
        "chronic_liver_disease": chronic_liver,
        "vasopressor_any_0_24h": vasopressor,
        "invasive_mechanical_ventilation_0_24h": ventilation,
        "hematocrit": hematocrit,
        "BUN": bun,
        "RRT_any_0_24h": rrt,
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
    }
    binary_inputs = {
        "Chronic liver disease": chronic_liver,
        "Vasopressor use": vasopressor,
        "Invasive mechanical ventilation": ventilation,
        "Renal replacement therapy": rrt,
    }
    unknown_binary = [label for label, value in binary_inputs.items() if value is None]
    if unknown_binary:
        st.session_state.pop("prediction_result", None)
        st.error(
            "Prediction withheld: confirm Yes or No for "
            + ", ".join(unknown_binary)
            + ". Unknown binary status is not coded as absence."
        )
    else:
        probability = predict_probability(model, input_values)
        threshold = float(manifest["decision_threshold"])
        st.session_state["prediction_result"] = {
            "values": input_values,
            "probability": probability,
            "threshold": threshold,
        }

if "prediction_result" in st.session_state:
    result = st.session_state["prediction_result"]
    probability = float(result["probability"])
    threshold = float(result["threshold"])
    above = probability >= threshold

    st.divider()
    st.subheader("Model estimate")
    status = "Above study-derived threshold" if above else "Below study-derived threshold"
    st.markdown(
        f"""
        <div class="result-card">
          <div class="fine-print">Estimated probability of unfavorable trajectory membership</div>
          <div class="probability">{probability * 100:.1f}%</div>
          <div class="status-chip">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(probability, 0.0), 1.0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated probability", f"{probability:.3f}")
    c2.metric("Study-derived threshold", f"{threshold:.3f}")
    c3.metric("Unfavorable prevalence", f"{manifest['unfavorable_n'] / manifest['cohort_n']:.1%}")
    st.info(
        "The threshold was selected from nested out-of-fold training predictions using the maximum "
        "Youden index. It is a study-specific operating point—not a treatment threshold."
    )

    shap_table = local_shap_values(model, result["values"])
    with st.expander("Explain this model estimate", expanded=True):
        st.caption(
            "SHAP values describe how each input moved this model estimate on the model-output "
            "log-odds scale. They do not represent causal effects or adjusted clinical associations."
        )
        chart_data = shap_table.copy()
        chart_data["display_name"] = pd.Categorical(
            chart_data["display_name"],
            categories=chart_data["display_name"].tolist()[::-1],
            ordered=True,
        )
        chart = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X("shap_value:Q", title="SHAP value (model-output log-odds scale)"),
                y=alt.Y("display_name:N", title=None, sort=chart_data["display_name"].tolist()),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(
                        domain=["Toward unfavorable", "Away from unfavorable"],
                        range=["#c24b4b", "#2f78a8"],
                    ),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("display_name:N", title="Predictor"),
                    alt.Tooltip("shap_value:Q", title="SHAP value", format=".3f"),
                    alt.Tooltip("direction:N", title="Direction"),
                ],
            )
            .properties(height=330)
        )
        zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
            color="#59636e", strokeDash=[4, 3]
        ).encode(x="x:Q")
        st.altair_chart(chart + zero_line, width="stretch")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": manifest["selected_model_training_only"],
        "target": manifest["target"],
        "estimated_probability": probability,
        "study_derived_threshold": threshold,
        "threshold_signal": status,
        "inputs": result["values"],
        "warning": "Research prototype; not for clinical decision-making.",
    }
    st.download_button(
        "Download this result (JSON)",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="sofa2_trajectory_prediction.json",
        mime="application/json",
    )

with st.expander("Input definitions and applicable population"):
    st.markdown(
        """
        - Intended research population: adult ICU patients with severe acute pancreatitis who meet the
          study's 72-h landmark cohort definition.
        - Heart rate and systolic blood pressure: earliest valid measurement during 0–6 h.
        - Blood urea nitrogen and hematocrit: earliest valid measurement during 0–24 h.
        - Treatments: any documented exposure during 0–24 h.
        - Unknown binary status is never interpreted as absence. The app withholds a prediction until
          each binary input is confirmed as Yes or No.
        - This tool estimates binary trajectory membership and does not assign one of the four individual
          trajectory classes.
        """
    )

st.caption(
    "Model developed in MIMIC-IV (N=637) and evaluated once in a held-out test set (N=192). "
    "Independent prospective validation and clinical-impact assessment have not been performed."
)
