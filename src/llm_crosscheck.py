"""
llm_crosscheck.py
Module 6 - Multimodal LLM Cross-Check & Clinical QA Verification Layer.

Provides clinical quality assurance by presenting:
  1. Tabular clinical profile
  2. Sequential vitals summary (HR, SpO2 trend)
  3. Visual ECG waveform plot
to a Multimodal LLM (e.g. Gemini / GPT-4o), comparing the LLM's
qualitative diagnostic reasoning against the Deep Learning model's
numerical cardiac risk probability.
"""

import io
import base64
import numpy as np
import matplotlib.pyplot as plt


def render_ecg_plot_base64(ecg_signal, title="Lead II Beat Morphology"):
    """
    Renders an ECG beat window into a high-contrast base64 PNG string
    suitable for submission to a multimodal LLM vision API.
    """
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=120)
    ax.plot(ecg_signal, color="#d62728", linewidth=1.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Sample Index (Time)", fontsize=9)
    ax.set_ylabel("Amplitude (mV)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_b64


def build_clinical_vignette(metadata_dict, vitals_seq):
    """
    Formats raw clinical numbers into a realistic clinical chart summary.
    """
    hr_mean = float(np.mean(vitals_seq[:, 0]))
    hr_min = float(np.min(vitals_seq[:, 0]))
    hr_max = float(np.max(vitals_seq[:, 0]))

    spo2_mean = float(np.mean(vitals_seq[:, 1]))
    spo2_min = float(np.min(vitals_seq[:, 1]))

    def _fmt(val, spec=""):
        if val is None or val == "N/A":
            return "N/A"
        try:
            if spec:
                return f"{float(val):{spec}}"
            return str(val)
        except Exception:
            return str(val)

    age_str = _fmt(metadata_dict.get("age", "N/A"))
    bmi_str = _fmt(metadata_dict.get("bmi", None), ".1f")
    bp_str = _fmt(metadata_dict.get("systolic_bp", metadata_dict.get("trestbps", None)), ".0f")
    chol_str = _fmt(metadata_dict.get("cholesterol", metadata_dict.get("chol", None)), ".0f")
    peak_hr_str = _fmt(metadata_dict.get("max_hr", metadata_dict.get("thalach", None)), ".0f")

    vignette = (
        f"PATIENT ICU CLINICAL SUMMARY:\n"
        f"-----------------------------\n"
        f"Demographics & History:\n"
        f"  - Age: {age_str} years\n"
        f"  - BMI: {bmi_str}\n"
        f"  - Resting Systolic BP: {bp_str} mmHg\n"
        f"  - Total Cholesterol: {chol_str} mg/dL\n"
        f"  - Peak Heart Rate: {peak_hr_str} bpm\n\n"
        f"ICU 24-Hour Continuous Vital Trends:\n"
        f"  - Heart Rate Range: {hr_min:.1f} - {hr_max:.1f} bpm (Mean: {hr_mean:.1f} bpm)\n"
        f"  - SpO2 Range: {spo2_min:.1f}% - 100.0% (Mean: {spo2_mean:.1f}%)\n"
        f"  - Trend Trajectory: {'Deteriorating (Tachycardia / Hypoxia)' if hr_max > 120 or spo2_min < 90 else 'Stable baseline'}\n"
    )
    return vignette


def evaluate_concordance(dl_risk_prob: float, llm_risk_category: str):
    """
    Compares the deep learning model's output probability against the LLM's
    clinical risk category.

    Categories: 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'
    Returns:
      is_concordant: bool
      discrepancy_note: str
    """
    cat = llm_risk_category.upper().strip()
    if dl_risk_prob >= 0.70:
        expected = ["HIGH", "CRITICAL"]
    elif dl_risk_prob >= 0.35:
        expected = ["MODERATE", "HIGH"]
    else:
        expected = ["LOW", "MODERATE"]

    is_concordant = any(e in cat for e in expected)
    if is_concordant:
        note = f"CONCORDANT: DL Model ({dl_risk_prob*100:.1f}%) aligns with LLM assessment ({cat})."
    else:
        note = (
            f"DISCREPANCY FLAGGED: DL Model predicted {dl_risk_prob*100:.1f}% risk, "
            f"but LLM assessed clinical risk as '{cat}'. Human clinician review advised."
        )
    return is_concordant, note

