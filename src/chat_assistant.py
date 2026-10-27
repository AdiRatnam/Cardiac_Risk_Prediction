"""
chat_assistant.py
Module 7 Extension - Interactive Bedside Clinical AI Assistant (CardioBot).

Provides real-time clinical reasoning and Q&A grounded in active patient telemetry,
neural network attention drivers, VAE reconstruction anomaly scores, and ACLS guidelines.
Works seamlessly offline via a deterministic clinical reasoning engine, and can
optionally leverage the Gemini API when GEMINI_API_KEY is configured.
"""

import os
import re
import numpy as np


class CardioBotAssistant:
    """
    Bedside Clinical AI Co-Pilot for ICU Intensivists and Nurses.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    def answer_query(self, query: str, context: dict) -> str:
        """
        Generates an authoritative, clinically grounded answer to the user's question.
        """
        if self.api_key:
            try:
                ans = self._query_gemini(query, context)
                if ans:
                    return ans
            except Exception:
                pass  # Fall back gracefully to deterministic engine

        return self._clinical_reasoning_engine(query, context)

    def _clinical_reasoning_engine(self, query: str, ctx: dict) -> str:
        """
        High-fidelity clinical reasoning engine matching cardiological protocols.
        """
        q = query.lower()
        dl = ctx.get("dl_inference", {})
        clin = ctx.get("raw_clinical", {})
        vits = ctx.get("raw_vitals", np.array([[80, 98]]))
        p_name = ctx.get("patient_name", "Active ICU Patient")
        p_risk = dl.get("risk_probability", 0.0) * 100.0
        tier = dl.get("risk_tier", "UNKNOWN")
        attn = dl.get("modality_attention", {})
        fid = dl.get("ecg_fiducials", {})
        recon = dl.get("reconstruction_error", 0.0)
        bios = dl.get("top_biomarkers", [])
        ci = dl.get("confidence_interval", [0.0, 1.0])
        std = dl.get("uncertainty_std", 0.0) * 100.0

        hr_curr = float(vits[-1, 0])
        spo2_curr = float(vits[-1, 1])
        sbp = clin.get("systolic_bp", clin.get("trestbps", 120))

        primary_mod = max(attn, key=attn.get) if attn else "Vitals_Trend"

        # 1. Why High Risk / Explain Risk
        if any(w in q for w in ["why", "cause", "risk", "reason", "critical", "score", "flagged"]):
            bio_str = ", ".join([f"{b['name']} ({b['weight']:.1f}%)" for b in bios[:2]])
            return (
                f"**Clinical Assessment for {p_name}:**\n\n"
                f"- **Acute Deterioration Probability:** **{p_risk:.1f}%** (+/-{std:.1f}%) [{tier}]\n"
                f"- **Primary Neural Driver:** **{primary_mod.replace('_', ' ')}** ({attn.get(primary_mod, 0)*100:.1f}% gating weight).\n"
                f"- **ECG Morphology:** {fid.get('morphology_tag', 'Evaluated')} with QRS duration of ~{fid.get('qrs_duration_ms', 90):.0f}ms.\n"
                f"- **Hemodynamic Status:** Current HR {hr_curr:.0f} bpm, SpO2 {spo2_curr:.1f}%, Systolic BP {sbp} mmHg.\n"
                f"- **Key Contributing Biomarkers:** {bio_str}.\n\n"
                f"*Summary:* The late-fusion head triggered this alert primarily because the combination of "
                f"conduction morphology and vital sign instability violates homeostatic boundaries learned from MIMIC-IV and MIT-BIH cohorts."
            )

        # 2. Medication / Drugs / Beta-blocker / Metoprolol / Vasopressors
        if any(w in q for w in ["med", "drug", "beta", "metoprolol", "pressor", "norepinephrine", "amiodarone", "give"]):
            med_advice = []
            if hr_curr < 55:
                med_advice.append("[HOLD] **Beta-Blockers (Metoprolol):** Current HR is bradycardic (<55 bpm). Negative chronotropes could precipitate complete AV block or cardiogenic shock.")
            elif hr_curr > 110 and sbp >= 100:
                med_advice.append("[CONSIDER] **Beta-Blocker Consideration:** If tachycardia is sinus in origin and patient is hemodynamically stable without acute heart failure decompensation, cautious low-dose IV Metoprolol (2.5 - 5mg) may be evaluated.")

            if sbp < 90:
                med_advice.append("[HOLD] **Hypotension Caution:** SBP < 90 mmHg. Vasodilators, nitrates, and ACE-inhibitors are contraindicated. First-line support: isotonic IV crystalloid bolus (500mL) or IV Norepinephrine infusion titrated to MAP >= 65 mmHg.")
            elif sbp > 175:
                med_advice.append("[ALERT] **Hypertensive Urgency:** SBP > 175 mmHg. Titrate IV arterial vasodilators (Nicardipine / Labetalol) with target 15-20% reduction in first hour.")

            if fid.get("is_arrhythmia", False):
                med_advice.append("[ALERT] **Arrhythmia Management:** Ectopic ventricular morphology noted. Obtain STAT 12-lead ECG, check serum K+ (target >4.0 mEq/L) and Mg2+ (target >2.0 mg/dL), and consider IV Amiodarone 150mg over 10 min if ventricular ectopy persists.")

            if not med_advice:
                med_advice.append("[STABLE] Patient parameters are currently stable. Maintain routine scheduled ICU home medications and monitor continuous telemetry.")

            return "**Bedside Pharmacotherapy & Safety Guidance:**\n\n" + "\n\n".join(med_advice)

        # 3. ECG / Waveform / QRS / Rhythm
        if any(w in q for w in ["ecg", "rhythm", "wave", "qrs", "arrhythmia", "st", "morphology"]):
            return (
                f"**Lead-II Beat Morphology Analysis:**\n\n"
                f"- **Diagnostic Tag:** `{fid.get('morphology_tag', 'Sinus Rhythm')}`\n"
                f"- **Calculated QRS Duration:** **{fid.get('qrs_duration_ms', 90):.0f} ms** "
                f"({'[ALERT] Prolonged / Wide Complex (>120ms)' if fid.get('qrs_duration_ms', 90) >= 120 else '[NORMAL] Standard duration (<100ms)'})\n"
                f"- **R-Peak Amplitude:** {fid.get('r_peak_amplitude', 1.0):.2f} mV at sample index {fid.get('r_peak_index', 108)}\n"
                f"- **ST-Segment Deviation:** {fid.get('st_elevation_mv', 0.0):+.2f} mV relative to isoelectric baseline\n\n"
                f"*Clinical Significance:* The multi-scale 1D-CNN extracts beat-level features using 15, 9, and 5-sample kernel filters, "
                f"passing them to spatial self-attention to identify ectopic ventricular beats and conduction blocks."
            )

        # 4. VAE / Anomaly / Reconstruction
        if any(w in q for w in ["vae", "anomaly", "recon", "unsupervised", "error"]):
            return (
                f"**BiLSTM-VAE Generative Anomaly Interpretation:**\n\n"
                f"- **Reconstruction Residual (MSE):** `{recon:.2f}`\n"
                f"- **Latent Manifold Status:** {'[ALERT] High Anomaly Residual (Unstable Pathological Trajectory)' if recon > 1.0 else '[NORMAL] Low Residual (Trajectory conforms to homeostatic norms)'}\n\n"
                f"*How it works:* The BiLSTM-VAE (Module 5) was trained strictly on stable, surviving ICU sequences in MIMIC-IV. "
                f"When fed acute deteriorating vitals (such as sudden hypoxia or rapid rate escalation), the decoder cannot reconstruct "
                f"the aberrant signal, causing a sharp surge in reconstruction error that feeds directly into the late fusion classifier."
            )

        # 5. Interventions / Orders / Next Steps
        if any(w in q for w in ["interven", "action", "order", "plan", "next", "protocol", "treat"]):
            if p_risk >= 70:
                return (
                    f"**Immediate Bedside Orders for {p_name} (Critical Triage):**\n\n"
                    f"1. **STAT 12-Lead Diagnostic ECG** to confirm rhythm and evaluate acute ischemic injury / STEMI.\n"
                    f"2. **STAT Cardiac Biomarkers:** High-sensitivity Troponin-I, CK-MB, and Serum Lactate.\n"
                    f"3. **Arterial Line Placement:** Immediate beat-to-beat hemodynamic BP surveillance.\n"
                    f"4. **Oxygen Escalation:** Titrate supplemental O2 via high-flow nasal cannula or non-invasive positive pressure ventilation (BiPAP) if SpO2 < 90%.\n"
                    f"5. **Bedside Echocardiogram (POCUS):** Assess LV ejection fraction, wall motion abnormalities, and pericardial effusion."
                )
            elif p_risk >= 35:
                return (
                    f"**Close Surveillance Orders for {p_name} (Moderate Risk):**\n\n"
                    f"1. Continuous telemetry rhythm monitoring; repeat strip check within 45 minutes.\n"
                    f"2. Check serum electrolytes (Potassium K+, Magnesium Mg2+) to prevent arrhythmogenesis.\n"
                    f"3. Cycle NIBP every 15 minutes.\n"
                    f"4. Review intake/output fluid balance over the preceding 12 hours."
                )
            else:
                return (
                    f"**Routine ICU Monitoring Protocol for {p_name} (Stable):**\n\n"
                    f"1. Maintain current telemetry monitoring and hourly vitals recording.\n"
                    f"2. Continue maintenance IV fluids and supportive care.\n"
                    f"3. Re-evaluate clinical risk score prior to step-down transfer."
                )

        # Default clinical summary response
        return (
            f"**CardioBot Bedside Summary for {p_name}:**\n\n"
            f"Patient is currently stratified at **{p_risk:.1f}% risk** of acute deterioration ({tier}). "
            f"Current vital signs show HR {hr_curr:.0f} bpm, SpO2 {spo2_curr:.1f}%, and SBP {sbp} mmHg. "
            f"Primary neural attention is focused on **{primary_mod.replace('_', ' ')}** with a VAE reconstruction error of `{recon:.2f}`. "
            f"You can ask me questions regarding pharmacotherapy, ECG morphology, VAE anomaly interpretation, or ACLS protocol recommendations!"
        )

    def _query_gemini(self, query: str, ctx: dict) -> str:
        """Calls Google GenAI / Gemini API if configured."""
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            prompt = (
                f"You are CardioBot, an expert ICU Cardiologist AI Assistant. "
                f"Answer the following question about this patient accurately and concisely:\n\n"
                f"PATIENT CONTEXT:\n"
                f"- Name/Bed: {ctx.get('patient_name')}\n"
                f"- Risk Probability: {ctx.get('dl_inference', {}).get('risk_probability', 0)*100:.1f}%\n"
                f"- Risk Tier: {ctx.get('dl_inference', {}).get('risk_tier')}\n"
                f"- Vitals: HR={ctx.get('raw_vitals', [[80, 98]])[-1, 0]:.0f} bpm, SpO2={ctx.get('raw_vitals', [[80, 98]])[-1, 1]:.1f}%\n"
                f"- ECG: {ctx.get('dl_inference', {}).get('ecg_fiducials', {}).get('morphology_tag')}\n"
                f"- Modality Attention: {ctx.get('dl_inference', {}).get('modality_attention')}\n"
                f"- Official Report:\n{ctx.get('official_report', '')[:600]}\n\n"
                f"PHYSICIAN QUESTION: {query}\n\n"
                f"Provide a structured, clinically authoritative, concise response."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception:
            return ""
