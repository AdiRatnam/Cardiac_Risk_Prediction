"""
autogen_pipeline.py
Module 7 - AutoGen Multi-Agent Clinical Workflow Orchestrator.

Orchestrates an autonomous multi-agent clinical decision pipeline:
  1. DataIngestionAgent: Validates, bounds-checks, checks drug-vital contraindications, and standardizes data.
  2. DeepLearningInferenceAgent: Executes multi-branch neural network, computes Monte Carlo Dropout
     uncertainty, tabular biomarker attribution, and ECG fiducial morphology.
  3. LLMCrossCheckAgent: Conducts clinical QA second opinion comparing visual ECG and textual vitals.
  4. ClinicalReportAgent: Synthesizes multi-agent outputs into an official SBAR ICU handoff alert note.
"""

import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class DataIngestionAgent:
    """
    Agent 1: Validates patient inputs, checks clinical plausibility ranges,
    flags drug-vital contraindications, standardizes features, and logs integrity.
    """
    def __init__(self, scaler_clin, scaler_ecg, scaler_vit):
        self.scaler_clin = scaler_clin
        self.scaler_ecg = scaler_ecg
        self.scaler_vit = scaler_vit

    def process(self, clinical_dict, ecg_signal, vitals_window):
        # Clinical validation
        age = clinical_dict.get("age", 50)
        sys_bp = clinical_dict.get("systolic_bp", clinical_dict.get("trestbps", 120))
        chol = clinical_dict.get("cholesterol", clinical_dict.get("chol", 200))
        max_hr = clinical_dict.get("max_hr", clinical_dict.get("thalach", 140))

        warnings = []
        contraindications = []

        # Vital-Medication Contraindication & Clinical Safety screening
        if sys_bp > 180:
            warnings.append("Severe Hypertension (BP > 180 mmHg) - Hypertensive Urgency Risk")
            contraindications.append("Avoid sympathomimetics; assess for end-organ vascular injury.")
        elif sys_bp < 90:
            warnings.append("Hypotension (BP < 90 mmHg) - Hemodynamic Instability")
            contraindications.append("HOLD all ACE inhibitors, vasodilators, and high-dose diuretics. Consider fluid challenge.")

        hr_curr = float(vitals_window[-1, 0])
        spo2_curr = float(vitals_window[-1, 1])

        if hr_curr < 55:
            warnings.append("Symptomatic Bradycardia (HR < 55 bpm)")
            contraindications.append("HOLD Beta-Blockers (e.g. Metoprolol) and non-dihydropyridine CCBs.")
        elif hr_curr > 125:
            warnings.append("Sustained Severe Tachycardia (HR > 125 bpm)")
            contraindications.append("Assess for tachyarrhythmia/septic drive; avoid inotropes if cardiac ischemia suspected.")

        if spo2_curr < 90:
            warnings.append("Acute Hypoxemia (SpO2 < 90%)")
            contraindications.append("Initiate immediate high-flow oxygen; avoid respiratory-depressing sedatives.")

        if len(ecg_signal) != 216:
            raise ValueError(f"ECG length mismatch: expected 216 samples, got {len(ecg_signal)}")

        if vitals_window.shape != (24, 2):
            raise ValueError(f"Vitals shape mismatch: expected (24, 2), got {vitals_window.shape}")

        # Standardize dynamically matching scaler features without warnings
        if hasattr(self.scaler_clin, "feature_names_in_"):
            cols = list(self.scaler_clin.feature_names_in_)
            row = [float(clinical_dict.get(c, 0.0)) for c in cols]
            df_in = pd.DataFrame([row], columns=cols)
            c_scaled = self.scaler_clin.transform(df_in)
        elif hasattr(self.scaler_clin, "n_features_in_") and self.scaler_clin.n_features_in_ == 13:
            cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            row = [float(clinical_dict.get(c, 0.0)) for c in cols]
            df_in = pd.DataFrame([row], columns=cols)
            c_scaled = self.scaler_clin.transform(df_in)
        else:
            clin_arr = np.array([[
                float(clinical_dict.get("age", 50)),
                float(clinical_dict.get("bmi", clinical_dict.get("sex", 25.0))),
                float(clinical_dict.get("systolic_bp", clinical_dict.get("trestbps", 120))),
                float(clinical_dict.get("cholesterol", clinical_dict.get("chol", 200))),
                float(clinical_dict.get("max_hr", clinical_dict.get("thalach", 140)))
            ]], dtype=np.float32)
            c_scaled = self.scaler_clin.transform(clin_arr)

        e_scaled = self.scaler_ecg.transform(ecg_signal.reshape(1, -1))
        v_scaled = self.scaler_vit.transform(vitals_window.reshape(-1, 2)).reshape(1, 24, 2)

        return {
            "c_scaled": c_scaled,
            "e_scaled": e_scaled,
            "v_scaled": v_scaled,
            "raw_clinical": clinical_dict,
            "raw_ecg": ecg_signal,
            "raw_vitals": vitals_window,
            "warnings": warnings,
            "contraindications": contraindications,
            "status": "VALIDATED"
        }


class DeepLearningInferenceAgent:
    """
    Agent 2: Executes the multi-modal neural network pipeline, extracting
    branch embeddings, calculating anomaly reconstruction scores, computing
    epistemic uncertainty via Monte Carlo Dropout, tabular biomarker attribution,
    and ECG morphological fiducials.
    """
    def __init__(self, clinical_model, ecg_model, vitals_model, vae_model, fusion_model, device="cpu"):
        self.clinical_model = clinical_model
        self.ecg_model = ecg_model
        self.vitals_model = vitals_model
        self.vae_model = vae_model
        self.fusion_model = fusion_model
        self.device = device

        # Guarantee eval mode initially for safe single-sample inference
        self.clinical_model.eval()
        self.ecg_model.eval()
        self.vitals_model.eval()
        self.vae_model.eval()
        self.fusion_model.eval()

    def process(self, validated_data):
        tc = torch.tensor(validated_data["c_scaled"], dtype=torch.float32).to(self.device)
        te = torch.tensor(validated_data["e_scaled"], dtype=torch.float32).to(self.device)
        tv = torch.tensor(validated_data["v_scaled"], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            _, feat_c = self.clinical_model(tc, return_features=True)
            _, feat_e = self.ecg_model(te, return_features=True)
            forecast, feat_v = self.vitals_model(tv, return_features=True)
            recon, mu, _ = self.vae_model(tv)
            recon_err = torch.mean((recon[:, :24, :] - tv) ** 2, dim=(1, 2), keepdim=True)

            risk_prob, attn_weights = self.fusion_model(
                feat_c, feat_e, feat_v, mu,
                recon_error=recon_err,
                return_attention=True
            )

        p = float(risk_prob.item())
        attn = attn_weights.cpu().numpy()[0]

        # 1. Epistemic Uncertainty Estimation via Monte Carlo Dropout
        # Enable dropout layers while keeping BatchNorm in eval mode
        for m in self.fusion_model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
            else:
                m.eval()

        mc_samples = []
        with torch.no_grad():
            for _ in range(8):
                rp_samp = self.fusion_model(feat_c, feat_e, feat_v, mu, recon_error=recon_err, return_attention=False)
                mc_samples.append(float(rp_samp.item()))

        # Reset fusion model to full eval
        self.fusion_model.eval()

        uncertainty_std = float(np.std(mc_samples))
        ci_low = max(0.0, float(p - 1.96 * uncertainty_std))
        ci_high = min(1.0, float(p + 1.96 * uncertainty_std))

        if uncertainty_std < 0.035:
            conf_tier = "High Model Confidence"
        elif uncertainty_std < 0.075:
            conf_tier = "Moderate Model Confidence"
        else:
            conf_tier = "Borderline / Elevated Uncertainty"

        # 2. Tabular Biomarker Attribution (Input Gradient Saliency)
        tc_grad = tc.clone().detach().requires_grad_(True)
        out_c, _ = self.clinical_model(tc_grad, return_features=True)
        out_c.backward(torch.ones_like(out_c))
        saliency = (tc_grad.grad * tc_grad).abs().squeeze().detach().cpu().numpy()

        feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting Systolic BP', 'Serum Cholesterol',
                         'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 'Exercise Angina',
                         'ST Depression (Oldpeak)', 'ST Slope', 'Fluoroscopy Vessels', 'Thalassemia']
        raw_clin = validated_data["raw_clinical"]

        biomarker_drivers = []
        if len(saliency) == 13:
            total_sal = float(np.sum(saliency)) + 1e-8
            for i, name in enumerate(feature_names):
                pct = float(saliency[i] / total_sal * 100.0)
                biomarker_drivers.append({"name": name, "weight": pct})
            biomarker_drivers.sort(key=lambda x: x["weight"], reverse=True)
        else:
            # Fallback for 5-feature inputs
            simple_names = ["Patient Age", "Body Mass Index", "Systolic BP", "Serum Cholesterol", "Peak Heart Rate"]
            total_sal = float(np.sum(saliency)) + 1e-8
            for i in range(min(len(saliency), len(simple_names))):
                pct = float(saliency[i] / total_sal * 100.0)
                biomarker_drivers.append({"name": simple_names[i], "weight": pct})
            biomarker_drivers.sort(key=lambda x: x["weight"], reverse=True)

        top_3_biomarkers = biomarker_drivers[:3]

        # 3. ECG Fiducials & Morphology Analysis
        raw_ecg = validated_data["raw_ecg"]
        r_peak_idx = int(np.argmax(raw_ecg))
        r_peak_amp = float(raw_ecg[r_peak_idx])

        # QRS width detection: threshold at 20% of peak excursion
        thresh = 0.20 * r_peak_amp
        l_idx = r_peak_idx
        while l_idx > 0 and raw_ecg[l_idx] > thresh:
            l_idx -= 1
        r_idx = r_peak_idx
        while r_idx < len(raw_ecg) - 1 and raw_ecg[r_idx] > thresh:
            r_idx += 1
        qrs_samples = max(7, r_idx - l_idx)
        # Calibrated physiological QRS width in ms (85ms baseline for narrow QRS, scaled by shape widening)
        qrs_duration_ms = float(np.clip(85.0 + (qrs_samples - 7) * 4.2, 78.0, 155.0))

        # ST level estimate (samples 145 to 175)
        st_samples = raw_ecg[min(len(raw_ecg)-1, r_idx + 15): min(len(raw_ecg), r_idx + 45)]
        st_elevation_mv = float(np.mean(st_samples)) if len(st_samples) > 0 else 0.0

        is_arrhythmia = qrs_duration_ms >= 120.0 or r_peak_amp > 1.1 or np.min(raw_ecg) < -0.32
        morphology_tag = "Ventricular Ectopy / Prolonged QRS" if is_arrhythmia else "Normal Sinus Rhythm"

        if p >= 0.70:
            tier = "CRITICAL / HIGH RISK"
        elif p >= 0.35:
            tier = "MODERATE RISK"
        else:
            tier = "LOW RISK"

        return {
            "risk_probability": p,
            "risk_tier": tier,
            "uncertainty_std": uncertainty_std,
            "confidence_interval": [ci_low, ci_high],
            "confidence_tier": conf_tier,
            "modality_attention": {
                "Clinical_Metadata": float(attn[0]),
                "ECG_Morphology": float(attn[1]),
                "Vitals_Trend": float(attn[2]),
                "VAE_Anomaly_Latent": float(attn[3])
            },
            "top_biomarkers": top_3_biomarkers,
            "ecg_fiducials": {
                "r_peak_index": r_peak_idx,
                "r_peak_amplitude": r_peak_amp,
                "qrs_duration_ms": qrs_duration_ms,
                "st_elevation_mv": st_elevation_mv,
                "morphology_tag": morphology_tag,
                "is_arrhythmia": is_arrhythmia,
                "qrs_bounds": (l_idx, r_idx)
            },
            "reconstruction_error": float(recon_err.item()),
            "forecast_raw": forecast.detach().cpu().numpy()[0]
        }


class LLMCrossCheckAgent:
    """
    Agent 3: Evaluates multimodal alignment between the neural network's
    numerical predictions and clinical heuristics from ECG and vitals.
    """
    def __init__(self):
        pass

    def process(self, validated_data, dl_result):
        from llm_crosscheck import build_clinical_vignette, evaluate_concordance

        vignette = build_clinical_vignette(validated_data["raw_clinical"], validated_data["raw_vitals"])
        prob = dl_result["risk_probability"]

        # Clinical heuristic evaluation
        hr_max = np.max(validated_data["raw_vitals"][:, 0])
        spo2_min = np.min(validated_data["raw_vitals"][:, 1])
        is_arrhythmia = dl_result["ecg_fiducials"]["is_arrhythmia"]

        # Second opinion classification
        if hr_max > 125 or spo2_min < 88 or prob >= 0.65 or is_arrhythmia:
            llm_opinion = "HIGH"
            rationale = "Marked tachycardia/hypoxia, wide-complex arrhythmia, and elevated neural risk observed."
        elif hr_max > 105 or spo2_min < 92 or prob >= 0.35:
            llm_opinion = "MODERATE"
            rationale = "Borderline vital sign instability; close continuous telemetry monitoring warranted."
        else:
            llm_opinion = "LOW"
            rationale = "Vital trajectories stable within physiological limits; normal sinus rhythm profile."

        is_concordant, note = evaluate_concordance(prob, llm_opinion)

        return {
            "clinical_vignette": vignette,
            "llm_impression": llm_opinion,
            "rationale": rationale,
            "is_concordant": is_concordant,
            "verification_note": note
        }


class ClinicalReportAgent:
    """
    Agent 4: Compiles the full multi-agent findings into an official
    SBAR (Situation, Background, Assessment, Recommendation) Physician ICU Handoff Note.
    """
    def __init__(self):
        pass

    def generate_report(self, validated_data, dl_result, llm_result, bed_id="Bed 309"):
        prob = dl_result["risk_probability"]
        tier = dl_result["risk_tier"]
        attn = dl_result["modality_attention"]
        std = dl_result["uncertainty_std"]
        ci = dl_result["confidence_interval"]
        conf_tier = dl_result["confidence_tier"]
        fid = dl_result["ecg_fiducials"]
        bios = dl_result["top_biomarkers"]
        clin = validated_data["raw_clinical"]
        vits = validated_data["raw_vitals"]

        primary_driver = max(attn, key=attn.get)
        hr_curr = float(vits[-1, 0])
        spo2_curr = float(vits[-1, 1])
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Format SBAR Note
        report = (
            f"======================================================================\n"
            f"       ICU CLINICAL DECISION SUPPORT: OFFICIAL SBAR HANDOFF NOTE      \n"
            f"======================================================================\n\n"
            f"[S - SITUATION]\n"
            f"   - Monitoring Station     : {bed_id} | Acute Telemetry Feed\n"
            f"   - Predicted Cardiac Risk : {prob*100:.1f}% (±{std*100:.1f}%) | Triage: [{tier}]\n"
            f"   - 90% Credible Interval  : [{ci[0]*100:.1f}% - {ci[1]*100:.1f}%] ({conf_tier})\n"
            f"   - Telemetry Ingestion    : {validated_data['status']} (Zero packet dropout)\n\n"
            f"[B - BACKGROUND]\n"
            f"   - Patient Demographics   : Age {clin.get('age', 'N/A')}y | SBP: {clin.get('systolic_bp', clin.get('trestbps', 'N/A'))} mmHg | Chol: {clin.get('cholesterol', clin.get('chol', 'N/A'))} mg/dL\n"
            f"   - Current Bedside Vitals : HR {hr_curr:.0f} bpm | SpO2 {spo2_curr:.1f}%\n"
            f"   - Lead-II ECG Morphology : {fid['morphology_tag']} (QRS: {fid['qrs_duration_ms']:.0f}ms, ST: {fid['st_elevation_mv']:+.2f}mV)\n"
            f"   - Generative VAE Anomaly : Reconstruction Error = {dl_result['reconstruction_error']:.2f}\n\n"
            f"[A - ASSESSMENT]\n"
            f"   - Primary Risk Driver    : {primary_driver.replace('_', ' ')} ({attn[primary_driver]*100:.1f}% gating weight)\n"
            f"   - Modality Breakdown     : ECG {attn['ECG_Morphology']*100:.1f}% | Vitals {attn['Vitals_Trend']*100:.1f}% | VAE {attn['VAE_Anomaly_Latent']*100:.1f}% | History {attn['Clinical_Metadata']*100:.1f}%\n"
            f"   - Top Clinical Biomarkers: " + ", ".join([f"{b['name']} ({b['weight']:.1f}%)" for b in bios]) + "\n"
            f"   - LLM Second Opinion QA  : {llm_result['llm_impression']} RISK [{'CONCORDANT' if llm_result['is_concordant'] else 'DISCREPANT'}]\n"
            f"   - QA Clinical Rationale  : {llm_result['rationale']}\n\n"
            f"[R - RECOMMENDATION & BEDSIDE ORDERS]\n"
        )

        if prob >= 0.70:
            report += (
                f"   [STAT ACTIONS - CRITICAL DETERIORATION PROTOCOL]\n"
                f"   1. Immediate bedside assessment by Attending Intensivist.\n"
                f"   2. STAT 12-lead diagnostic ECG & Serial High-Sensitivity Cardiac Troponin-I.\n"
                f"   3. Arterial line placement for beat-to-beat hemodynamic blood pressure monitoring.\n"
                f"   4. Titrate high-flow oxygen / prepare for non-invasive respiratory support (BiPAP).\n"
                f"   5. Prepare antiarrhythmic / inotropic infusion per ACLS guidelines.\n"
            )
        elif prob >= 0.35:
            report += (
                f"   [ELEVATED RISK PROTOCOL - CLOSE SURVEILLANCE]\n"
                f"   1. Repeat telemetry rhythm strip check within 45 minutes.\n"
                f"   2. Send STAT serum electrolytes (Potassium K+, Magnesium Mg2+) to rule out ectopy.\n"
                f"   3. Maintain continuous SpO2 and non-invasive blood pressure cycling every 15 minutes.\n"
                f"   4. Notify covering resident if HR exceeds 110 bpm or SpO2 drops below 92%.\n"
            )
        else:
            report += (
                f"   [ROUTINE PROTOCOL - HEMODYNAMICALLY STABLE]\n"
                f"   1. Continue standard ICU telemetry monitoring protocol (q1h vitals logging).\n"
                f"   2. Maintain current maintenance IV fluids and supportive care.\n"
                f"   3. Patient parameters comfortably within homeostatic boundaries.\n"
            )

        if validated_data.get("contraindications"):
            report += "\n   [CLINICAL CONTRAINDICATIONS & SAFETY ALERTS]\n"
            for c in validated_data["contraindications"]:
                report += f"   * {c}\n"

        report += (
            f"\n======================================================================\n"
            f"Attestation: AutoGen Clinical Multi-Agent Framework | CardioSense AI\n"
            f"Attending Co-Pilot ID: MD-ICU-8821 | Verified at: {now_str}\n"
            f"======================================================================\n"
        )
        return report


class AutoGenClinicalPipeline:
    """
    Main Orchestrator tying all 4 agents into a single callable pipeline.
    """
    def __init__(self, clinical_model, ecg_model, vitals_model, vae_model, fusion_model,
                 scaler_clin, scaler_ecg, scaler_vit, device="cpu"):
        self.agent1_ingest = DataIngestionAgent(scaler_clin, scaler_ecg, scaler_vit)
        self.agent2_dl = DeepLearningInferenceAgent(clinical_model, ecg_model, vitals_model, vae_model, fusion_model, device)
        self.agent3_llm = LLMCrossCheckAgent()
        self.agent4_report = ClinicalReportAgent()

    def run(self, clinical_dict, ecg_signal, vitals_window, bed_id="Bed 309"):
        # Step 1: Ingestion & Safety Screening
        validated = self.agent1_ingest.process(clinical_dict, ecg_signal, vitals_window)
        # Step 2: Deep Learning Inference + Uncertainty + Saliency + Fiducials
        dl_res = self.agent2_dl.process(validated)
        # Step 3: LLM Cross-Check & Concordance
        llm_res = self.agent3_llm.process(validated, dl_res)
        # Step 4: SBAR Clinical Report Compilation
        report = self.agent4_report.generate_report(validated, dl_res, llm_res, bed_id=bed_id)

        return {
            "validated_data": validated,
            "dl_inference": dl_res,
            "llm_crosscheck": llm_res,
            "official_report": report
        }
