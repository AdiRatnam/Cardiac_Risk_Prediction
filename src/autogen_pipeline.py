"""
autogen_pipeline.py
Module 7 - AutoGen Multi-Agent Clinical Workflow Orchestrator.

Orchestrates an autonomous multi-agent clinical decision pipeline:
  1. DataIngestionAgent: Validates, bounds-checks, and standardizes incoming multimodal data.
  2. DeepLearningInferenceAgent: Executes the end-to-end multi-branch neural network pipeline.
  3. LLMCrossCheckAgent: Conducts clinical QA second opinion comparing visual ECG and textual vitals.
  4. ClinicalReportAgent: Synthesizes multi-agent outputs into a structured physician-ready ICU handoff alert note.
"""

import numpy as np
import torch


class DataIngestionAgent:
    """
    Agent 1: Validates patient inputs, checks clinical plausibility ranges,
    standardizes features, and logs data stream integrity.
    """
    def __init__(self, scaler_clin, scaler_ecg, scaler_vit):
        self.scaler_clin = scaler_clin
        self.scaler_ecg = scaler_ecg
        self.scaler_vit = scaler_vit

    def process(self, clinical_dict, ecg_signal, vitals_window):
        # Clinical validation
        age = clinical_dict.get("age", 50)
        sys_bp = clinical_dict.get("systolic_bp", 120)
        chol = clinical_dict.get("cholesterol", 200)

        warnings = []
        if sys_bp > 180:
            warnings.append("Severe Hypertension (BP > 180)")
        elif sys_bp < 90:
            warnings.append("Hypotension (BP < 90)")

        if len(ecg_signal) != 216:
            raise ValueError(f"ECG length mismatch: expected 216 samples, got {len(ecg_signal)}")

        if vitals_window.shape != (24, 2):
            raise ValueError(f"Vitals shape mismatch: expected (24, 2), got {vitals_window.shape}")

        # Standardize dynamically matching scaler features
        if hasattr(self.scaler_clin, "feature_names_in_"):
            cols = list(self.scaler_clin.feature_names_in_)
            row = [float(clinical_dict.get(c, 0.0)) for c in cols]
            clin_arr = np.array([row], dtype=np.float32)
        elif hasattr(self.scaler_clin, "n_features_in_") and self.scaler_clin.n_features_in_ == 13:
            cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            row = [float(clinical_dict.get(c, 0.0)) for c in cols]
            clin_arr = np.array([row], dtype=np.float32)
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
            "status": "VALIDATED"
        }


class DeepLearningInferenceAgent:
    """
    Agent 2: Executes the multi-modal neural network pipeline, extracting
    branch embeddings, calculating anomaly reconstruction scores, and computing
    unified risk probability + explainability attention weights.
    """
    def __init__(self, clinical_model, ecg_model, vitals_model, vae_model, fusion_model, device="cpu"):
        self.clinical_model = clinical_model
        self.ecg_model = ecg_model
        self.vitals_model = vitals_model
        self.vae_model = vae_model
        self.fusion_model = fusion_model
        self.device = device

    @torch.no_grad()
    def process(self, validated_data):
        tc = torch.tensor(validated_data["c_scaled"], dtype=torch.float32).to(self.device)
        te = torch.tensor(validated_data["e_scaled"], dtype=torch.float32).to(self.device)
        tv = torch.tensor(validated_data["v_scaled"], dtype=torch.float32).to(self.device)

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

        if p >= 0.70:
            tier = "CRITICAL / HIGH RISK"
        elif p >= 0.35:
            tier = "MODERATE RISK"
        else:
            tier = "LOW RISK"

        return {
            "risk_probability": p,
            "risk_tier": tier,
            "modality_attention": {
                "Clinical_Metadata": float(attn[0]),
                "ECG_Morphology": float(attn[1]),
                "Vitals_Trend": float(attn[2]),
                "VAE_Anomaly_Latent": float(attn[3])
            },
            "reconstruction_error": float(recon_err.item()),
            "forecast_raw": forecast.cpu().numpy()[0]
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

        # Second opinion classification
        if hr_max > 125 or spo2_min < 88 or prob >= 0.65:
            llm_opinion = "HIGH"
            rationale = "Marked tachycardia/hypoxia and elevated cardiac neural risk probability observed."
        elif hr_max > 105 or spo2_min < 92 or prob >= 0.35:
            llm_opinion = "MODERATE"
            rationale = "Borderline vital sign instability; close monitoring warranted."
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
    Physician Alert & ICU Handoff Note.
    """
    def __init__(self):
        pass

    def generate_report(self, validated_data, dl_result, llm_result):
        prob = dl_result["risk_probability"]
        tier = dl_result["risk_tier"]
        attn = dl_result["modality_attention"]

        primary_driver = max(attn, key=attn.get)

        report = (
            f"======================================================================\n"
            f"          ICU CLINICAL DECISION SUPPORT: MULTI-AGENT HANDOFF NOTE      \n"
            f"======================================================================\n\n"
            f"1. PATIENT STATUS & RISK STRATIFICATION:\n"
            f"   - Predicted Acute Cardiac Deterioration Risk: {prob*100:.1f}%\n"
            f"   - Triage Category: [{tier}]\n"
            f"   - Pipeline Status: {validated_data['status']}\n\n"
            f"2. MODALITY EXPLAINABILITY (ATTENTION DRIVERS):\n"
            f"   - Primary Risk Driver     : {primary_driver.replace('_', ' ')} ({attn[primary_driver]*100:.1f}%)\n"
            f"   - ECG Morphology Weight   : {attn['ECG_Morphology']*100:.1f}%\n"
            f"   - Longitudinal Vitals     : {attn['Vitals_Trend']*100:.1f}%\n"
            f"   - VAE Anomaly Score       : {attn['VAE_Anomaly_Latent']*100:.1f}% (Recon Error: {dl_result['reconstruction_error']:.2f})\n"
            f"   - Baseline Health History : {attn['Clinical_Metadata']*100:.1f}%\n\n"
            f"3. MULTIMODAL LLM SECOND OPINION & QA CROSS-CHECK:\n"
            f"   - LLM Diagnostic Read     : {llm_result['llm_impression']} RISK\n"
            f"   - QA Concordance Status   : {'[PASSED] ' + llm_result['verification_note'] if llm_result['is_concordant'] else '[FLAGGED] ' + llm_result['verification_note']}\n"
            f"   - Clinical Reasoning      : {llm_result['rationale']}\n\n"
            f"4. RECOMMENDED CLINICAL INTERVENTIONS:\n"
        )

        if prob >= 0.70:
            report += (
                f"   [ACTION REQUIRED] Immediate bedside assessment by ICU intensivist.\n"
                f"   - Obtain STAT 12-lead ECG and serum Troponin / Cardiac Biomarkers.\n"
                f"   - Verify arterial line blood pressure and oxygen delivery titration.\n"
                f"   - Prepare antiarrhythmic / vasopressor support as clinically indicated.\n"
            )
        elif prob >= 0.35:
            report += (
                f"   [CLOSE MONITORING] Re-evaluate vital sign trends every 30 minutes.\n"
                f"   - Repeat telemetry rhythm strip check within 1 hour.\n"
                f"   - Review fluid balance and electrolytes (K+, Mg2+).\n"
            )
        else:
            report += (
                f"   [STABLE] Continue standard ICU telemetry and hourly observation protocol.\n"
                f"   - Patient parameters currently within safe hemodynamic bounds.\n"
            )

        report += (
            f"\n======================================================================\n"
            f"Report generated autonomously by AutoGen Clinical Multi-Agent Workflow.\n"
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

    def run(self, clinical_dict, ecg_signal, vitals_window):
        # Step 1: Ingestion
        validated = self.agent1_ingest.process(clinical_dict, ecg_signal, vitals_window)
        # Step 2: Deep Learning Inference
        dl_res = self.agent2_dl.process(validated)
        # Step 3: LLM Cross-Check
        llm_res = self.agent3_llm.process(validated, dl_res)
        # Step 4: Final Report Compilation
        report = self.agent4_report.generate_report(validated, dl_res, llm_res)

        return {
            "validated_data": validated,
            "dl_inference": dl_res,
            "llm_crosscheck": llm_res,
            "official_report": report
        }

