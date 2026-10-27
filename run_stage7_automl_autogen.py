"""
run_stage7_automl_autogen.py
Executes Stage 7:
  1. Optuna Bayesian Optimization (AutoML search over hyperparameters).
  2. AutoGen Multi-Agent Clinical Workflow evaluation on sample patients.
  3. Saves all Stage 7 output artifacts to outputs/.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_utils import load_clinical_metadata, load_cleveland_heart_disease, load_realistic_clinical_metadata
from models import ClinicalMLP
from ecg_models import ECG1DCNN
from vitals_models import VitalsBiLSTM
from vae_models import VitalsBiLSTMVAE
from fusion_models import CardiacRiskFusionNet
from automl_tuning import run_clinical_automl_search, plot_automl_results
from autogen_pipeline import AutoGenClinicalPipeline
from sklearn.preprocessing import StandardScaler

device = torch.device("cpu")
print("============================================================")
print("STAGE 7 EXECUTION: AUTOML & AUTOGEN CLINICAL PIPELINE")
print("============================================================\n")

# 1. Run AutoML Bayesian Search
print(">>> Running Optuna Bayesian Hyperparameter Search (AutoML)...")
try:
    df_c, y_c = load_cleveland_heart_disease()
    print(f"AutoML running on real UCI Cleveland dataset ({len(df_c)} records, {df_c.shape[1]} features).")
except Exception:
    df_c, y_c = load_clinical_metadata(n_samples=1000)

study, best_params = run_clinical_automl_search(df_c, y_c, n_trials=12, timeout=60, device=device)

print(f"AutoML Search Completed: Best Trial Value = {study.best_value:.4f}")
print("Optimal Hyperparameters Found:")
for k, v in best_params.items():
    print(f"  - {k}: {v}")

# Save Best Parameters
df_best = pd.DataFrame([{"Parameter": k, "Optimal_Value": v} for k, v in best_params.items()])
df_best.loc[len(df_best)] = {"Parameter": "Best_Validation_AUC", "Optimal_Value": round(study.best_value, 4)}
df_best.to_csv(os.path.join(OUTPUTS_DIR, "stage7_automl_best_trial.csv"), index=False)

# Generate and save AutoML plots
hist_path, imp_path = plot_automl_results(study, output_dir=OUTPUTS_DIR)
print(f"Saved AutoML Convergence Plot: {hist_path}")
if imp_path:
    print(f"Saved Parameter Importance Plot: {imp_path}")


# 2. Run AutoGen Multi-Agent Clinical Workflow
print("\n>>> Executing AutoGen Multi-Agent Clinical Pipeline...")

# Prepare scalers
scaler_c = StandardScaler().fit(df_c)
t_ax = np.linspace(-1, 1, 216)
dummy_ecg = np.sin(2 * np.pi * 5 * t_ax).reshape(1, -1)
scaler_e = StandardScaler().fit(dummy_ecg)
dummy_v = np.stack([np.random.normal(75, 10, 24), np.random.normal(98, 2, 24)], axis=-1)
scaler_v = StandardScaler().fit(dummy_v.reshape(-1, 2))

# Load checkpoints
clin_m = ClinicalMLP(input_dim=df_c.shape[1]).to(device)
ecg_m = ECG1DCNN(input_length=216).to(device)
vit_m = VitalsBiLSTM(n_features=2, horizon=6).to(device)
vae_m = VitalsBiLSTMVAE(n_features=2, seq_len=24, horizon=6).to(device)
fuse_m = CardiacRiskFusionNet(clinical_dim=32, ecg_dim=64, vitals_dim=64, vae_dim=16, include_recon_error=True).to(device)

def try_load(m, fname):
    p = os.path.join(CHECKPOINTS_DIR, fname)
    if os.path.exists(p):
        cp = torch.load(p, map_location=device)
        st = cp["model_state_dict"] if "model_state_dict" in cp else cp
        m.load_state_dict(st, strict=False)

try_load(clin_m, "module2_stable_mlp.pt")
try_load(ecg_m, "module3_ecg_cnn.pt")
try_load(vit_m, "module4_bilstm.pt")
try_load(vae_m, "module5_vae.pt")
try_load(fuse_m, "module6_fusion.pt")

clin_m.eval(); ecg_m.eval(); vit_m.eval(); vae_m.eval(); fuse_m.eval()

pipeline = AutoGenClinicalPipeline(
    clin_m, ecg_m, vit_m, vae_m, fuse_m,
    scaler_c, scaler_e, scaler_v, device=device
)

# Test on 2 contrasting patient profiles
p_norm = 0.15 * np.exp(-((t_ax + 0.35)**2)/(2*0.05**2))
qrs_norm = 1.0 * np.exp(-(t_ax**2)/(2*0.015**2)) - 0.2 * np.exp(-((t_ax + 0.05)**2)/(2*0.01**2))
t_norm = 0.3 * np.exp(-((t_ax - 0.35)**2)/(2*0.08**2))
norm_beat = (p_norm + qrs_norm + t_norm).astype(np.float32)
abnorm_beat = (0.05 * p_norm + 1.2 * np.exp(-(t_ax**2)/(2*0.04**2)) - 0.4 * np.exp(-((t_ax-0.1)**2)/(2*0.05**2))).astype(np.float32)

patient_a_meta = {
    "age": 45, "sex": 0, "cp": 0, "trestbps": 116, "chol": 178,
    "fbs": 0, "restecg": 0, "thalach": 165, "exang": 0, "oldpeak": 0.2, "slope": 1, "ca": 0, "thal": 2,
    "bmi": 23.5, "systolic_bp": 116, "cholesterol": 178, "max_hr": 165
}
patient_a_ecg = norm_beat
patient_a_vit = np.stack([np.random.normal(72, 2, 24), np.random.normal(98, 0.5, 24)], axis=-1).astype(np.float32)

patient_b_meta = {
    "age": 72, "sex": 1, "cp": 3, "trestbps": 175, "chol": 280,
    "fbs": 1, "restecg": 2, "thalach": 115, "exang": 1, "oldpeak": 3.4, "slope": 2, "ca": 3, "thal": 3,
    "bmi": 33.1, "systolic_bp": 175, "cholesterol": 280, "max_hr": 115
}
patient_b_ecg = abnorm_beat
patient_b_vit = np.stack([np.linspace(80, 132, 24), np.linspace(96, 86, 24)], axis=-1).astype(np.float32)

res_a = pipeline.run(patient_a_meta, patient_a_ecg, patient_a_vit)
res_b = pipeline.run(patient_b_meta, patient_b_ecg, patient_b_vit)

log_path = os.path.join(OUTPUTS_DIR, "stage7_agent_orchestration_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("AUTOGEN MULTI-AGENT CLINICAL DECISION SUPPORT ORCHESTRATION LOG\n")
    f.write("=================================================================\n\n")
    f.write("[CASE 1: STABLE ICU PATIENT]\n")
    f.write(res_a["official_report"])
    f.write("\n\n" + "="*70 + "\n\n")
    f.write("[CASE 2: ACUTE CRITICAL CARDIAC DETERIORATION]\n")
    f.write(res_b["official_report"])

print(f"AutoGen Multi-Agent Orchestration Log Saved: {log_path}")
print(f"  - Case 1 (Stable) Predicted Risk : {res_a['dl_inference']['risk_probability']*100:.2f}% ({res_a['dl_inference']['risk_tier']})")
print(f"  - Case 2 (Acute)  Predicted Risk : {res_b['dl_inference']['risk_probability']*100:.2f}% ({res_b['dl_inference']['risk_tier']})")
print("\n============================================================")
print("STAGE 7 EXECUTION COMPLETE & ALL ARTIFACTS SAVED SUCCESSFULLY!")
print("============================================================\n")

