# Week 4 Progress Report
## Multi-Modal Deep Learning Framework for Real-Time Cardiac Risk Prediction in ICU

### Module Covered This Week
**Module 4 - Temporal Vital Forecasting (BiLSTM)**

### What Was Completed
- **`src/vitals_data.py`**: real-data-first loader for ICU vitals
  (Heart Rate, SpO2), following the same pattern as `ecg_data.py` -
  tries a MIMIC-III/IV demo folder first (`icu/chartevents.csv` +
  `icu/icustays.csv`), falls back to a synthetic-but-physiologically-
  autocorrelated vitals generator (with occasional simulated
  deterioration events) if the folder isn't present. Shared by Module 5.
- **`src/vitals_models.py`**: `VitalsBiLSTM` - stacked Bidirectional
  LSTM over a rolling vitals window, forecasting the next
  `FORECAST_HORIZON` timesteps. He-initialization on the head layers,
  Dropout(0.3), BatchNorm on the pooled forward+backward hidden state.
  Exposes `return_features=True`, same convention as `ECG1DCNN`, so
  Week 6 fusion can pull this branch's feature vector identically.
- **`notebooks/Week4_Module4_BiLSTM.ipynb`**: mounts Drive, loads
  vitals data, standardizes, trains (or loads an existing checkpoint -
  smart checkpoint loading, same as Weeks 1-3), plots loss curves and
  an example forecast-vs-actual, reports MAE/RMSE.
- Verified data loader logic (shapes, deterioration-event labeling)
  runs correctly; model code checked for shape consistency by hand
  across the full forward pass (BiLSTM output dims -> BatchNorm ->
  Linear head -> reshaped forecast) since a full GPU training run
  wasn't performed outside Colab.

### What's Left (Upcoming Weeks)
- Run the actual training pass in Colab and record MAE/RMSE + loss
  curve in this report (currently unrun outside Colab).
- Wire real MIMIC deterioration/sepsis labels into `vitals_data.py`
  once credentialing is approved (currently synthetic-labels-only for
  the abnormal/normal split used by Module 5).
- **Week 5:** BiLSTM-VAE anomaly detection + synthetic generation
  (`vae_models.py`, reusing `vitals_data.py`).
- **Week 6:** fusion layer combining all branches' feature vectors +
  multimodal LLM cross-check.

### Files Delivered This Week
- `src/vitals_data.py`
- `src/vitals_models.py`
- `notebooks/Week4_Module4_BiLSTM.ipynb`

### How to Run
1. Ensure `Cardiac_Risk_Prediction/` (this project) is in
   `MyDrive/Cardiac_Risk_Prediction/`.
2. Open `notebooks/Week4_Module4_BiLSTM.ipynb` in Colab, Run All.
3. To use real MIMIC data instead of synthetic: extract a MIMIC-III/IV
   demo folder into `data/`, and set `MIMIC_DIR` in the notebook's data
   cell to that path.
