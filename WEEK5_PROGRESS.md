# Week 5 Progress Report
## Multi-Modal Deep Learning Framework for Real-Time Cardiac Risk Prediction in ICU

### Module Covered This Week
**Module 5 - Generative Anomaly Detection (BiLSTM-VAE)**

### What Was Completed
- **`src/vae_models.py`**: `VitalsBiLSTMVAE` - the BiLSTM serves as BOTH
  encoder and decoder of a VAE, exactly per the architecture spec:
  - Encoder: BiLSTM -> pooled hidden state -> (mu, log_var) -> z
    (reparameterization trick).
  - Decoder: BiLSTM that reconstructs the input window AND forecasts
    the horizon steps beyond it from z.
  - `reconstruction_error()`: per-sample MSE anomaly score.
  - `generate()`: samples z ~ N(0, I), decodes into new synthetic
    vitals sequences for rare-event augmentation.
  - `vae_loss()`: reconstruction MSE + beta-weighted KL divergence
    (standard VAE objective).
- **`notebooks/Week5_Module5_VAE.ipynb`**: loads vitals data via the
  shared `vitals_data.py`, **filters to normal-only sequences for
  training** (critical for a valid anomaly detector - stated explicitly
  in both the code docstring and the notebook), trains/loads checkpoint,
  plots the normal-vs-abnormal reconstruction-error histogram with a
  95th-percentile threshold, computes a detection rate on held-out
  abnormal sequences, and generates + plots 5 synthetic vitals samples.
- Verified model logic by hand: traced tensor shapes through the full
  encoder -> reparameterize -> decoder path and confirmed every
  BiLSTM output dimension matches the next layer's expected input
  dimension (encoder's bidirectional hidden concat -> `fc_mu`/
  `fc_log_var`; decoder's expanded latent -> repeated -> BiLSTM ->
  `fc_out`). Full GPU training run not yet performed outside Colab.

### Why "Normal-Only Training" Matters (for the report / PPT)
A VAE trained on a mix of normal and abnormal vitals will learn to
reconstruct abnormal patterns too, which erases the reconstruction-error
spike the whole anomaly-detection mechanism depends on. This notebook
filters the synthetic dataset's `labels==0` sequences before training,
and holds out the `labels==1` (deterioration-event) sequences purely for
evaluating whether the trained model's reconstruction error actually
separates them from normal data.

### What's Left (Upcoming Weeks)
- Run the actual training pass in Colab; record the real normal-vs-
  abnormal error separation and detection rate in this report.
- Once real MIMIC deterioration/sepsis labeling exists (see Week 4
  notes), repeat this experiment on real ICU data.
- Use `generate()`'s synthetic output to augment Module 1 (clinical)
  and Module 3 (ECG) training sets for their own rare-event class
  imbalance.
- **Week 6:** fusion layer - load all 5 modules' checkpoints, extract
  each branch's feature vector (`return_features=True` / `mu`),
  concatenate, train one small fusion classifier on top. Also the
  multimodal LLM cross-check layer.
- **Week 7:** AutoML search, AutoGen orchestration, Streamlit dashboard.

### Files Delivered This Week
- `src/vae_models.py`
- `notebooks/Week5_Module5_VAE.ipynb`
- (`src/vitals_data.py` reused from Week 4, no changes needed)

### How to Run
1. Ensure Week 4's files are already in place (`vitals_data.py` is a
   shared dependency).
2. Open `notebooks/Week5_Module5_VAE.ipynb` in Colab, Run All.
3. Uses the synthetic fallback by default (`mimic_dir=None`) since
   real-data deterioration labels aren't wired up yet - see notebook's
   data-loading markdown cell for why.
