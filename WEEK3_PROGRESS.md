# Week 3 Progress Report
## Multi-Modal Deep Learning Framework for Real-Time Cardiac Risk Prediction in ICU

### Module Covered This Week
**Module 3 - ECG Waveform Analysis** (CNN, Kernels, Padding, Stride, 1D-CNN,
Transfer Learning)

### What Was Completed
- Wired in the **first real dataset** for the project: the **MIT-BIH
  Arrhythmia Database** (PhysioNet, `mitdb`), via `wfdb`. This is open
  access (no PhysioNet credentialing required), unlike MIMIC.
- `src/ecg_data.py` - downloads MIT-BIH records, uses the annotation
  files to locate each heartbeat's R-peak and its label, and cuts a
  fixed-length window per beat -> one row per heartbeat, Normal (0) vs
  Abnormal/arrhythmic (1). Falls back automatically to a synthetic but
  ECG-shaped beat generator (P/QRS/T-wave templates) if the real
  download fails in a given session, so the pipeline is never blocked.
- `src/ecg_models.py` - **`ECG1DCNN`**: a 3-layer 1D convolutional stack
  that demonstrates all of this module's theory topics directly in the
  architecture:
  - **Kernels:** wide kernel (15) in the first conv layer for whole-beat
    morphology, narrowing to kernel 9 then 5 in later layers for
    finer detail.
  - **Padding:** `same`-style padding on the first two conv layers so
    the P/QRS/T span isn't clipped at the window edges.
  - **Stride:** the third conv layer downsamples via stride 2 directly
    (rather than another pooling layer), showing stride as an
    alternative downsampling mechanism.
  - Reuses the Module 2 stability pattern (He-init, BatchNorm, Dropout
    0.3) inside this branch too.
  - Adds a lightweight **self-attention** block after the conv stack so
    the model can up-weight the QRS-complex region relative to flatter,
    noisier segments of the beat.
- Notebook follows the same pattern as Weeks 1-2: Drive mount, smart
  checkpoint loading, training loop with saved loss/AUC plot, final
  validation metrics, classification report, and confusion matrix.

### Results
*(Fill in after your Colab run: real vs synthetic data used, best
val AUC, and a note on the confusion matrix - e.g. whether false
negatives on Abnormal beats are rare, since that's the clinically
important error to minimize.)*

### What's Left (Upcoming Weeks)
- **Week 4 - Module 4:** BiLSTM for temporal vitals forecasting
  (heart rate, SpO2) - still needs MIMIC-III/IV, so start the PhysioNet
  credentialing request now if not already submitted, since it can take
  a few days to be approved.
- **Week 5 - Module 5:** BiLSTM-VAE for generative anomaly detection and
  synthetic data augmentation for rare cardiac events.
- **Week 6:** Fuse Module 1/2 (clinical), Module 3 (ECG), and Module 4/5
  (temporal/anomaly) feature vectors into one risk score; add the
  multimodal LLM cross-check layer (ECG image read vs tabular/text read).
- **Week 7:** AutoML search over architecture/hyperparameters; AutoGen
  orchestration across pipeline stages; Streamlit demo dashboard.
- **Final week:** Ablations, metrics table, final report and slides.

### Files Delivered This Week
- `src/ecg_data.py` - MIT-BIH loader with synthetic ECG-shaped fallback
- `src/ecg_models.py` - `ECG1DCNN` (1D-CNN + self-attention)
- `notebooks/Week3_Module3_ECG_CNN.ipynb` - Colab notebook, ready to run

### How to Run
1. Open `notebooks/Week3_Module3_ECG_CNN.ipynb` in Colab (folder must
   already be `MyDrive/Cardiac_Risk_Prediction/`).
2. Runtime -> Run all. It installs `wfdb`, downloads a small MIT-BIH
   record subset (or falls back to synthetic data if that fails),
   trains the CNN, and saves plots/checkpoint back to Drive.
3. Check the printed line right after data loading - it tells you
   whether you got **real MIT-BIH beats** or the **synthetic fallback**,
   which matters for how you present the results.
