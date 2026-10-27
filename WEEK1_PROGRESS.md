# Week 1 Progress Report
## Multi-Modal Deep Learning Framework for Real-Time Cardiac Risk Prediction in ICU

### Module Covered This Week
**Module 1 - Clinical Data Classifier** (ANN / MLP, Activations, Optimizers)

### What Was Completed
- Built an MLP (Artificial Neural Network) that processes structured patient
  metadata: age, BMI, systolic blood pressure, cholesterol.
- Hidden layers use **ReLU** activation; output layer uses **Sigmoid** to
  produce a probability-style baseline health/risk score in [0, 1].
- Model is trained using the **Adam optimizer** with early stopping on
  validation AUC.
- Data pipeline (`data_utils.py`) implemented with a dataset loader that
  is dataset-source-agnostic: it will use a real public clinical dataset
  when pointed at one, and otherwise generates a clinically-plausible
  synthetic dataset of the same shape, so the pipeline is never blocked
  by dataset access/download issues.
- Preprocessing (feature scaling via `StandardScaler`) implemented and
  validated.
- End-to-end run validated in Colab: data load -> preprocess -> train ->
  plot loss/AUC curves -> save model checkpoint to Drive.

### Results (this run)
- Training converges within ~20-30 epochs with early stopping.
- Validation AUC and loss curves saved to `outputs/week1_loss_auc.png`.
- Model checkpoint saved to `checkpoints/module1_mlp.keras`.

### What's Left (Upcoming Weeks)
- **Week 2 - Module 2:** Apply He Initialization, Batch Normalization, and
  Dropout (0.3) to this MLP for stability/regularization; run an ablation
  comparison (baseline vs stabilized) to quantify the improvement.
- **Week 3-4 - Module 3:** Build the 1D-CNN + lightweight self-attention
  branch for raw ECG waveform analysis (P/QRS/T wave feature extraction).
- **Week 5 (updated) - Modules 4 & 5:** Build the BiLSTM-VAE branch for
  temporal vitals forecasting and generative anomaly detection /
  synthetic data augmentation for rare cardiac events.
- **Final weeks:** Fuse all three branches into a unified multi-modal
  risk-scoring pipeline; build a Streamlit demo dashboard; final report
  and evaluation.

### Files Delivered This Week
- `src/data_utils.py` - shared data loading utilities (all modules)
- `src/module1_mlp.py` - Module 1 MLP implementation
- `notebooks/week1_module1.ipynb` - Colab notebook, ready to run
- `requirements.txt` - dependencies

### How to Run
1. Upload/extract this folder to Google Drive at
   `MyDrive/Cardiac_Risk_Project/`.
2. Open `notebooks/week1_module1.ipynb` in Colab.
3. Run all cells top to bottom - it mounts Drive, installs dependencies,
   trains the model, saves plots and a checkpoint back to Drive.
