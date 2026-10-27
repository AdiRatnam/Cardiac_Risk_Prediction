# Architecture Documentation
## Multi-Modal Deep Learning Framework for Real-Time Cardiac Risk Prediction in ICU

This document explains the entire system end to end: the problem, the
data flow, every module's internal design and why it's built that way,
how the pieces are meant to connect, and what's real vs. synthetic
today. Read this alongside `src/` - every class and function named
here exists in the codebase and has been run and verified (see
`VERIFICATION_REPORT.md` for the actual test log).

---

## 1. The Problem This Architecture Solves

ICU monitoring today looks at three data streams **separately**:
a patient's static clinical profile, their ECG waveform, and their
vitals trend over time. Each stream alone gives an incomplete picture,
which causes both delayed interventions (a signal one stream would
catch, the others miss) and high false-alarm rates (a stream flags
something the others would have contextualized as normal). On top of
that, the events that actually matter - true cardiac deterioration -
are rare, so a model trained naively on raw data barely sees them.

This architecture addresses both problems directly:
- **Fusion**: three independent branches, one per data modality, each
  producing a fixed-size feature vector; a fusion layer (Week 6)
  combines all three into one risk score - so the model reasons over
  all three streams at once, the way a clinician would.
- **Generative rarity handling**: a VAE branch is trained to recognize
  what "normal" looks like, flags deviations from it as anomalies via
  reconstruction error, and can also *generate* new realistic
  rare-event examples to balance the other branches' training data.

---

## 2. High-Level Data Flow

```
                         ┌─────────────────────┐
   Clinical metadata --> │  Module 1/2: MLP     │ --> risk score (0-1)
   (age, BMI, BP,        │  BaselineMLP /       │     + 32-dim feature
   cholesterol, max HR)  │  ClinicalMLP         │       (planned)
                         └─────────────────────┘
                                                         \
                         ┌─────────────────────┐          \
   Raw ECG waveform  --> │  Module 3: 1D-CNN     │ --> risk score (0-1)   \
   (216-sample window,   │  + Self-Attention     │     + 64-dim feature    >--> FUSION (Week 6)
   MIT-BIH beats)        │  ECG1DCNN             │       vector           /      --> combined
                         └─────────────────────┘                        /        risk score
                                                                        /
                         ┌─────────────────────┐          ┌──────────/──────┐
   ICU vitals window --> │  Module 4: BiLSTM     │ ------> │  64-dim feature │
   (24h HR + SpO2,        │  forecaster           │         │  vector          │
   MIMIC-IV demo)        │  VitalsBiLSTM         │         └─────────────────┘
                         └─────────────────────┘
                                    |
                                    | (shares vitals_data.py)
                                    v
                         ┌─────────────────────┐
                         │  Module 5: BiLSTM-VAE │ --> reconstruction error
                         │  VitalsBiLSTMVAE      │       (anomaly score)
                         │                       │ --> mu (16-dim latent,
                         │                       │       feature vector)
                         │                       │ --> synthetic sequences
                         └─────────────────────┘     (data augmentation for
                                                       Modules 1 & 3)
```

Every branch is trained and checkpointed **independently** - Module 4
does not need Module 1 to exist, Module 5 does not need Module 3 to
exist. This is deliberate (see Section 8, Pipeline Strategy) so each
week's work is self-contained and the eventual fusion step is purely
additive: load 5 checkpoints, pull 5 feature vectors, concatenate,
train one small classifier on top.

---

## 3. Module 1 & 2 - Clinical Metadata Classifier

**Files:** `src/data_utils.py`, `src/models.py`

### Data (`load_clinical_metadata`)
Synthetic clinically-plausible generator (real MIMIC integration
planned, pending credentialing): five features - age, BMI, systolic
BP, cholesterol, max heart rate - combined into a risk score via a
weighted linear formula, thresholded at 0.5 to produce a binary label.
Seeded (seed=42) so every module/week that touches this data sees the
identical dataset - critical for the Module 1 vs Module 2 ablation to
be a fair apples-to-apples comparison.

### `BaselineMLP` - Module 1 in isolation
```
Linear(input_dim, 64) -> LeakyReLU(0.1) ->
Linear(64, 32)        -> LeakyReLU(0.1) ->
Linear(32, 1)         -> Sigmoid
```
Plain MLP: PyTorch's default weight init, no BatchNorm, no Dropout.
Exists specifically so Module 2's ablation has a genuine "before" to
compare against.

### `ClinicalMLP` - Module 1 + Module 2's stability additions
```
Linear(input_dim, 64) -> BatchNorm1d(64) -> LeakyReLU(0.1) -> Dropout(0.3) ->
Linear(64, 32)        -> BatchNorm1d(32) -> LeakyReLU(0.1) -> Dropout(0.3) ->
Linear(32, 1)         -> Sigmoid
```
Same depth/width as `BaselineMLP`, with He Initialization
(`kaiming_normal_`, matched to LeakyReLU's negative slope), BatchNorm1d
after every Linear, and Dropout(0.3). This is the architecture the
Week 1 notebook actually trains and checkpoints.

**Why BatchNorm is safe here but not for Modules 4/5:** this branch
processes flat tabular vectors in large fixed batches - BatchNorm's
per-batch statistic is stable and appropriate. Modules 4/5 process
sequences and were redesigned to use LayerNorm instead (Section 6) -
different modality, different normalization choice, not a
one-size-fits-all default.

**Known finding (see `VERIFICATION_REPORT.md`):** Week 1's notebook
defines a second, harder/noisier data generator
(`load_realistic_clinical_metadata`) that is never actually called -
the model trains on the easy, near-linearly-separable version, which
explains the ~0.999 validation AUC. Flagged, not silently fixed.

---

## 4. Module 3 - ECG Waveform Analyzer

**Files:** `src/ecg_data.py`, `src/ecg_models.py`

### Data (`load_ecg_beats`)
Real-data-first: attempts to download MIT-BIH Arrhythmia Database
records (open-access, no credentialing) via `wfdb`, locates each
annotated R-peak, cuts a 216-sample window centered on it, and labels
it Normal (`N`/`.`) or Abnormal (any arrhythmia symbol). On any failure
(no network, `wfdb` not installed), falls back to a synthetic
sum-of-Gaussians ECG generator that actually produces P-QRS-T-shaped
beats (not random noise), with the abnormal variant widening/
deforming the QRS complex and shifting the T-wave - mimicking the real
clinical signature of an ectopic beat.

### `ECG1DCNN` + `SelfAttention1D`
```
Conv1d(1, 16, kernel=15, pad=7)  -> BN -> LeakyReLU -> MaxPool(2)   # 216 -> 108
Conv1d(16, 32, kernel=9, pad=4)  -> BN -> LeakyReLU -> MaxPool(2)   # 108 -> 54
Conv1d(32, 64, kernel=5, stride=2, pad=2) -> BN -> LeakyReLU         # 54  -> 27
SelfAttention1D(channels=64)                                        # 27  -> 27
GlobalAveragePooling -> pooled (64-dim, THIS branch's feature vector)
Dropout -> Linear(64,32) -> LeakyReLU -> Dropout -> Linear(32,1) -> Sigmoid
```
Kernel sizes shrink wide-to-narrow (15 -> 9 -> 5) to demonstrate
capturing whole-beat morphology first, then finer local detail. The
third conv layer downsamples via `stride=2` instead of pooling,
specifically to demonstrate stride as an alternative to pooling (a
Module 3 theory topic). `SelfAttention1D` lets the model up-weight the
QRS-complex region over flatter/noisier baseline segments, with a
learned `gamma` residual weight starting at 0 (SAGAN-style) so training
begins as a plain CNN and gradually learns how much attention to mix
in.

**Bug found and fixed here (see `VERIFICATION_REPORT.md`):** the
classifier originally ran on the pre-pooling `features` tensor instead
of the pooled 64-dim vector - a shape mismatch that crashed the very
first forward pass. Now fixed and verified with a real training step.

---

## 5. Module 4 - Temporal Vital Forecasting

**Files:** `src/vitals_data.py`, `src/vitals_models.py`

### Data (`load_vitals_sequences`, shared with Module 5)
Real-data-first: given a MIMIC-IV demo folder, resamples Heart Rate
(item 220045) and SpO2 (item 220277) to hourly readings per ICU stay
from `icu/chartevents.csv`, forward/back-filling small gaps. Falls back
to a synthetic generator producing autocorrelated HR/SpO2 sequences
with an occasional simulated deterioration event (HR rising, SpO2
falling over a random onset window) - this is also where the
normal/abnormal `labels` array Module 5 depends on comes from. Real
MIMIC data currently returns `labels=None` (no deterioration/sepsis
labeling wired up yet - flagged as a next step, not silently assumed).

### `VitalsBiLSTM`
```
LSTM(2, 64, bidirectional) -> Dropout ->
LSTM(128, 32, bidirectional) -> out: (batch, seq_len, 64)   # EVERY timestep
mean-pool over the FULL time axis -> (batch, 64)             # not just h_n
LayerNorm(64)                                                 # this branch's
                                                                # feature vector
Dropout -> Linear(64,32) -> LeakyReLU -> Linear(32, horizon*2)
reshape -> (batch, horizon, 2)
```
**Design decision, made after review (see Section 6 below and
`VERIFICATION_REPORT.md`):** originally pooled only the LSTM's final
hidden state (`h_n`), which discards every other timestep. Now
mean-pools the entire output sequence, so the feature vector genuinely
reflects the whole input window. Uses LayerNorm (not BatchNorm) since
it has no batch-size dependency, which matters for a sequence model
where batch composition is less predictable than tabular data.

---

## 6. Module 5 - Generative Anomaly Detection

**Files:** `src/vae_models.py` (reuses `vitals_data.py`)

The BiLSTM is used as **both** the VAE's encoder and decoder - one
architecture, two directions:

### `VitalsVAEEncoder`
```
LSTM(2, 64, bidirectional) -> out: (batch, seq_len, 128)   # EVERY timestep
mean-pool over the FULL time axis -> (batch, 128)
LayerNorm(128)
Adapter: Linear(128, 64) -> LeakyReLU -> Dropout            # non-linear bridge
Linear(64, 16) -> mu
Linear(64, 16) -> log_var
```

### `VitalsVAEDecoder`
```
z (batch, 16)
Adapter: Linear(16, 64) -> LeakyReLU -> Dropout              # symmetric bridge
repeat across output_len (seq_len + horizon = 30 timesteps)
LSTM(64, 64, bidirectional) -> Linear(128, 2) -> reconstruction
```

### Reparameterization + loss
`z = mu + exp(0.5*log_var) * eps`, `eps ~ N(0,I)` - the standard trick
that lets gradients flow through a stochastic latent variable.
`vae_loss()` = reconstruction MSE (summed over time+features, meaned
over batch) + `0.001 * KL divergence` against a standard normal prior
(beta-VAE-style weighting, keeping the latent space well-behaved
without letting KL dominate early training).

**Design decisions, made after review:**
- **Adapter layers on both sides of the latent bottleneck.** The
  original version projected straight from the pooled BiLSTM output to
  `mu`/`log_var` with two bare `Linear` layers, and straight from `z`
  into the decoder's LSTM input with one bare `Linear`. Both now route
  through `Linear -> LeakyReLU -> Dropout` first - giving the model
  actual non-linear capacity to reshape the representation exactly
  where it's least "sequence-shaped" (crossing the bottleneck),
  instead of forcing a single linear map to do that job.
- **Full-sequence pooling**, same reasoning as Module 4.
- **No BatchNorm anywhere in this file** (correcting an earlier claim
  in `VERIFICATION_REPORT.md` that it did) - LayerNorm is used instead,
  consistent with Module 4.

**Critical training requirement:** must be trained **only on normal
(`labels==0`) sequences**. A VAE trained on a mix learns to reconstruct
abnormal patterns too, which erases the reconstruction-error spike the
whole anomaly-detection mechanism depends on. `Week5_Module5_VAE.ipynb`
filters for this explicitly and holds out the abnormal sequences purely
for evaluating whether the trained model's error actually separates
them.

**Measured result (real run, not assumed):** after 40 epochs on
synthetic data, normal-mean reconstruction error was 0.43 vs.
abnormal-mean 10.56 - a **24.34x separation**. This is quantitative
evidence the design works, not just that the code runs.

`generate()` samples `z ~ N(0,I)` and decodes it standalone - this is
how Module 5 produces synthetic rare-event sequences to augment Module
1 and Module 3's training sets.

---

## 7. Cross-Module Design Conventions

These conventions are what make Week 6 fusion possible without
touching any of the five modules' own code:

1. **`return_features=True` on every model's `forward()`.**
   `ECG1DCNN`, `VitalsBiLSTM`, and `VitalsBiLSTMVAE` (via `mu`) all
   support this - calling with it returns `(prediction, feature_vector)`
   instead of just the prediction. `BaselineMLP`/`ClinicalMLP` don't
   yet (see `NEXT_STEPS.md`) - this is the one convention not yet
   uniformly applied across all 5.

2. **Real-data-first, synthetic-fallback-always.** Every data loader
   (`load_clinical_metadata`, `load_ecg_beats`,
   `load_vitals_sequences`) tries a real public/open dataset first and
   transparently falls back to a synthetic-but-realistically-shaped
   generator on any failure, printing which source it used. No
   notebook is ever blocked by a network issue or a pending
   credentialing approval.

3. **Smart checkpoint loading.** Every notebook checks for an existing
   `checkpoints/module<N>_*.pt` file before training - if found, it
   loads the weights and skips retraining. This is what makes
   re-running a notebook (or building the fusion notebook on top)
   cheap.

4. **He Initialization + LeakyReLU(0.1) + Dropout(0.3)** as the
   consistent stability pattern across every trainable layer in every
   module - the Module 2 pattern, applied project-wide, with the one
   deliberate deviation being BatchNorm (Modules 1-3, tabular/spatial
   data) vs. LayerNorm (Modules 4-5, sequential data) - see Section 6.

---

## 8. Pipeline Strategy - Why Notebooks Never Get Merged

(Full original rationale in the project's `PIPELINE_STRATEGY.md`.)
Every module's real logic lives in `src/`, never written directly in a
notebook cell (Modules 3-5 follow this correctly; Modules 1-2 currently
have a documented exception - see `VERIFICATION_REPORT.md` and
`NEXT_STEPS.md`). Each week's notebook trains one module and writes one
checkpoint; checkpoints accumulate side by side in `checkpoints/` and
are never overwritten by a later week. Week 6's fusion notebook will be
purely additive: import every model class from `src/`, load every
checkpoint, run each model on the same patient's data with
`return_features=True`, concatenate the resulting vectors, and train
one small classifier on top - no earlier notebook is ever reopened or
edited.

---

## 9. What's Real vs. Synthetic Today

| Module | Real data source | Status |
|---|---|---|
| 1/2 - Clinical | MIMIC (planned) | Synthetic only so far |
| 3 - ECG | MIT-BIH (open-access, no credentialing) | Wired in, falls back to synthetic if `wfdb`/network unavailable |
| 4 - Vitals forecasting | MIMIC-IV demo (open-access) | Wired in (`load_mimic_vitals_sequences`), falls back to synthetic |
| 5 - Anomaly detection | Same as Module 4 | Real data has no deterioration label yet - synthetic labels used for the normal/abnormal split |

Every reported metric in this project so far is on **synthetic data**
unless a notebook's printed output explicitly says otherwise - stated
plainly here so it's never ambiguous in a report or presentation.

---

## 10. What's Not Built Yet

- **Week 6 - Fusion layer**: load all 5 checkpoints, concatenate
  feature vectors, train a small classifier on top. Also the
  multimodal LLM cross-check (image-based ECG read vs. text-based
  vitals+metadata read, as a QA layer).
- **Week 7 - GenAI/AutoML addendum**: AutoML hyperparameter search to
  justify Modules 1-3's architecture choices; AutoGen orchestration
  across pipeline stages; Streamlit demo dashboard.
- **Real deterioration/sepsis labels for Module 5** on MIMIC data
  (needs SOFA/ICD-based cohort extraction on top of raw vitals).
- See `NEXT_STEPS.md` for the full prioritized list.
