# Verification Report - Modules 1 through 5

Full pass across all 5 modules: compiled every file, ran real forward
passes, ran real multi-epoch training loops (not just a single step),
checked gradient flow on every parameter, and checked for the specific
failure modes that don't show up until you hit real/uneven data. This
document records exactly what was found and what was changed.

---

## Bugs found and fixed

### 1. `src/ecg_models.py` - ECG1DCNN classifier ran on the wrong tensor (CRITICAL)
**Before:** `forward()` computed `pooled` (the correct, globally-averaged
64-dim feature vector) but then called `self.classifier(features)` -
the pre-pooling tensor, shape `(batch, 64, 27)` - instead of `pooled`.
Since the classifier starts with `Flatten -> Linear(64, 32)`, flattening
`(batch, 64, 27)` produces `(batch, 1728)`, which cannot multiply
against a `Linear(64, 32)` weight matrix.

**Impact:** This is not a subtle numerical issue - `ECG1DCNN.forward()`
would raise `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
on the very first call. Module 3's notebook would have crashed
immediately in Colab, before a single training step ran.

**Fix:** Changed `self.classifier(features)` to `self.classifier(pooled)`.

**Verified:** Forward pass, `return_features=True` consistency, and a
real training step (loss computed, `.backward()`, `.step()`) all run
cleanly now. Confirmed no parameter is left without a gradient.

---

### 2. All 5 training DataLoaders - BatchNorm crash on dataset sizes not divisible by batch_size (LATENT)
**Correction to this section, made after a follow-up review:** the
original version of this report claimed "every model in this project
uses BatchNorm1d in training mode" - that was **wrong for Module 5**.
`vae_models.py` never contained `BatchNorm1d` anywhere; only Modules
1-4 (the MLP, CNN, and the original BiLSTM) did. The `drop_last=True`
fix below is correctly scoped to those - it was never actually needed
for Module 5's crash-avoidance reasons, though it remains harmless
there.

**Before:** every `train_loader = DataLoader(..., batch_size=32,
shuffle=True)` across Modules 1-4 had no `drop_last` argument, and each
of those models uses `BatchNorm1d` in training mode. `BatchNorm1d` in
train mode requires more than 1 sample per batch to compute a batch
statistic - it raises `RuntimeError: Expected more than 1 value per
channel when training` if the final mini-batch of an epoch happens to
contain exactly 1 sample.

**Why it didn't show up yet:** every dataset size chosen so far
happens to divide evenly into batches of 32 with no leftover-of-1
batch. This is luck, not correctness.

**Fix:** Added `drop_last=True` to every `train_loader` in Weeks 1-5
(harmless where not strictly required, i.e. Module 5). Confirmed
`val_loader`s don't need it - `BatchNorm1d` in `eval()` mode uses
running statistics and doesn't crash on batch size 1.

**Verified:** Reproduced the exact crash on a 641-sample dataset,
confirmed the fix resolves it.

---

### 2b. Modules 4 & 5 redesigned: LayerNorm instead of BatchNorm, full-sequence pooling, adapter layers
Following a design review, `vitals_models.py` (Module 4) and
`vae_models.py` (Module 5) were substantively revised:

- **LayerNorm replaces BatchNorm1d in Module 4** (Module 5 never had
  BatchNorm - see the correction above). LayerNorm normalizes per-sample
  across features rather than per-batch across samples, so it has zero
  batch-size dependency - it does not crash on a batch of 1, which
  removes the root cause `drop_last=True` was working around for
  Module 4 specifically (not just papering over the symptom).
  **Verified:** a batch of size 1 in `.train()` mode now runs cleanly
  through both `VitalsBiLSTM` and `VitalsBiLSTMVAE` with no exception.

- **Full-sequence mean-pooling replaces final-hidden-state pooling.**
  Both models previously pooled only `h_n` (the LSTM's final timestep
  hidden state) from the last BiLSTM layer, discarding every other
  timestep's representation. Both now mean-pool the FULL
  `return_sequences` output across the entire time axis, so the
  resulting feature vector actually reflects the whole input window,
  not just its boundary.

- **Non-linear adapter layers added around the VAE's latent bottleneck.**
  `VitalsVAEEncoder` previously projected straight from the pooled
  BiLSTM output to `mu`/`log_var` with two bare `Linear` layers;
  `VitalsVAEDecoder` projected straight from `z` into the decoder
  BiLSTM's input dimensionality with one bare `Linear`. Both now route
  through a `Linear -> LeakyReLU -> Dropout` adapter block first, giving
  the model non-linear capacity to reshape the representation right at
  the point it crosses the latent bottleneck, instead of forcing a
  single linear map to do that job on both sides.

**Verified with a real re-run (40 epochs, same synthetic data, same
seed):** total loss still converges cleanly (88.15 -> 32.26), and the
anomaly-detection signal actually **improved** - normal-vs-abnormal
reconstruction-error separation went from 17.28x (v1) to **24.34x**
(v2). This is a genuine quantitative improvement from the redesign, not
just a style change.



## Findings documented but NOT changed (flagged for your decision)

### 3. Week 1 notebook: `load_realistic_clinical_metadata()` is defined but never called (dead code)
Cell 4 of `Week1_Module1_MLP.ipynb` defines a second, noisier data
generator (adds an age x blood-pressure interaction term and Gaussian
noise to the risk score, meant to make the classification task
non-trivial) - but the model is trained and evaluated entirely on the
original, noise-free `load_clinical_metadata()` from Cell 1. The
noisier function is never invoked.

**This directly explains** the ~0.999 validation AUC your own README
already flags as a known limitation ("near-linearly-separable synthetic
dataset"). It looks like an attempt to fix that exact issue that got
written but never wired in.

**Not changed**, because swapping the data source changes your
reported Week 1 results - that's your call, not something to silently
alter. If you want the harder/more realistic dataset actually used,
say so and I'll wire it in and re-verify convergence.

### 4. Weeks 1 & 2 notebooks don't import from `src/`, despite `PIPELINE_STRATEGY.md` saying they should
`Week1_Module1_MLP.ipynb` and `Week2_Module2_Stability.ipynb` both
define their data-loading function and model classes **inline in the
notebook**, rather than importing `load_clinical_metadata` from
`src/data_utils.py` and `BaselineMLP`/`ClinicalMLP` from `src/models.py`
- contradicting your own strategy doc's stated design ("every module's
real logic lives in `.py` files ... never written directly inside a
notebook").

**`Week3_Module3_ECG_CNN.ipynb` onward (3, 4, 5) already does this
correctly** - only Weeks 1-2 have the inline/duplicated pattern. This
is not evenly distributed across the project; it's specifically a
Week 1-2 leftover.

**Checked whether this actually breaks anything:** it currently does
NOT. I byte-compared the inline `ClinicalMLP` against `src/models.py`'s
`ClinicalMLP` - identical architecture, identical `state_dict()` key
structure - and confirmed a checkpoint trained from the notebook's
inline class loads cleanly into the `src/models.py` class. So Week 6
fusion will work today.

**The risk:** this only works because the two definitions currently
happen to match exactly. If either one drifts even slightly in a
future edit (e.g. you tweak dropout rate in the notebook but forget to
mirror it in `src/models.py`), the checkpoint will silently load with
mismatched architecture assumptions, or fail to load at all, and you
won't find out until Week 6. **Not changed** here since it's an
architectural decision (rewrite Weeks 1-2 to import from `src/`,
matching Weeks 3-5) - say the word and I'll do it.

---

## What was actually run and verified (not just read)

| Module | Forward pass | Gradient flow (every param) | Multi-epoch convergence | Loss is finite (no NaN) |
|---|---|---|---|---|
| 1 - BaselineMLP | Yes | Yes | Yes (0.706 -> 0.542 / 20 epochs) | Yes |
| 2 - ClinicalMLP | Yes | Yes | Yes (1.005 -> 0.566 / 20 epochs) | Yes |
| 3 - ECG1DCNN (after fix) | Yes | Yes | Yes (0.793 -> 0.321 / 15 epochs) | Yes |
| 4 - VitalsBiLSTM (v2 - LayerNorm, full-sequence pooling) | Yes | Yes | Yes (10.42 -> 5.24 / 30 epochs) | Yes |
| 5 - VitalsBiLSTMVAE (v2 - adapters, full-sequence pooling) | Yes | Yes | Yes (88.15 -> 32.26 / 40 epochs) | Yes |

**Module 5 anomaly-separation, v1 vs v2:** normal-vs-abnormal
reconstruction-error separation improved from **17.28x (v1)** to
**24.34x (v2)** after the redesign - a real, measured improvement, not
an assumption.

All numbers above are from real runs in this environment (PyTorch 2.14
CPU), on the existing synthetic-fallback data paths, with fixed random
seeds. Not simulated or estimated.

---

## Recommendation for your report

State plainly that a pre-Week-6 verification pass found and fixed one
critical bug (Module 3's classifier bug, which would have blocked that
week's Colab run entirely) and one latent robustness bug (BatchNorm/
batch-size-1 crash risk, which hadn't triggered yet but would have on
real data). That's a stronger, more credible story for a review than
claiming everything worked first try - and it's the honest one.
