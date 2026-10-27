"""
vitals_data.py
Module 4/5 - Temporal Vital Forecasting & Generative Anomaly Detection:
data loading.

Real data source: MIMIC-III/IV ICU vitals (Heart Rate, SpO2 over time),
per PATIENTS/ICUSTAYS/CHARTEVENTS. Requires PhysioNet credentialing
(pending per README Dataset Strategy table) - so, mirroring the pattern
in ecg_data.py, this module tries a real MIMIC folder first and falls
back to a synthetic-but-physiologically-shaped vitals generator if the
folder isn't present, so training/demo is never blocked on approval.

Works with either MIMIC-III (icustay_id) or MIMIC-IV (stay_id) demo
folder layouts - pass whichever `mimic_dir` you have.
"""

import os
import numpy as np

SEQ_LEN = 24            # hours of input history
FORECAST_HORIZON = 6    # hours to forecast ahead
N_FEATURES = 2          # heart rate, SpO2


def _try_load_mimic_vitals(mimic_dir, seq_len=SEQ_LEN, horizon=FORECAST_HORIZON,
                            hr_itemid=220045, spo2_itemid=220277):
    """
    Attempts a real MIMIC-IV-demo-style load from `mimic_dir/icu/chartevents.csv`
    + `icu/icustays.csv`, with optional deterioration labeling from
    `hosp/diagnoses_icd.csv`. Returns (X, y, labels) as numpy arrays.
    """
    import pandas as pd

    # Resolve paths for MIMIC-IV (icu/ and hosp/) or MIMIC-III (flat)
    chart_path = os.path.join(mimic_dir, "icu", "chartevents.csv")
    stays_path = os.path.join(mimic_dir, "icu", "icustays.csv")
    diag_path = os.path.join(mimic_dir, "hosp", "diagnoses_icd.csv")

    if not (os.path.exists(chart_path) and os.path.exists(stays_path)):
        # Try flat directory structure (MIMIC-III format)
        chart_path = os.path.join(mimic_dir, "CHARTEVENTS.csv")
        stays_path = os.path.join(mimic_dir, "ICUSTAYS.csv")
        diag_path = os.path.join(mimic_dir, "DIAGNOSES_ICD.csv")
        if not (os.path.exists(chart_path) and os.path.exists(stays_path)):
            raise FileNotFoundError(f"No chartevents/icustays found under {mimic_dir}")

    stays_df = pd.read_csv(stays_path)
    stay_col = "stay_id" if "stay_id" in stays_df.columns else "icustay_id"
    hadm_col = "hadm_id" if "hadm_id" in stays_df.columns else "HADM_ID"

    # Extract deterioration / cardiac risk diagnoses if available
    abnormal_hadms = set()
    if os.path.exists(diag_path):
        try:
            diag_df = pd.read_csv(diag_path, dtype={"icd_code": str})
            diag_hadm_col = "hadm_id" if "hadm_id" in diag_df.columns else "HADM_ID"
            code_col = "icd_code" if "icd_code" in diag_df.columns else "ICD9_CODE"

            # Match cardiac arrest, myocardial infarction, heart failure, arrhythmias, cardiogenic shock
            cardiac_prefixes = (
                "410", "411", "412", "414", "427", "428", "78551", "99591", "99592",
                "I21", "I22", "I25", "I46", "I47", "I48", "I49", "I50", "R570", "R652"
            )
            matched = diag_df[diag_df[code_col].astype(str).str.startswith(cardiac_prefixes, na=False)]
            abnormal_hadms = set(matched[diag_hadm_col].dropna().unique())
        except Exception:
            abnormal_hadms = set()

    stay_to_hadm = {}
    if hadm_col in stays_df.columns and stay_col in stays_df.columns:
        stay_to_hadm = dict(zip(stays_df[stay_col], stays_df[hadm_col]))

    total_len = seq_len + horizon
    usecols = [stay_col, "charttime" if "charttime" in pd.read_csv(chart_path, nrows=1).columns else "CHARTTIME",
               "itemid" if "itemid" in pd.read_csv(chart_path, nrows=1).columns else "ITEMID",
               "valuenum" if "valuenum" in pd.read_csv(chart_path, nrows=1).columns else "VALUENUM"]

    time_col, item_col, val_col = usecols[1], usecols[2], usecols[3]

    chunks = []
    for chunk in pd.read_csv(chart_path, usecols=usecols, chunksize=200_000):
        chunk = chunk[chunk[item_col].isin([hr_itemid, spo2_itemid])]
        if len(chunk):
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError("No HR/SpO2 rows found for given item IDs")

    events = pd.concat(chunks, ignore_index=True)
    events[time_col] = pd.to_datetime(events[time_col])
    events = events.dropna(subset=[val_col])

    X_list, y_list, labels_list = [], [], []
    for stay_id, group in events.groupby(stay_col):
        hr = (group[group[item_col] == hr_itemid]
              .set_index(time_col)[val_col].resample("1h").mean())
        spo2 = (group[group[item_col] == spo2_itemid]
                .set_index(time_col)[val_col].resample("1h").mean())
        if len(hr) < total_len or len(spo2) < total_len:
            continue
        merged = pd.concat([hr, spo2], axis=1)
        merged.columns = ["hr", "spo2"]
        merged = merged.ffill().bfill()
        arr = merged.to_numpy()

        hadm = stay_to_hadm.get(stay_id, None)
        is_abnormal = 1.0 if hadm in abnormal_hadms else 0.0

        # Physiological fallback if ICD not available: sustained severe tachycardia or hypoxia
        if not abnormal_hadms:
            if (arr[:, 0] > 115).sum() >= 4 or (arr[:, 1] < 90).sum() >= 2:
                is_abnormal = 1.0

        # Extract continuous sliding windows with stride 6
        for start_idx in range(0, len(arr) - total_len + 1, 6):
            window = arr[start_idx:start_idx + total_len]
            if not np.isnan(window).any():
                X_list.append(window[:seq_len])
                y_list.append(window[seq_len:])
                labels_list.append(is_abnormal)

    if not X_list:
        raise RuntimeError("No ICU stays had sufficient continuous HR/SpO2 coverage")

    return (np.stack(X_list).astype(np.float32),
            np.stack(y_list).astype(np.float32),
            np.array(labels_list, dtype=np.float32))


def _synthetic_vitals_sequence(rng, seq_len, horizon, deterioration_prob=0.2):
    """
    One synthetic ICU vitals sequence: heart rate (bpm) and SpO2 (%) with
    realistic autocorrelated drift, plus an occasional simulated
    deterioration event (HR rising, SpO2 falling) so the dataset has a
    genuine abnormal-vs-normal distinction for later anomaly-detection use.
    """
    total_len = seq_len + horizon
    hr = 75 + np.cumsum(rng.normal(0, 0.8, total_len))
    spo2 = 97 + np.cumsum(rng.normal(0, 0.15, total_len))

    is_deteriorating = rng.random() < deterioration_prob
    if is_deteriorating:
        onset = rng.integers(seq_len // 2, total_len - 3)
        hr[onset:] += np.linspace(0, rng.uniform(15, 30), total_len - onset)
        spo2[onset:] -= np.linspace(0, rng.uniform(3, 8), total_len - onset)

    hr = hr.clip(40, 180)
    spo2 = spo2.clip(80, 100)
    seq = np.stack([hr, spo2], axis=-1).astype(np.float32)
    return seq, float(is_deteriorating)


def _synthetic_vitals_dataset(n_samples=800, seq_len=SEQ_LEN, horizon=FORECAST_HORIZON, seed=42):
    rng = np.random.default_rng(seed)
    sequences, labels = [], []
    for _ in range(n_samples):
        seq, is_abnormal = _synthetic_vitals_sequence(rng, seq_len, horizon)
        sequences.append(seq)
        labels.append(is_abnormal)
    sequences = np.stack(sequences)
    labels = np.array(labels, dtype=np.float32)

    X = sequences[:, :seq_len, :]
    y = sequences[:, seq_len:, :]  # forecast target: the horizon steps
    return X, y, labels  # labels: 1.0 = sequence contains a deterioration event


def load_vitals_sequences(mimic_dir=None, seq_len=SEQ_LEN, horizon=FORECAST_HORIZON,
                           n_samples_fallback=800):
    """
    Returns (X, y, labels):
      X      - (n, seq_len, 2) input window (HR, SpO2)
      y      - (n, horizon, 2) forecast target (Module 4 use)
      labels - (n,) 0=normal-looking sequence, 1=contains a deterioration
               event (Module 5 use: train the VAE on labels==0 only)

    Searches for real MIMIC directories (MIMIC-IV / MIMIC-III) first;
    on any failure falls back to a synthetic dataset.
    """
    candidate_dirs = [
        mimic_dir,
        "mimic-iv-clinical-database-demo-2.2",
        "../mimic-iv-clinical-database-demo-2.2",
        "../../mimic-iv-clinical-database-demo-2.2",
        "d:/NN PROJECT/mimic-iv-clinical-database-demo-2.2",
        "mimic-iii-clinical-database-demo-1.4",
        "../mimic-iii-clinical-database-demo-1.4",
        "../../mimic-iii-clinical-database-demo-1.4",
        "d:/NN PROJECT/mimic-iii-clinical-database-demo-1.4",
    ]
    resolved_dir = None
    for d in candidate_dirs:
        if d and os.path.isdir(d):
            resolved_dir = d
            break

    if resolved_dir is not None:
        try:
            X, y, labels = _try_load_mimic_vitals(resolved_dir, seq_len, horizon)
            print(f"Loaded {len(X)} real MIMIC vitals sequences from {resolved_dir} "
                  f"({int(labels.sum())} marked with cardiac/deterioration events).")
            return X, y, labels
        except Exception as e:
            print(f"Could not load real MIMIC vitals data from {resolved_dir} ({e}). "
                  f"Falling back to synthetic vitals sequences.")

    X, y, labels = _synthetic_vitals_dataset(n_samples=n_samples_fallback,
                                              seq_len=seq_len, horizon=horizon)
    print(f"Loaded {len(X)} synthetic vitals sequences "
          f"({int(labels.sum())} with a deterioration event).")
    return X, y, labels
