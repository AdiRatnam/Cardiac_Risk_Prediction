"""
ecg_data.py
Module 3 - ECG Waveform Analysis: data loading.

Real data source: MIT-BIH Arrhythmia Database (PhysioNet, record 'mitdb').
Unlike MIMIC-III/IV, MIT-BIH requires NO credentialing - it's open access,
so this is the one real database we can wire in immediately.

Strategy:
  1. Try to download a handful of MIT-BIH records via `wfdb.dl_database`.
  2. Use the built-in annotation file (.atr) to find each heartbeat's
     R-peak location and its label (Normal 'N' vs one of several
     abnormal/arrhythmia beat classes).
  3. Cut a fixed-length window (e.g. 216 samples, ~0.6s at 360Hz) centered
     on each R-peak -> one row per heartbeat, label = normal/abnormal.
  4. If the download fails (no network, PhysioNet unreachable, wfdb not
     installed), fall back to a synthetic-but-ECG-shaped signal generator
     so the rest of the pipeline (model, training loop, notebook) is never
     blocked by a network/environment issue.

This mirrors the same "real data with a safety-net fallback" pattern used
in src/data_utils.py for Module 1/2.
"""

import os
import numpy as np

WINDOW_SIZE = 216  # samples either side of center is WINDOW_SIZE//2
ABNORMAL_SYMBOLS = {"L", "R", "V", "A", "F", "J", "E", "a", "j", "S", "e"}
NORMAL_SYMBOLS = {"N", "."}


def _try_load_mitbih(data_dir, records, window_size=WINDOW_SIZE):
    """
    Attempts a real MIT-BIH load. Returns (X, y) as numpy arrays, or
    raises - caller is expected to catch and fall back to synthetic data.
    """
    import wfdb  # imported lazily so the whole module still imports
                 # even when wfdb isn't installed (e.g. this sandbox)

    os.makedirs(data_dir, exist_ok=True)
    X, y = [], []

    for rec_name in records:
        local_path = os.path.join(data_dir, rec_name)
        if not os.path.exists(local_path + ".dat"):
            wfdb.dl_database("mitdb", dl_dir=data_dir, records=[rec_name])

        record = wfdb.rdrecord(local_path)
        annotation = wfdb.rdann(local_path, "atr")
        signal = record.p_signal[:, 0]  # lead I (MLII in most MIT-BIH records)

        half = window_size // 2
        for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
            if symbol not in NORMAL_SYMBOLS and symbol not in ABNORMAL_SYMBOLS:
                continue  # skip non-beat annotations (rhythm/quality markers)
            start, end = sample_idx - half, sample_idx + half
            if start < 0 or end > len(signal):
                continue  # skip beats too close to the record's edge
            window = signal[start:end]
            label = 0 if symbol in NORMAL_SYMBOLS else 1
            X.append(window)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _synthetic_ecg_beat(is_abnormal, window_size=WINDOW_SIZE, rng=None):
    """
    Generates one ECG-shaped beat using a sum-of-Gaussians template for the
    P, QRS, and T waves, so the fallback signal actually looks like an ECG
    (not random noise) and the CNN has real morphology to learn from.
    """
    rng = rng or np.random
    t = np.linspace(-1, 1, window_size)

    def gaussian(center, width, amplitude):
        return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    # Normal beat template: P wave, QRS complex, T wave
    p_wave = gaussian(-0.35, 0.05, 0.15)
    q_wave = gaussian(-0.05, 0.01, -0.15)
    r_wave = gaussian(0.0, 0.015, 1.0)
    s_wave = gaussian(0.05, 0.01, -0.25)
    t_wave = gaussian(0.35, 0.08, 0.3)

    if is_abnormal:
        # Widen/deform the QRS and shift the T wave to mimic an ectopic/
        # arrhythmic beat morphology (wide QRS is the real clinical signature
        # of e.g. ventricular ectopic beats).
        r_wave = gaussian(0.0 + rng.uniform(-0.05, 0.05), 0.04, rng.uniform(0.6, 1.3))
        s_wave = gaussian(0.08, 0.03, -rng.uniform(0.2, 0.5))
        t_wave = gaussian(0.45, 0.1, rng.uniform(-0.3, 0.3))
        p_wave *= rng.uniform(0.0, 0.5)  # P wave often diminished/absent

    beat = p_wave + q_wave + r_wave + s_wave + t_wave
    beat += rng.normal(0, 0.02, size=window_size)  # baseline noise
    return beat.astype(np.float32)


def _synthetic_ecg_dataset(n_samples=2000, abnormal_ratio=0.3, seed=42,
                            window_size=WINDOW_SIZE):
    rng = np.random.RandomState(seed)
    n_abnormal = int(n_samples * abnormal_ratio)
    n_normal = n_samples - n_abnormal

    X = [_synthetic_ecg_beat(False, window_size, rng) for _ in range(n_normal)]
    X += [_synthetic_ecg_beat(True, window_size, rng) for _ in range(n_abnormal)]
    y = [0.0] * n_normal + [1.0] * n_abnormal

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    shuffle_idx = rng.permutation(len(X))
    return X[shuffle_idx], y[shuffle_idx]


def load_ecg_beats(data_dir="./data/mitbih", records=("100", "101", "103", "106"),
                    n_samples_fallback=2000):
    """
    Returns (X, y): X is (n_beats, WINDOW_SIZE) raw ECG windows, y is
    (n_beats,) with 0=normal beat, 1=abnormal/arrhythmic beat.

    Tries real MIT-BIH data first; on ANY failure (no network, wfdb not
    installed, PhysioNet unreachable), transparently falls back to a
    synthetic-but-ECG-shaped dataset so the notebook never breaks - it
    just prints which source it used.
    """
    try:
        X, y = _try_load_mitbih(data_dir, records)
        if len(X) == 0:
            raise RuntimeError("MIT-BIH load returned zero beats")
        print(f"Loaded {len(X)} real MIT-BIH beats from records {records}.")
        return X, y
    except Exception as e:
        print(f"Could not load real MIT-BIH data ({e}). "
              f"Falling back to synthetic ECG-shaped beats.")
        return _synthetic_ecg_dataset(n_samples=n_samples_fallback)
