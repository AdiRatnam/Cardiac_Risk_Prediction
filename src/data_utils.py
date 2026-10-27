"""
data_utils.py
Shared data loading utility - Module 1 clinical metadata.

Pulled directly out of Week1_Module1_MLP.ipynb so both notebooks (and
later weeks) import the identical generator instead of copy-pasting it,
which is what kept the Week 2 ablation an apples-to-apples comparison.
"""

import numpy as np
import pandas as pd


def load_clinical_metadata(n_samples: int = 2000):
    """
    Synthetic clinically-plausible dataset: age, BMI, systolic BP,
    cholesterol, max heart rate -> binary risk label.
    Identical to the generator used in Week 1 (same seed=42), so any
    later comparison across weeks is on the same data.
    """
    np.random.seed(42)
    age = np.random.randint(25, 85, size=n_samples)
    bmi = np.random.normal(26.5, 4.5, size=n_samples).clip(15, 45)
    systolic_bp = np.random.normal(125, 18, size=n_samples).clip(80, 200)
    cholesterol = np.random.normal(210, 40, size=n_samples).clip(120, 350)
    max_hr = (220 - age) - np.random.normal(10, 8, size=n_samples)

    risk_score = (
        0.03 * (age - 50)
        + 0.04 * (bmi - 25)
        + 0.02 * (systolic_bp - 120)
        + 0.01 * (cholesterol - 200)
        - 0.02 * (max_hr - 140)
    )
    prob = 1 / (1 + np.exp(-risk_score))
    y = (prob > 0.5).astype(np.float32)

    df = pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "cholesterol": cholesterol,
        "max_hr": max_hr,
    })
    return df, y


def load_realistic_clinical_metadata(n_samples: int = 2000):
    """
    Clinically-plausible dataset with non-linear interaction terms
    and physiological Gaussian noise to provide a harder, realistic benchmark.
    """
    np.random.seed(42)
    age = np.random.randint(25, 85, size=n_samples)
    bmi = np.random.normal(26.5, 4.5, size=n_samples).clip(15, 45)
    systolic_bp = np.random.normal(125, 18, size=n_samples).clip(80, 200)
    cholesterol = np.random.normal(210, 40, size=n_samples).clip(120, 350)
    max_hr = (220 - age) - np.random.normal(10, 8, size=n_samples)

    # Non-linear clinical interaction + physiological Gaussian noise
    interaction = ((age - 50) / 10.0) * ((systolic_bp - 120) / 20.0)
    noise = np.random.normal(0, 0.45, size=n_samples)

    risk_score = (
        0.035 * (age - 50)
        + 0.030 * (bmi - 25)
        + 0.020 * (systolic_bp - 120)
        + 0.012 * (cholesterol - 200)
        - 0.018 * (max_hr - 140)
        + 0.15 * interaction
        + noise
    )
    prob = 1 / (1 + np.exp(-risk_score))
    y = (prob > 0.5).astype(np.float32)

    df = pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "cholesterol": cholesterol,
        "max_hr": max_hr,
    })
    return df, y


def load_cleveland_heart_disease(csv_path: str = None, core_5_features: bool = False):
    """
    Loads real UCI Cleveland Heart Disease clinical records from Heart_disease_cleveland_new.csv.
    Searches common relative and absolute paths if csv_path is None.
    If core_5_features is True, extracts the 5 primary clinical features:
      ['age', 'sex', 'trestbps', 'chol', 'thalach'].
    Returns (X_df, y) where y is binary risk target (0=low risk, 1=high risk).
    """
    import os
    candidate_paths = [
        csv_path,
        "Heart_disease_cleveland_new.csv",
        "../Heart_disease_cleveland_new.csv",
        "../../Heart_disease_cleveland_new.csv",
        "d:/NN PROJECT/Heart_disease_cleveland_new.csv",
        "./data/Heart_disease_cleveland_new.csv"
    ]
    resolved_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            resolved_path = p
            break

    if resolved_path is None:
        raise FileNotFoundError("Heart_disease_cleveland_new.csv could not be found.")

    df = pd.read_csv(resolved_path)
    if "target" not in df.columns:
        raise ValueError(f"Target column 'target' missing from {resolved_path}")

    y = df["target"].to_numpy(dtype=np.float32)
    X_df = df.drop(columns=["target"])
    if core_5_features:
        X_df = X_df[["age", "sex", "trestbps", "chol", "thalach"]]
    return X_df, y
