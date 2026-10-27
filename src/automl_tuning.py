"""
automl_tuning.py
Module 7 - Automated Machine Learning (AutoML) Hyperparameter & Architecture Search.

Uses Optuna (Bayesian Optimization via Tree-structured Parzen Estimator) to:
  1. Search optimal learning rates, dropout rates, layer widths, and weight decays.
  2. Empirically justify the architectural choices made in Modules 1-6.
  3. Generate hyperparameter convergence plots and parameter importance charts.
"""

import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

# Suppress verbose optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_clinical_automl_search(X, y, n_trials=20, timeout=180, device="cpu"):
    """
    Runs Bayesian hyperparameter search for Clinical MLP.
    Optimizes:
      - lr (1e-4 to 1e-2, log scale)
      - dropout_rate (0.1 to 0.5)
      - hidden_dim_1 (32, 64, 128)
      - hidden_dim_2 (16, 32, 64)
      - weight_decay (1e-5 to 1e-2, log scale)
    Returns:
      study: optuna.Study
      best_params: dict
    """
    X_tr_df, X_val_df, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_df).astype(np.float32)
    X_val = scaler.transform(X_val_df).astype(np.float32)

    ds_tr = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    ds_val = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

    loader_tr = DataLoader(ds_tr, batch_size=32, shuffle=True, drop_last=True)
    loader_val = DataLoader(ds_val, batch_size=32, shuffle=False)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)
        hidden_1 = trial.suggest_categorical("hidden_dim_1", [32, 64, 128])
        hidden_2 = trial.suggest_categorical("hidden_dim_2", [16, 32, 64])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        model = nn.Sequential(
            nn.Linear(X_tr.shape[1], hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 1),
            nn.Sigmoid()
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_auc = 0.0
        for epoch in range(15):
            model.train()
            for bx, by in loader_tr:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for bx, by in loader_val:
                    bx, by = bx.to(device), by.to(device)
                    preds.extend(model(bx).cpu().numpy())
                    targets.extend(by.cpu().numpy())

            auc = roc_auc_score(targets, preds)
            if auc > best_auc:
                best_auc = auc

            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_auc

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return study, study.best_params


def plot_automl_results(study, output_dir="./outputs"):
    """
    Renders matplotlib optimization history and parameter importance plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    trials = [t for t in study.trials if t.value is not None]

    # 1. Optimization History
    fig, ax = plt.subplots(figsize=(7, 4))
    trial_numbers = [t.number for t in trials]
    trial_values = [t.value for t in trials]
    best_values = np.maximum.accumulate(trial_values)

    ax.scatter(trial_numbers, trial_values, color="#4285F4", alpha=0.7, label="Trial Validation AUC")
    ax.plot(trial_numbers, best_values, color="#EA4335", linewidth=2.2, label="Best Objective Value")
    ax.set_title("AutoML: Bayesian Optimization Convergence History", fontsize=11, fontweight="bold")
    ax.set_xlabel("Trial Number", fontsize=10)
    ax.set_ylabel("Validation AUC-ROC", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "stage7_automl_optimization_history.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    # 2. Parameter Importances
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        params = list(importances.keys())
        scores = list(importances.values())
        y_pos = np.arange(len(params))

        ax.barh(y_pos, scores, color="#34A853", align="center", height=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params, fontsize=10)
        ax.set_xlabel("Relative Importance Score", fontsize=10)
        ax.set_title("AutoML: Hyperparameter Sensitivity Analysis", fontsize=11, fontweight="bold")
        for i, v in enumerate(scores):
            ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontweight="bold")
        ax.set_xlim(0, max(scores) + 0.1)
        plt.tight_layout()
        imp_path = os.path.join(output_dir, "stage7_automl_param_importances.png")
        fig.savefig(imp_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Parameter importance plot note: {e}")
        imp_path = None

    return hist_path, imp_path

