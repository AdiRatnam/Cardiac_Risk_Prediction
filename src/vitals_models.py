"""
vitals_models.py
Module 4 - Temporal Vital Forecasting: BiLSTM.

Theory topic covered: BiLSTM. Processes a rolling window of ICU vitals
(heart rate, SpO2) forward AND backward, so the representation at each
timestep is informed by both what came before and what comes after it
within the window.

Design notes (v2):
  - Pools over the FULL output sequence (mean over all timesteps' forward+
    backward hidden states), not just the final hidden state h_n. h_n only
    reflects the boundary timestep; mean-pooling the full `return_sequences`
    output actually uses every timestep the BiLSTM produced, which is the
    point of a bidirectional read over the whole window.
  - Uses LayerNorm instead of BatchNorm1d. LayerNorm normalizes per-sample
    (across features), not per-batch, so it has no batch-size dependency
    and no risk of crashing on a batch of 1 - a more natural fit for
    sequence models than reusing the Module 2 BatchNorm pattern verbatim.
"""

import torch
import torch.nn as nn


class VitalsBiLSTM(nn.Module):
    """
    BiLSTM forecaster: takes a (batch, seq_len, n_features) window of
    vitals and predicts the next `horizon` timesteps.

    Stacked BiLSTM -> mean-pool over the FULL output sequence (every
    timestep, not just the last) -> LayerNorm -> small MLP head ->
    horizon*n_features forecast, reshaped to (batch, horizon, n_features).
    """

    def __init__(self, n_features=2, hidden_size=64, horizon=6, dropout_rate=0.3):
        super(VitalsBiLSTM, self).__init__()
        self.n_features = n_features
        self.horizon = horizon
        self.hidden_size = hidden_size

        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                              batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        # hidden_size//2 per direction * 2 directions = hidden_size total
        self.lstm2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size // 2,
                              batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, horizon * n_features),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_features=False):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)  # out: (batch, seq_len, hidden_size) - every timestep

        # Mean-pool over the FULL time axis (uses every timestep's forward+
        # backward representation), not just the final hidden state.
        pooled = out.mean(dim=1)  # (batch, hidden_size)
        pooled = self.layer_norm(pooled)  # feature vector reused by Week 6 fusion

        forecast = self.head(pooled)
        forecast = forecast.view(-1, self.horizon, self.n_features)

        if return_features:
            return forecast, pooled
        return forecast
