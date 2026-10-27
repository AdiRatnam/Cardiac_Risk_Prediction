"""
ecg_models.py
Module 3 - ECG Waveform Analysis: 1D-CNN + lightweight self-attention.

Theory topics covered: CNN, Kernels, Padding, Stride, 1D-CNN.
"Transfer learning" for this branch means training the 1D-CNN from
scratch on MIT-BIH beats and reusing it as a feature extractor for the
fusion layer in Week 6 - there is no pretrained 1D-ECG backbone to
transfer from, so this is the standard approach for ECG deep learning.
"""

import torch
import torch.nn as nn


class SelfAttention1D(nn.Module):
    """
    Lightweight single-head self-attention over the CNN's output time
    steps. Lets the model learn to weight the QRS-complex region (or
    wherever the abnormality is) more heavily than flat/noisy baseline
    segments, instead of pooling every time step equally.
    """

    def __init__(self, channels):
        super(SelfAttention1D, self).__init__()
        self.query = nn.Conv1d(channels, channels // 4, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // 4, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))  # learned residual weight,
        # starts at 0 so training begins as a plain CNN and gradually
        # learns how much attention to mix in (standard SAGAN-style trick)

    def forward(self, x):
        # x: (batch, channels, length)
        q = self.query(x)  # (B, C/4, L)
        k = self.key(x)    # (B, C/4, L)
        v = self.value(x)  # (B, C, L)

        attn_scores = torch.bmm(q.transpose(1, 2), k)  # (B, L, L)
        attn_weights = self.softmax(attn_scores)
        attended = torch.bmm(v, attn_weights.transpose(1, 2))  # (B, C, L)

        return x + self.gamma * attended  # residual connection


class ECG1DCNN(nn.Module):
    """
    1D-CNN over raw ECG beat windows, with He-init + BatchNorm + Dropout
    (reusing the Module 2 stability pattern) and a self-attention block
    after the conv stack, before the classification head.
    """

    def __init__(self, input_length=216, dropout_rate=0.3):
        super(ECG1DCNN, self).__init__()

        self.conv_block = nn.Sequential(
            # Conv 1: wide kernel to capture whole-beat morphology (P-QRS-T span)
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2),  # downsample: 216 -> 108

            # Conv 2: medium kernel for QRS-scale features
            nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 108 -> 54

            # Conv 3: narrow kernel, finer local detail, stride 2 downsamples
            # directly in the conv (demonstrates stride as an alternative
            # to pooling, per the Module 3 "Stride" theory topic)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # 54 -> 27
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )

        self.attention = SelfAttention1D(channels=64)

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, 64, 27) -> (B, 64, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_features=False):
        # x: (batch, input_length) -> add channel dim -> (batch, 1, input_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        features = self.conv_block(x)
        features = self.attention(features)
        pooled = self.global_pool(features).squeeze(-1)  # (B, 64) - reused
        # by the Week 6 fusion layer as this branch's feature vector

        out = self.classifier(pooled)
        if return_features:
            return out, pooled
        return out
