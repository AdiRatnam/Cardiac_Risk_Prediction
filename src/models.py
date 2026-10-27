"""
models.py
Module 1 (BaselineMLP) and Module 2 (ClinicalMLP) model definitions.

BaselineMLP - Module 1 in isolation: Linear -> LeakyReLU stack, PyTorch's
default weight init, no BatchNorm, no Dropout.

ClinicalMLP - Module 1 + Module 2: same depth/width, but with
He Initialization (kaiming_normal_, matched to LeakyReLU), BatchNorm1d
after every Linear, and Dropout(0.3) for regularization. This is the
architecture used in the Week 1 notebook; kept here as an importable
module so Week 2's ablation notebook (and later weeks) don't redefine it.
"""

import torch.nn as nn


class BaselineMLP(nn.Module):
    """Plain MLP: no He-init, no BatchNorm, no Dropout (Module 1 alone)."""

    def __init__(self, input_dim):
        super(BaselineMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, return_features=False):
        features = self.net[:4](x)  # 32-dim feature representation
        out = self.net[4:](features)
        if return_features:
            return out, features
        return out


class ClinicalMLP(nn.Module):
    """Module 2: He Initialization, Dropout (0.3), BatchNorm."""

    def __init__(self, input_dim):
        super(ClinicalMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")  # He Normal init
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_features=False):
        features = self.net[:8](x)  # 32-dim stabilized feature representation
        out = self.net[8:](features)
        if return_features:
            return out, features
        return out
