"""
fusion_models.py
Module 6 - Multi-Modal Late Fusion & Cross-Modal Attention.

Combines the extracted feature representations from all four project branches:
  1. Clinical Metadata (ClinicalMLP)           -> 32-dim feature vector
  2. ECG Waveform (ECG1DCNN + SelfAttention)   -> 64-dim feature vector
  3. ICU Vitals Trends (VitalsBiLSTM)          -> 64-dim feature vector
  4. Vitals Anomaly / Latent (VitalsBiLSTMVAE) -> 16-dim latent mu (+ anomaly score)

Total concatenated embedding: 176 dimensions.

Includes:
  - ModalityAttention: Learns dynamic importance weights across modalities for clinical explainability.
  - CardiacRiskFusionNet: End-to-end multi-branch fusion network with calibrated risk probability output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityGating(nn.Module):
    """
    Learns dynamic soft-attention weights across the 4 modalities.
    Provides clinical explainability: which stream (clinical history,
    ECG morphology, or vitals trajectory) drove the alert?
    """

    def __init__(self, clinical_dim=32, ecg_dim=64, vitals_dim=64, vae_dim=16):
        super(ModalityGating, self).__init__()
        # Project each modality to a common 32-dim semantic space
        self.proj_clinical = nn.Linear(clinical_dim, 32)
        self.proj_ecg = nn.Linear(ecg_dim, 32)
        self.proj_vitals = nn.Linear(vitals_dim, 32)
        self.proj_vae = nn.Linear(vae_dim, 32)

        # Attention scoring vector
        self.attn_net = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, f_clin, f_ecg, f_vit, f_vae):
        # f_*: (batch, dim)
        p_clin = torch.tanh(self.proj_clinical(f_clin))  # (B, 32)
        p_ecg = torch.tanh(self.proj_ecg(f_ecg))        # (B, 32)
        p_vit = torch.tanh(self.proj_vitals(f_vit))      # (B, 32)
        p_vae = torch.tanh(self.proj_vae(f_vae))        # (B, 32)

        stacked = torch.stack([p_clin, p_ecg, p_vit, p_vae], dim=1)  # (B, 4, 32)
        scores = self.attn_net(stacked).squeeze(-1)                   # (B, 4)
        weights = F.softmax(scores, dim=-1)                          # (B, 4)

        return weights


class CardiacRiskFusionNet(nn.Module):
    """
    Multi-Modal Late Fusion Neural Network.
    Fuses extracted representations from:
      - ClinicalMLP (32)
      - ECG1DCNN (64)
      - VitalsBiLSTM (64)
      - VitalsBiLSTMVAE (16)
    Optionally incorporates the VAE scalar reconstruction error.
    """

    def __init__(self, clinical_dim=32, ecg_dim=64, vitals_dim=64, vae_dim=16,
                 include_recon_error=True, dropout_rate=0.3):
        super(CardiacRiskFusionNet, self).__init__()
        self.include_recon_error = include_recon_error
        self.gating = ModalityGating(clinical_dim, ecg_dim, vitals_dim, vae_dim)

        in_dim = clinical_dim + ecg_dim + vitals_dim + vae_dim
        if include_recon_error:
            in_dim += 1

        self.fusion_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate / 2.0),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, f_clinical, f_ecg, f_vitals, f_vae, recon_error=None, return_attention=False):
        """
        Args:
          f_clinical: (batch, 32)
          f_ecg:      (batch, 64)
          f_vitals:   (batch, 64)
          f_vae:      (batch, 16)
          recon_error: (batch, 1) or None
          return_attention: bool
        """
        # Compute explainability weights
        attn_weights = self.gating(f_clinical, f_ecg, f_vitals, f_vae)  # (batch, 4)

        tensors = [f_clinical, f_ecg, f_vitals, f_vae]
        if self.include_recon_error and recon_error is not None:
            if recon_error.dim() != 2:
                recon_error = recon_error.view(f_clinical.size(0), -1)
            tensors.append(recon_error)

        fused_embedding = torch.cat(tensors, dim=1)
        risk_probability = self.fusion_head(fused_embedding)

        if return_attention:
            return risk_probability, attn_weights
        return risk_probability


class MultiModalPatientPipeline(nn.Module):
    """
    End-to-End Orchestrator:
    Encapsulates all 4 frozen pre-trained branches + the trainable fusion layer.
    Takes raw multimodal inputs:
      - x_clinical: (B, num_clinical_features)
      - x_ecg:      (B, 216)
      - x_vitals:   (B, 24, 2)
    Extracts features through frozen branches and outputs final combined risk.
    """

    def __init__(self, clinical_model, ecg_model, vitals_model, vae_model, fusion_model):
        super(MultiModalPatientPipeline, self).__init__()
        self.clinical_branch = clinical_model
        self.ecg_branch = ecg_model
        self.vitals_branch = vitals_model
        self.vae_branch = vae_model
        self.fusion_net = fusion_model

    def forward(self, x_clinical, x_ecg, x_vitals, return_attention=True):
        # Extract features from each modality branch
        _, f_clinical = self.clinical_branch(x_clinical, return_features=True)
        _, f_ecg = self.ecg_branch(x_ecg, return_features=True)
        _, f_vitals = self.vitals_branch(x_vitals, return_features=True)
        
        # VAE branch: feature is mu, plus reconstruction error
        recon, mu, log_var = self.vae_branch(x_vitals)
        recon_err = torch.mean((recon[:, :x_vitals.shape[1], :] - x_vitals) ** 2, dim=(1, 2), keepdim=True)

        return self.fusion_net(
            f_clinical, f_ecg, f_vitals, mu,
            recon_error=recon_err,
            return_attention=return_attention
        )

