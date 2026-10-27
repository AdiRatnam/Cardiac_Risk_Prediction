"""
vae_models.py
Module 5 - Generative Anomaly Detection: BiLSTM-VAE.

Theory topic covered: VAE. Reuses vitals_data.py (same data as Module 4).
The BiLSTM itself serves as BOTH the encoder AND decoder of the VAE:

  - Encoder: BiLSTM over the input window -> mean-pool the FULL output
    sequence (every timestep, not just the final hidden state) -> a
    non-linear adapter bridge -> (mu, log_var) -> reparameterized z.
  - Decoder: a symmetric adapter bridge expands z back into the BiLSTM's
    working dimensionality -> BiLSTM reconstructs the input window (+
    forecasts `horizon` steps beyond it).

Design notes (v2):
  - Adapter layers (Linear -> LeakyReLU -> Dropout) sit on both sides of
    the latent bottleneck, rather than a single bare Linear projecting
    straight from the BiLSTM's pooled output to mu/log_var (encoder side)
    or straight from z into the decoder BiLSTM's input (decoder side).
    This gives the model actual non-linear capacity to reshape the
    representation at the point where it's least "sequence-shaped" -
    right where it crosses the latent bottleneck - instead of forcing a
    single linear map to do that job.
  - Encoder pools the FULL BiLSTM output sequence (mean over every
    timestep), not just the final hidden state - the same "sync via the
    output, across the entire timespan" pooling used in vitals_models.py,
    so the latent code actually reflects the whole window rather than
    just its boundary.
  - No BatchNorm anywhere in this file (there never was - this model
    normalizes via the adapter's LayerNorm instead, which has no
    batch-size dependency, unlike Modules 1-3's MLP/CNN branches).

Trained ONLY on normal vitals sequences (see vitals_data.py's `labels`
output - filter to labels==0 before training this model). At inference,
an abnormal sequence produces a reconstruction-error spike relative to
the normal-data error distribution - that spike IS the anomaly signal.
The decoder is also reused standalone (sample z ~ N(0, I) -> decode) to
generate synthetic rare-event-like sequences for Module 1/3 augmentation.
"""

import torch
import torch.nn as nn


class VitalsVAEEncoder(nn.Module):
    def __init__(self, n_features=2, hidden_size=64, latent_dim=16, dropout_rate=0.3):
        super(VitalsVAEEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                             batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Adapter bridge: gives the encoder non-linear capacity to reshape
        # the pooled sequence representation before it's split into
        # mu/log_var, rather than two bare Linear layers reading directly
        # off the BiLSTM output.
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
        )
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_log_var = nn.Linear(hidden_size, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in list(self.adapter.modules()) + [self.fc_mu, self.fc_log_var]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size*2) - every timestep
        pooled = out.mean(dim=1)  # mean-pool the FULL sequence, not just h_n
        pooled = self.layer_norm(pooled)
        bridged = self.adapter(pooled)
        mu = self.fc_mu(bridged)
        log_var = self.fc_log_var(bridged)
        return mu, log_var


class VitalsVAEDecoder(nn.Module):
    def __init__(self, n_features=2, hidden_size=64, latent_dim=16,
                 output_len=30, dropout_rate=0.3):
        super(VitalsVAEDecoder, self).__init__()
        self.output_len = output_len
        self.hidden_size = hidden_size

        # Adapter bridge: symmetric to the encoder's - expands the latent
        # code with a non-linear projection before it's repeated across
        # the output timesteps and fed to the decoder BiLSTM.
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
        )
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                             batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(hidden_size * 2, n_features)
        self._init_weights()

    def _init_weights(self):
        for m in list(self.adapter.modules()) + [self.fc_out]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, z):
        # z: (batch, latent_dim)
        bridged = self.adapter(z)                              # (batch, hidden_size)
        repeated = bridged.unsqueeze(1).repeat(1, self.output_len, 1)  # (batch, output_len, hidden_size)
        out, _ = self.lstm(repeated)
        reconstruction = self.fc_out(out)       # (batch, output_len, n_features)
        return reconstruction


class VitalsBiLSTMVAE(nn.Module):
    """
    Wraps encoder + decoder + reparameterization into one module so a
    single forward pass returns everything train_step needs, while
    encoder/decoder stay separately accessible for the Week 6 fusion
    (encoder mu = this branch's feature vector) and for standalone
    generation (decoder(z) with sampled z).
    """

    def __init__(self, n_features=2, hidden_size=64, latent_dim=16,
                 seq_len=24, horizon=6, dropout_rate=0.3):
        super(VitalsBiLSTMVAE, self).__init__()
        self.latent_dim = latent_dim
        self.output_len = seq_len + horizon  # reconstructs window + forecasts horizon
        self.encoder = VitalsVAEEncoder(n_features, hidden_size, latent_dim, dropout_rate)
        self.decoder = VitalsVAEDecoder(n_features, hidden_size, latent_dim,
                                         self.output_len, dropout_rate)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_features=False):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        if return_features:
            return reconstruction, mu, log_var, mu  # mu doubles as the feature vector
        return reconstruction, mu, log_var

    @torch.no_grad()
    def generate(self, n_samples=5, device="cpu"):
        """Sample z ~ N(0, I) and decode - synthetic rare-event sequences
        for Module 1/3 data augmentation."""
        self.eval()
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z)

    @torch.no_grad()
    def reconstruction_error(self, x, target):
        """Per-sample MSE reconstruction error - the anomaly score.
        target should be torch.cat([x, y_horizon], dim=1), matching
        this model's output_len."""
        self.eval()
        mu, log_var = self.encoder(x)
        recon = self.decoder(mu)  # deterministic (no sampling) at eval time
        return torch.mean((recon - target) ** 2, dim=(1, 2))


def vae_loss(reconstruction, target, mu, log_var, kl_weight=0.001):
    """Standard VAE objective: reconstruction MSE + beta-weighted KL
    divergence against a standard normal prior."""
    recon_loss = torch.mean(torch.sum((reconstruction - target) ** 2, dim=(1, 2)))
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
