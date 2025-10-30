import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial

from models.s4.s4 import S4Block as S4

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class DeepARS4(nn.Module):

    def __init__(
        self,
        d_input: int,
        d_model: int = 256,
        n_layers: int = 4,
        dropout: float = 0.2,
        prenorm: bool = False,
    ):
        """
        Args:
            d_input: Input dimension of model. It should be set to 1+(#features), 1 for the last observed value + features for the current timestep.
            d_model: Model dimension, or number of independent convolution kernels created
            n_layers: How many S4/S4D layers to stack
            dropout: Dropout
            prenorm: True if normalization should happen before applying the S4 layer. This is what newer models use today. The original transformer used postnorm where the residual input also gets normalized.
        """
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (1 + #features)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(d_model=d_model, bidirectional=False, l_max=3000, final_act="glu", dropout=dropout, transposed=False)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder_mu = nn.Linear(d_model, 1)
        self.decoder_alpha = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, L, d_model) -> (B, L, d_model)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x)

        # Decode the outputs
        mu = F.softplus(self.decoder_mu(x))  # (B, L, d_model) -> (B, L, 1)
        alpha = F.softplus(self.decoder_alpha(x))  # (B, L, d_model) -> (B, L, 1)

        return mu, alpha

    def sample(self, x, n_samples, last_only=False):
        """
        Generate forecasts by sampling from the predictive distribution.

        Args:
            x: input tensor of shape (batch_size, seq_length, input_size)
            n_samples: number of forecast samples to draw

        Returns:
            samples: tensor of shape (n_samples, batch_size, seq_length)
        """
        self.eval()
        with torch.no_grad():
            mu, alpha = self.forward(x)
            if last_only:
                mu_last, alpha_last = mu[:,-1], alpha[:,-1]
                r = 1.0/alpha_last
                p = 1.0-1.0/(1.0+alpha_last*mu_last)
                dist = NegativeBinomial(total_count=r, probs=p)
                samples = dist.sample((n_samples,)) # (n_samples, batch, seq, 1)
                return samples
            else:
                r = 1.0/alpha
                p = 1.0-1.0/(1.0+alpha*mu)

                dist = NegativeBinomial(total_count=r, probs=p)
                samples = dist.sample((n_samples,)) # (n_samples, batch, seq, 1)
                return samples.squeeze(-1) # (n_samples, batch, seq)

class NegativeBinomialNLL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, alpha, targets):
        return (
            -NegativeBinomial(
                total_count=1.0 / alpha, 
                probs= 1.0 - 1.0 / (1.0 + alpha*mu)
            ).log_prob(targets).mean()
        )
