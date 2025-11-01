import torch
import torch.nn.functional as F
from torch import nn, lgamma, log, log1p
from torch.distributions import NegativeBinomial

class DeepAR(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 40,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        likelihood: str = "negbin",
        init_lstm_forget_bias: float = 1.0,
        scaled: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.likelihood = likelihood
        self.dropout_rate = dropout_rate
        self.scaled = scaled

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True
        )

        self._init_lstm_forget_bias(init_lstm_forget_bias)

        if likelihood == "negbin":
            self.proj_mu = nn.Linear(hidden_size, 1)
            self.proj_alpha = nn.Linear(hidden_size, 1)
        else:
            raise ValueError("likelihood must be negbin")

    def _init_lstm_forget_bias(self, b):
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(b)

    def forward(self, x, targets=None):
        """
        Forward pass.
        Args:
           x: tensor of shape (batch_size, seq_length, input_size)
           targets: tensor of shape (batch_size, seq_length)
        Returns:
           dist_params: dictionary of distribution parameters
           loss: loss of the forward pass
        """

        #nu = 1.0 + x[..., 0].mean(dim=1, keepdim=True).unsqueeze(-1) # (batch, 1, 1)
        x_scaled = x.clone() if self.scaled else x
        if self.scaled:
            nu = 1.0 + x[..., 0].max(dim=1, keepdim=True).values.unsqueeze(-1) # (batch, 1, 1)
            x_scaled[..., 0] = x_scaled[..., 0] / nu.squeeze(-1)

        # lstm: (batch_size, seq_length, input_size) -> (batch_size, seq_length, hidden_size)
        out, (hidden, cell) = self.lstm(x_scaled)

        raw_mu = F.softplus(self.proj_mu(out)) # (batch_size, seq_length)
        raw_alpha = F.softplus(self.proj_alpha(out)) # (batch_size, seq_length)

        if self.scaled:
            mu = nu * raw_mu
            alpha = raw_alpha / torch.sqrt(nu)
        else:
            mu = raw_mu
            alpha = raw_alpha

        return mu, alpha

    def forward_with_activations(self, x, targets=None):
        out, (hidden, cell) = self.lstm(x)

        if self.likelihood == "negbin":
            raw_mu = F.softplus(self.proj_mu(out)) # (batch_size, seq_length)
            raw_alpha = F.softplus(self.proj_alpha(out)) # (batch_size, seq_length)

            #mu = nu * raw_mu
            #alpha = raw_alpha / torch.sqrt(nu)
            mu = raw_mu
            alpha = raw_alpha

            loss = None
            if targets is not None:
                loss = self._negbin_nll(targets, mu, alpha)

            return mu, alpha, loss, out
        else:
            raise ValueError("Unexpected likelihood in forward pass.")


    def _negbin_nll(self, z, mu, alpha):
        """
        Negative log likelihood of negative binomial.
        Args:
            z, mu, alpha: shape (batch_size, seq_length)
        """
        return (
            -NegativeBinomial(total_count=1.0/alpha, probs=1.0 - 1.0/(1.0+alpha*mu)).log_prob(z).mean()
        )
        #ll = lgamma(z + 1.0 / alpha) - lgamma(z+1) - lgamma(1.0 / alpha) - 1.0/alpha * log1p(alpha*mu + 1e-8) + z*(log(alpha*mu) - log1p(alpha*mu + 1e-8))
        #return -torch.mean(ll)

    def sample(self, x, n_samples=200, last_only=False):
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
            mu, alpha, _ = self.forward(x)
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

