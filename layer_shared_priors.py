"""
This file contains a BNN module in which each layer shares learnable priors.
Requires: !pip install torchbnn
"""

import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class HierarchicalBayesLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(HierarchicalBayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Posterior parameters for weights: these are learned to fit the data.
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))

        self.prior_mu = Parameter(torch.zeros(1, 1))
        self.prior_log_sigma = Parameter(torch.full((1, 1), -3.0))

        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
            # Learnable prior for bias.
            self.prior_bias_mu = Parameter(torch.zeros(1))
            self.prior_bias_log_sigma = Parameter(torch.full((1,), -3.0))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_parameter('prior_bias_mu', None)
            self.register_parameter('prior_bias_log_sigma', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        # Initialize posterior parameters for weights.
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(-5.0)
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(-5.0)

    def forward(self, input):
        # Sample weights from the posterior distribution using the reparameterization trick.
        weight_eps = torch.randn_like(self.weight_log_sigma)
        weight = self.weight_mu + torch.exp(self.weight_log_sigma) * weight_eps

        if self.bias_mu is not None:
            bias_eps = torch.randn_like(self.bias_log_sigma)
            bias = self.bias_mu + torch.exp(self.bias_log_sigma) * bias_eps
        else:
            bias = None

        return F.linear(input, weight, bias)

    def kl_divergence(self):
        # Compute KL divergence for weights.
        kl_weight = self._kl_divergence(self.weight_mu, self.weight_log_sigma,
                                        self.prior_mu, self.prior_log_sigma)
        # And if present, for biases.
        if self.bias_mu is not None:
            kl_bias = self._kl_divergence(self.bias_mu, self.bias_log_sigma,
                                          self.prior_bias_mu, self.prior_bias_log_sigma)
            return kl_weight + kl_bias
        return kl_weight

    def _kl_divergence(self, mu_q, log_sigma_q, mu_p, log_sigma_p):
        """
        Computes the element-wise KL divergence between two Gaussians:
        q ~ N(mu_q, sigma_q) and p ~ N(mu_p, sigma_p)
        where sigma = exp(log_sigma).
        The formula is:
            KL(q||p) = log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2)/(2*sigma_p^2) - 0.5
        """
        sigma_q = torch.exp(log_sigma_q)
        sigma_p = torch.exp(log_sigma_p)
        kl = (log_sigma_p - log_sigma_q +
              (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5)
        return kl.sum()
