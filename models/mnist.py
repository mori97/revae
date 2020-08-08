import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .common import reparameterize


class _Encoder(nn.Module):

    def __init__(self, z_dim):
        super(_Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 2 * z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu, logvar = torch.chunk(self.fc3(h), 2, dim=1)
        return mu, logvar


class _Decoder(nn.Module):

    def __init__(self, z_dim):
        super(_Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 784)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = torch.sigmoid(self.fc3(h))
        return h


class _Classifier(nn.Module):

    def __init__(self, z_c_dim):
        super(_Classifier, self).__init__()

        self.fc1 = nn.Linear(z_c_dim, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, z_c):
        h = F.relu(self.fc1(z_c))
        h = self.fc2(h)
        return h


class _ConditionalPrior(nn.Module):

    def __init__(self, z_c_dim):
        super(_ConditionalPrior, self).__init__()

        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2 * z_c_dim)

    def forward(self, y):
        h = F.relu(self.fc1(y))
        mu, logvar = torch.chunk(self.fc2(h), 2, dim=1)
        return mu, logvar


class REVAEMNIST(nn.Module):
    """Reparameterized VAE for the MNIST dataset.

    Args:
        z_c_dim (int): The dimension of z_c.
        z_exc_dim (int): The dimension of z_\c.
    """
    def __init__(self, z_c_dim, z_exc_dim):
        super(REVAEMNIST, self).__init__()

        self._z_c_dim = z_c_dim
        self._z_exc_dim = z_exc_dim

        self.encoder = _Encoder(z_c_dim + z_exc_dim)
        self.decoder = _Decoder(z_c_dim + z_exc_dim)
        self.classifier = _Classifier(z_c_dim)
        self.cond_prior = _ConditionalPrior(z_c_dim)

    @property
    def z_c_dim(self):
        return self._z_c_dim

    @property
    def z_exc_dim(self):
        return self._z_exc_dim

    def forward(self, x, y):
        """Output the upper bound (the negative lower bound) of each sample for
        optimization.

        Args:
            x (torch.Tensor): Samples of shape (N, 784).
                where N is the batch size.
            y (torch.Tensor): The labels of shape (N,).

        Returns:
            torch.Tensor : The negative lower bound of each sample.
                The shape of returned value is (N,).
        """
        eye = torch.eye(10, device=y.device)
        uns_mask = (y == -1)  # Unsupervised mask
        batch_size = x.size(0)

        z_mu, z_logvar = self.encoder(x)
        z = reparameterize(z_mu, z_logvar)
        z_c, z_exc = z[:, :self.z_c_dim], z[:, self.z_c_dim:]
        recon = self.decoder(z)

        # Convert y to one-hot vector and Sample y for those without labels
        y[uns_mask] = 0
        y = eye[y]
        h = F.log_softmax(self.classifier(z_c), dim=1)
        y[uns_mask] = F.gumbel_softmax(h[uns_mask], tau=0.5)
        # log q(y|z_c)
        log_q_y_zc = torch.sum(h * y, dim=1)
        # log p(x|z)
        log_p_x_z = -F.binary_cross_entropy(recon, x, reduction='none').sum(1)
        # log p(z_c|y)
        z_c_mu, z_c_logvar = self.cond_prior(y)
        z_c_std = torch.exp(0.5 * z_c_logvar)
        log_p_zc_y = Normal(z_c_mu, z_c_std).log_prob(z_c).sum(1)
        # log p(z_\c)
        dist = Normal(torch.zeros_like(z_exc), torch.ones_like(z_exc))
        log_p_zexc = dist.log_prob(z_exc).sum(1)
        # log p(z|y)
        log_p_z_y = log_p_zc_y + log_p_zexc
        # log q(y|x)  ( Draw 128 points from q(z_c|x) )
        h = reparameterize(z_mu[:, :self.z_c_dim], z_logvar[:, :self.z_c_dim],
                           n_samples=128)
        h = self.classifier(h.reshape(128 * batch_size, h.size(2)))
        h = F.log_softmax(h, dim=1).reshape(128, batch_size, h.size(1))
        h = torch.logsumexp(h, dim=0) - math.log(128)
        log_q_y_x = torch.sum(h * y, dim=1)
        # log q(z|x)
        z_std = torch.exp(0.5 * z_logvar)
        log_q_z_x = Normal(z_mu, z_std).log_prob(z).sum(1)

        # Calculate the lower bound
        h = log_p_x_z + log_p_z_y - log_q_y_zc - log_q_z_x
        zeros = torch.zeros_like(h)
        ones = torch.ones_like(h)
        coef = torch.where(uns_mask, ones, torch.exp(log_q_y_zc - log_q_y_x))
        lb = coef * h + torch.where(uns_mask, zeros, log_q_y_x)

        return -lb
