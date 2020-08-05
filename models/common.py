import torch


def reparameterize(mu, logvar, n_samples=1):
    """Reparameterization trick.

    Args:
        mu (torch.Tensor): Mean.
        logvar (torch.Tensor): Logarithm of variation.
        n_samples (int): The number of samples.

    Returns:
        torch.Tensor: Samples drawn from the given Gaussian distribution.
            The shape is equal to mu if n_samples is 1,
            and (n_samples, *mu.shape) if n_samples is larger than 1.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(n_samples, *std.size(), device=std.device)
    z = mu + eps * std
    return z.squeeze(0)
