import torch
from torch.distributions import Normal, Laplace, Gamma, Beta
from scipy.special import i0

# Define Loss Functions (Negative Log-Likelihood)
def gaussian_nll_loss(mean, log_std, target):
    dist = Normal(mean, torch.exp(log_std))
    return -dist.log_prob(target).mean()

def laplace_nll_loss(loc, log_scale, target):
    dist = Laplace(loc, torch.exp(log_scale))
    return -dist.log_prob(target).mean()

def gamma_nll_loss(concentration, rate, target):
    dist = Gamma(concentration, rate)
    return -dist.log_prob(target).mean()

def von_mises_nll_loss(mu, kappa, target):
    """
    Negative log-likelihood for the Von Mises distribution.
    Implemented with numerical stability for large kappa.
    """
    cos_diff = torch.cos(target - mu)
    # For large kappa, log(I0(kappa)) is approximately kappa - 0.5 * log(2 * pi * kappa)
    log_bessel = torch.where(
        kappa < 50,  # Threshold can be adjusted
        torch.log(torch.tensor(i0(kappa.detach().cpu().numpy()))).to(kappa.device),
        kappa - 0.5 * torch.log(2 * torch.pi * kappa)
    )
    log_likelihood = kappa * cos_diff - log_bessel
    return -torch.mean(log_likelihood)

def beta_nll_loss(alpha, beta, target):
    dist = Beta(alpha, beta)
    return -dist.log_prob(target).mean()