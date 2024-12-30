import torch
from torch.distributions import Normal, Laplace, Gamma, Beta, VonMises
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
    dist = VonMises(mu, kappa)
    return -dist.log_prob(target).mean()

def beta_nll_loss(alpha, beta, target):
    dist = Beta(alpha, beta)
    return -dist.log_prob(target).mean()