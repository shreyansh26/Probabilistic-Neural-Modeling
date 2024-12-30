import torch
import torch.nn as nn

class GaussianNet(nn.Module):
    def __init__(self):
        super(GaussianNet, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc_mean = nn.Linear(50, 1)
        self.fc_log_std = nn.Linear(50, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mean = self.fc_mean(h)
        log_std = self.fc_log_std(h)
        return mean, log_std

class LaplaceNet(nn.Module):
    def __init__(self):
        super(LaplaceNet, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc_loc = nn.Linear(50, 1)
        self.fc_log_scale = nn.Linear(50, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        loc = self.fc_loc(h)
        log_scale = self.fc_log_scale(h)
        return loc, log_scale

class GammaNet(nn.Module):
    def __init__(self):
        super(GammaNet, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc_concentration = nn.Linear(50, 1)
        self.fc_log_rate = nn.Linear(50, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        concentration = nn.functional.softplus(self.fc_concentration(h)) + 1e-6 # Ensure positive concentration
        log_rate = self.fc_log_rate(h)
        return concentration, torch.exp(log_rate) # Return rate (beta)

class VonMisesNet(nn.Module):
    def __init__(self):
        super(VonMisesNet, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc_mu = nn.Linear(50, 1)  # Mean location
        self.fc_log_kappa = nn.Linear(50, 1) # Log of concentration

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(h)) * torch.pi # Bound mu to [-pi, pi]
        log_kappa = self.fc_log_kappa(h)
        return mu, torch.exp(log_kappa) # Return mu and kappa

class BetaNet(nn.Module):
    def __init__(self):
        super(BetaNet, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc_log_alpha = nn.Linear(50, 1)
        self.fc_log_beta = nn.Linear(50, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        log_alpha = self.fc_log_alpha(h)
        log_beta = self.fc_log_beta(h)
        return torch.exp(log_alpha) + 1e-6, torch.exp(log_beta) + 1e-6 # Ensure alpha, beta > 0