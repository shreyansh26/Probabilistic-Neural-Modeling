import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import GaussianNet, LaplaceNet, GammaNet, VonMisesNet, BetaNet
from losses import gaussian_nll_loss, laplace_nll_loss, gamma_nll_loss, von_mises_nll_loss, beta_nll_loss
from generate_data import generate_gaussian_data, generate_skewed_data, generate_laplacian_data, generate_gamma_data, generate_circular_data, generate_bounded_data

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training script
def train_model(model, data_generator, loss_fn, epochs=1000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    X_data, Y_data = data_generator()
    X_data, Y_data = X_data.to(device), Y_data.to(device)
    dataset = torch.utils.data.TensorDataset(X_data, Y_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            if isinstance(model, GaussianNet):
                mean, log_std = model(batch_x)
                loss = loss_fn(mean, log_std, batch_y)
            elif isinstance(model, LaplaceNet):
                loc, log_scale = model(batch_x)
                loss = loss_fn(loc, log_scale, batch_y)
            elif isinstance(model, GammaNet):
                concentration, rate = model(batch_x)
                loss = loss_fn(concentration, rate, batch_y)
            elif isinstance(model, VonMisesNet):
                mu, kappa = model(batch_x)
                loss = loss_fn(mu, kappa, batch_y)
            elif isinstance(model, BetaNet):
                alpha, beta = model(batch_x)
                loss = loss_fn(alpha, beta, batch_y)
            else:
                raise ValueError("Unknown model type")

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {losses[-1]:.4f}')
    return model, losses, X_data, Y_data

# Plot results
def plot_results(model, x_data, y_data, distribution_type="Gaussian"):
    model.eval()
    with torch.no_grad():
        x_range = torch.linspace(x_data.min(), x_data.max(), 100).unsqueeze(1).to(x_data.device)
        if isinstance(model, GaussianNet):
            mean, log_std = model(x_range)
            std = torch.exp(log_std)
            upper_bound = mean + 2 * std
            lower_bound = mean - 2 * std
            predicted_mean = mean
        elif isinstance(model, LaplaceNet):
            loc, log_scale = model(x_range)
            scale = torch.exp(log_scale)
            # Approximate 95% CI for Laplace (not as straightforward as Gaussian)
            upper_bound = loc + 2 * scale
            lower_bound = loc - 2 * scale
            predicted_mean = loc
        elif isinstance(model, GammaNet):
            concentration, rate = model(x_range)
            # For Gamma, plotting mean and some quantiles might be more informative
            predicted_mean = concentration / rate
            from scipy.stats import gamma
            alpha = concentration.cpu().numpy().flatten()
            beta = rate.cpu().numpy().flatten()
            quantiles_025 = torch.tensor([gamma.ppf(0.025, a, scale=1/b) for a, b in zip(alpha, beta)]).unsqueeze(1)
            quantiles_975 = torch.tensor([gamma.ppf(0.975, a, scale=1/b) for a, b in zip(alpha, beta)]).unsqueeze(1)
            upper_bound = quantiles_975
            lower_bound = quantiles_025
        elif isinstance(model, VonMisesNet):
            mu, kappa = model(x_range)
            predicted_mean = mu # Mean location of the Von Mises distribution
        elif isinstance(model, BetaNet):
            alpha, beta = model(x_range)
            predicted_mean = alpha / (alpha + beta)
        else:
            raise ValueError("Unknown model type")

    plt.figure(figsize=(10, 6))
    if distribution_type != "VonMises":
        plt.scatter(x_data.cpu().numpy(), y_data.cpu().numpy(), label='Data Points')
        plt.plot(x_range.cpu().numpy(), predicted_mean.cpu().numpy(), color='red', label=f'Predicted Mean ({distribution_type})')
        if distribution_type not in ["Gamma", "VonMises", "Beta"]:
            plt.plot(x_range.cpu().numpy(), upper_bound.cpu().numpy(), color='red', linestyle='--', alpha=0.5, label='Approx. 95% Interval')
            plt.plot(x_range.cpu().numpy(), lower_bound.cpu().numpy(), color='red', linestyle='--', alpha=0.5)
        elif distribution_type == "Gamma":
            plt.plot(x_range.cpu().numpy(), upper_bound.cpu().numpy(), color='red', linestyle='--', alpha=0.5, label='97.5% Quantile')
            plt.plot(x_range.cpu().numpy(), lower_bound.cpu().numpy(), color='red', linestyle='--', alpha=0.5, label='2.5% Quantile')
    elif distribution_type == "VonMises":
        plt.scatter(x_data.cpu().numpy(), y_data.cpu().numpy(), label='Data Points')
        plt.plot(x_range.cpu().numpy(), predicted_mean.cpu().numpy(), color='red', label=f'Predicted Mean Location (Von Mises)')
    elif distribution_type == "Beta":
        plt.scatter(x_data.cpu().numpy(), y_data.cpu().numpy(), label='Data Points')
        plt.plot(x_range.cpu().numpy(), predicted_mean.cpu().numpy(), color='red', label=f'Predicted Mean (Beta)')

    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.title(f'Regression with {distribution_type} Distribution')
    plt.legend()
    # plt.show()
    plt.savefig(f'plots/{distribution_type}.png')

if __name__ == "__main__":
    # --- Gaussian ---
    gaussian_model = GaussianNet().to(device)
    gaussian_trained_model, gaussian_losses, x_gaussian, y_gaussian = train_model(
        gaussian_model, generate_gaussian_data, gaussian_nll_loss
    )
    print("Gaussian Model Training Complete.\n")
    plot_results(gaussian_trained_model, x_gaussian, y_gaussian, distribution_type="Gaussian")

    # --- Laplace ---
    laplace_model = LaplaceNet().to(device)
    laplace_trained_model, laplace_losses, x_skewed, y_skewed = train_model(
        laplace_model, generate_laplacian_data, laplace_nll_loss
    )
    print("Laplace Model Training Complete.\n")
    plot_results(laplace_trained_model, x_skewed, y_skewed, distribution_type="Laplace")

    # --- Gamma ---
    gamma_model = GammaNet().to(device)
    gamma_trained_model, gamma_losses, x_skewed_gamma, y_skewed_gamma = train_model(
        gamma_model, generate_gamma_data, gamma_nll_loss
    )
    print("Gamma Model Training Complete.\n")
    plot_results(gamma_trained_model, x_skewed_gamma[y_skewed_gamma > 0], y_skewed_gamma[y_skewed_gamma > 0], distribution_type="Gamma")

    # --- Von Mises ---
    von_mises_model = VonMisesNet().to(device)
    von_mises_trained_model, von_mises_losses, x_circular, y_circular = train_model(
        von_mises_model, generate_circular_data, von_mises_nll_loss
    )
    print("Von Mises Model Training Complete.\n")
    plot_results(von_mises_trained_model, x_circular, y_circular, distribution_type="VonMises")

    # --- Beta ---
    beta_model = BetaNet().to(device)
    beta_trained_model, beta_losses, x_bounded, y_bounded = train_model(
        beta_model, generate_bounded_data, beta_nll_loss
    )
    print("Beta Model Training Complete.\n")
    plot_results(beta_trained_model, x_bounded, y_bounded, distribution_type="Beta")