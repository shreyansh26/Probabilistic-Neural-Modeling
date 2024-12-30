import torch

# Generate Synthetic Data
def generate_gaussian_data(n_samples=1000):
    x = torch.randn(n_samples, 1) * 5
    y = 2 * x + torch.randn(n_samples, 1) * 2
    return x, y

def generate_skewed_data(n_samples=1000):
    x = torch.randn(n_samples, 1) * 5
    # Simulate a skewed distribution, e.g., by exponentiating
    base_y = 2 * x + torch.randn(n_samples, 1) * 1
    y = torch.exp(base_y / 5) # Waiting time example, cannot be negative
    return x, y

def generate_laplacian_data(n_samples=1000):
    x = torch.randn(n_samples, 1) * 5
    # Generate Laplace noise with location=0, scale=2
    noise = torch.distributions.Laplace(0, 2).sample((n_samples, 1))
    y = 2 * x + noise
    return x, y

def generate_gamma_data(n_samples=1000):
    x = torch.rand(n_samples, 1) * 2  # Shape: [n_samples, 1]
    
    # True function: parameters of Gamma distribution depend on x
    # For example, shape (alpha) increases with x, rate (beta) is constant
    true_alpha = 2 + x.squeeze() * 3  # Shape parameter α > 0
    true_beta = 1.0  # Rate parameter β > 0 (constant)
    
    # Sample y from Gamma distribution with parameters depending on x
    y = torch.distributions.Gamma(concentration=true_alpha, rate=true_beta).sample()
    
    # Reshape y to [n_samples, 1]
    y = y.unsqueeze(1)
    return x, y

def generate_circular_data():
    # Example: Generate data in the range [-pi, pi]
    x_data = torch.linspace(-torch.pi, torch.pi, 100).unsqueeze(1)
    y_data = torch.sin(x_data)
    return x_data, y_data

def generate_bounded_data(n_samples=100):
    x = torch.randn(n_samples, 1) * 5
    # Generate values between 0 and 1, with a dependency on x
    y_unscaled = torch.sigmoid(x) + torch.randn(n_samples, 1) * 0.1
    y = torch.clamp(y_unscaled, 0.01, 0.99) # Keep within (0, 1) for Beta
    return x, y

# def generate_gamma_samples(n_samples, alpha=5.0, beta=1.0):
#     """
#     Generates samples from a Gamma distribution.

#     Args:
#         num_samples (int): Number of samples to generate.
#         alpha (float, optional): Shape parameter. Defaults to 2.0.
#         beta (float, optional): Rate parameter. Defaults to 1.0.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: Tensors containing the samples (x) and their probabilities (y).
#     """
#     gamma_dist = torch.distributions.Gamma(alpha, beta)
#     x = gamma_dist.sample((n_samples,))
#     y = torch.exp(gamma_dist.log_prob(x))
#     return x, y