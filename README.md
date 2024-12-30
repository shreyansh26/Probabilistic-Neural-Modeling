# Probabilistic Neural Networks for Different Posterior Distributions

This project implements neural networks that outputs parameters for different probability distributions, enabling probabilistic regression for various types of data. I read this great [blog](https://hr0nix.github.io/ml-notes/dangers-of-l2-loss.html) on how, for scalar outcomes, the model should be based on the posterior distribution. This is a quick proof-of-concept on my learnings from the article.

## Supported Distributions

- **Gaussian**: For standard regression with homoscedastic/heteroscedastic noise
- **Laplace**: For regression with heavy-tailed noise
- **Gamma**: For positive-valued targets with multiplicative noise
- **Von Mises**: For circular/angular data
- **Beta**: For bounded [0,1] targets

## Project Structure

- `generate_data.py`: Synthetic data generators for each distribution type
- `models.py`: Neural network architectures that output distribution parameters
- `losses.py`: Negative log-likelihood loss functions for each distribution
- `train.py`: Training loop and visualization code

## Usage

```python
python train.py
```

This will:
1. Train models for each distribution type
2. Generate plots in `figures/` showing the predicted distributions

## Requirements

- PyTorch
- NumPy
- Matplotlib
- SciPy

## Example Output

The models learn to predict:
- mean and standard deviation for Gaussian
- location and scale for Laplace
- concentration and rate for Gamma
- mean direction and concentration for Von Mises
- alpha and beta parameters for Beta

Each model outputs uncertainty estimates appropriate for its distribution type.
