import torch
import numpy as np
from .base import BaseSet
from .viz import *
from .metric import wasserstein


class GMM25(BaseSet):
    """
    2D Gaussian Mixture Model with modes on a 5x5 grid
    """

    def __init__(self, sigma=1.0, device="cuda"):
        super().__init__(dim=2)
        self.sigma = sigma
        self.device = device

        # Create grid of means
        grid = [-10, -5, 0, 5, 10]
        means = []
        for x in grid:
            for y in grid:
                means.append([x, y])
        self.means = torch.tensor(means, device=device)  # [25, 2]
        self.n_components = len(self.means)

        # Equal weights for all components
        self.weights = torch.ones(self.n_components, device=device) / self.n_components

    def energy(self, x):
        """
        Compute the energy of samples under the GMM

        Args:
            x: Samples [batch_dim..., 2]

        Returns:
            Energy [batch_dim...]
        """
        # Reshape for broadcasting
        # x: [batch_dim..., 2] -> [batch_dim..., 1, 2]
        x = x.unsqueeze(-2)
        # means: [n_components, 2] -> [1, n_components, 2]
        means = self.means.unsqueeze(0)

        # Compute squared distances
        # [batch_dim..., n_components]
        squared_dists = torch.sum((x - means) ** 2, dim=-1)

        # Compute log probabilities
        # [batch_dim..., n_components]
        log_probs = -0.5 * squared_dists / (self.sigma**2)
        log_probs = log_probs + torch.log(self.weights)

        # Log-sum-exp for mixture
        # [batch_dim..., 1]
        log_probs = torch.logsumexp(log_probs, dim=-1, keepdim=True)

        return -log_probs.squeeze(-1)

    def sample(self, batch_size):
        """
        Sample from the GMM

        Args:
            batch_size: Number of samples to generate

        Returns:
            Samples [batch_size, 2]
        """
        # Sample components
        component_indices = torch.multinomial(
            self.weights, batch_size, replacement=True
        )

        # Sample from selected components
        selected_means = self.means[component_indices]
        samples = selected_means + self.sigma * torch.randn(
            batch_size, 2, device=self.device
        )

        return samples

    def visualize(self, samples, show_ground_truth=False):
        """
        Visualize GMM samples and their statistics

        Args:
            samples: Generated samples [batch_size, 2]
            show_ground_truth: Whether to show ground truth samples for comparison

        Returns:
            dict: Dictionary containing matplotlib figures for each visualization
        """
        # Calculate energies
        energies = self.energy(samples).detach().cpu().numpy()

        # Get ground truth samples if needed
        if show_ground_truth:
            gt_samples = self.sample(len(samples))
            gt_energies = self.energy(gt_samples).detach().cpu().numpy()
        else:
            gt_samples = None
            gt_energies = None

        # 1. Energy histogram
        energy_hist = plot_histogram(
            energies,
            gt_energies,
            title="Energy Distribution",
            xlabel="Energy",
            ylabel="Density",
            color="skyblue",
            gt_color="lightgray",
        )

        # 2. 2D Projection plot
        projection_plot = plot_contour_and_sample(
            samples,
            self.energy,
            title="2D Projection of GMM Samples",
        )

        return {
            "energy_hist": energy_hist,
            "projection_plot": projection_plot,
        }

    def metric(self, samples):
        energies = self.energy(samples).detach().cpu().unsqueeze(-1)
        gt_samples = self.sample(len(samples))
        gt_energies = self.energy(gt_samples).detach().cpu().unsqueeze(-1)

        return {
            "energy_w1": wasserstein(energies, gt_energies, power=1),
            "sample_w2": wasserstein(
                samples.detach().cpu(), gt_samples.detach().cpu(), power=2
            ),
        }
