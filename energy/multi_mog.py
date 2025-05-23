import torch
import numpy as np
from .base import BaseSet
from .viz import *
from .metric import wasserstein


class MultiMoG(BaseSet):
    """
    Multi-modal GMM
    """

    sample_dir = "energy/multi_mog"

    num_disc_tokens = 2

    def __init__(
        self,
        num_projs=5,
        sigma=1.0,
        device="cuda",
        mode_distance=1.0,
    ):
        super().__init__(dim=2 * num_projs)
        self.num_projs = num_projs
        self.n_disc_dim = num_projs
        self.n_conti_dim = num_projs

        self.sigma = sigma
        self.mode_distance = mode_distance
        self.device = device

    def split(self, x):
        return (
            x[..., : self.n_conti_dim],
            x[..., self.n_conti_dim :].to(torch.long),
        )

    def energy(self, x):
        """
        Compute the energy of samples under the GMM

        Args:
            x: Samples [batch_dim..., 2 * num_projs]

        Returns:
            Energy [batch_dim...]
        """
        # x_disc: [batch_dim..., num_projs]
        x_conti, x_disc = self.split(x)

        mode_loc = torch.where(x_disc == 1, self.mode_distance, -self.mode_distance)

        squared_diff = (x_conti - mode_loc) ** 2
        energy_per_proj = 0.5 * squared_diff / (self.sigma**2)

        total_energy = torch.sum(energy_per_proj, dim=-1)

        return total_energy

    def sample(self, batch_size):
        """
        Generate samples from the MultiMoG distribution

        Args:
            batch_size: Number of samples to generate

        Returns:
            Samples [batch_size, 2 * num_projs]
        """
        x_conti = (
            torch.randn(batch_size, self.num_projs, device=self.device) * self.sigma
        )

        x_disc = torch.randint(
            0, self.num_disc_tokens, (batch_size, self.num_projs), device=self.device
        )

        mode_loc = torch.where(x_disc == 1, self.mode_distance, -self.mode_distance)
        x_conti = x_conti + mode_loc

        # Combine continuous and discrete variables
        samples = torch.cat([x_conti, x_disc.float()], dim=-1)

        return samples

    def visualize(self, samples, show_ground_truth=False):
        """
        Visualize MultiMoG samples and their statistics

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
        # Select the first two continuous dimensions and plot projected samples.
        x_conti, _ = self.split(samples)
        x_conti = x_conti[..., :2]

        projection_plot = plot_projection(
            x_conti,
            ground_truth=gt_samples[..., :2],
            title="2D Projection of MultiMoG Samples",
            lim=(-2.5, 2.5, -2.5, 2.5),
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
            "energy_w2": wasserstein(energies, gt_energies, power=2),
            "x_w2": wasserstein(
                samples[..., : self.n_conti_dim].detach().cpu(),
                gt_samples[..., : self.n_conti_dim].detach().cpu(),
                power=2,
            ),
        }
