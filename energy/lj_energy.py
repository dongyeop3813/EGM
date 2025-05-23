import logging

import numpy as np
import torch

from .base import BaseSet
from .viz import *
from .metric import wasserstein


def lennard_jones_energy(r, eps=1.0, rm=1.0):
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


def interatomic_distance(
    x: torch.Tensor,
    n_particles: int,
    n_dimensions: int,
    remove_duplicates: bool = True,
):
    """
    Computes the distances between all the particle pairs.

    Parameters
    ----------
    x : torch.Tensor
        Positions of n_particles in n_dimensions.
        Tensor of shape `[n_batch, n_particles * n_dimensions]`.
    n_particles : int
        Number of particles.
    n_dimensions : int
        Number of dimensions.
    remove_duplicates : bool, optional
        Flag indicating whether to remove duplicate distances.
        If False, the all-distance matrix is returned instead.

    Returns
    -------
    distances : torch.Tensor
        All-distances between particles in a configuration.
        Tensor of shape `[n_batch, n_particles * (n_particles - 1) // 2]` if remove_duplicates.
        Otherwise `[n_batch, n_particles, n_particles]`.
    """

    batch_shape = x.shape[:-1]
    x = x.view(-1, n_particles, n_dimensions)

    distances = torch.cdist(x, x, p=2)

    if remove_duplicates:
        distances = distances[
            :, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1
        ]
        distances = distances.reshape(
            *batch_shape, n_particles * (n_particles - 1) // 2
        )
    else:
        distances = distances.reshape(*batch_shape, n_particles, n_particles)

    return distances


def remove_mean(x: torch.Tensor, n_particles: int, spatial_dim: int):
    """
    Removes the mean of the input tensor x.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape `[n_batch, n_particles * n_dimensions]`.
    n_particles : int
        Number of particles.
    spatial_dim : int
        Spatial dimension.

    Returns
    -------
    x : torch.Tensor
        Input tensor with mean removed.
    """

    batch_shape = x.shape[:-1]
    x = x.reshape(*batch_shape, n_particles, spatial_dim)
    x = x - x.mean(dim=-2, keepdim=True)
    return x.reshape(*batch_shape, n_particles * spatial_dim)


class LennardJonesEnergy(BaseSet):
    def __init__(
        self,
        spatial_dim: int,
        n_particles: int,
        epsilon: float = 1.0,
        min_radius: float = 1.0,
        oscillator: bool = True,
        oscillator_scale: float = 1.0,
        energy_factor: float = 1.0,
    ):
        super().__init__(dim=spatial_dim * n_particles)

        self.spatial_dim = spatial_dim
        self.n_particles = n_particles

        self.epsilon = epsilon
        self.min_radius = min_radius

        self.oscillator = oscillator
        self.oscillator_scale = oscillator_scale

        self.energy_factor = energy_factor

    def energy(self, x: torch.Tensor):
        assert x.shape[-1] == self.ndim

        # dists is a tensor of shape [..., n_particles * (n_particles - 1) // 2]
        dists = interatomic_distance(x, self.n_particles, self.spatial_dim)

        lj_energies = lennard_jones_energy(dists, self.epsilon, self.min_radius)

        # Each interaction is counted twice
        lj_energies = lj_energies.sum(dim=-1) * self.energy_factor * 2.0

        if self.oscillator:
            x = remove_mean(x, self.n_particles, self.spatial_dim)
            osc_energies = 0.5 * x.pow(2).sum(dim=-1)
            lj_energies = lj_energies + osc_energies * self.oscillator_scale

        return lj_energies

    def sample(self, batch_size: int):
        raise NotImplementedError

    def remove_mean(self, x: torch.Tensor):
        return remove_mean(x, self.n_particles, self.spatial_dim)

    def interatomic_distance(self, x: torch.Tensor):
        return interatomic_distance(
            x, self.n_particles, self.spatial_dim, remove_duplicates=True
        )

    def visualize(self, samples: torch.Tensor, show_ground_truth: bool = False):
        """
        Visualize LJ samples and their statistics

        Args:
            samples: Generated samples [batch_size, n_particles * 3]
            show_ground_truth: Whether to show ground truth samples for comparison

        Returns:
            dict: Dictionary containing matplotlib figures for each visualization
        """
        # Calculate interatomic distances
        dists = self.interatomic_distance(samples).view(-1).to("cpu").numpy()

        # Calculate energies
        energies = self.energy(samples).detach().cpu().numpy()

        # Get ground truth samples if needed
        if show_ground_truth:
            gt_samples = self.sample(len(samples))
            gt_dists = self.interatomic_distance(gt_samples).view(-1).to("cpu").numpy()
            gt_energies = self.energy(gt_samples).detach().cpu().numpy()
        else:
            gt_dists = None
            gt_energies = None

        # 1. Interatomic distance histogram
        dist_hist = plot_histogram(
            dists,
            gt_dists,
            title="Interatomic Distance Distribution",
            xlabel="Distance",
            ylabel="Density",
            color="skyblue",
            gt_color="lightgray",
            bins=100,
        )

        # 2. Energy histogram
        energy_hist = plot_histogram(
            energies,
            gt_energies,
            title="Energy Distribution",
            xlabel="Energy",
            ylabel="Density",
            color="lightgreen",
            gt_color="lightgray",
        )

        return {
            "dist_hist": dist_hist,
            "energy_hist": energy_hist,
        }

    def metric(self, x: torch.Tensor):
        """
        Energy W1
        """

        energies = self.energy(x).detach().cpu().view(-1, 1)

        gt_samples = self.sample(len(x))
        gt_energies = self.energy(gt_samples).detach().cpu().view(-1, 1)

        energy_w1 = wasserstein(energies, gt_energies, power=1)

        return {
            "energy_w1": energy_w1,
        }


class LJ13(LennardJonesEnergy):
    can_sample = True

    def __init__(self):
        super().__init__(
            spatial_dim=3,
            n_particles=13,
        )
        self.approx_sample = torch.tensor(
            np.load(f"energy/lj_samples/LJ13.npy"),
        )

    def sample(self, batch_size: int):
        indices = torch.randint(
            0,
            len(self.approx_sample),
            (batch_size,),
        )
        return self.approx_sample[indices]


class LJ55(LennardJonesEnergy):
    can_sample = True

    def __init__(self):
        super().__init__(
            spatial_dim=3,
            n_particles=55,
        )
        self.approx_sample = torch.tensor(
            np.load(f"energy/lj_samples/LJ55.npy"),
        )

    def sample(self, batch_size: int):
        indices = torch.randint(
            0,
            len(self.approx_sample),
            (batch_size,),
        )
        return self.approx_sample[indices]
