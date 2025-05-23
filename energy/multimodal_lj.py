import os
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.nn.functional as F

from .base import BaseSet
from .viz import plot_histogram

_EPS = 1e-6


class JointLJCluster(BaseSet):
    """
    LJ cluster with multiple atom types.
    The energy is the sum of LJ interactions between all pairs of atoms,
    with different parameters for each pair.

    The number of particles is 4 and the number of atom types is 2.
    """

    sample_dir = "energy/lj_cluster"
    n_particles = 4
    n_types = 2

    num_disc_tokens = 2
    n_disc_dim = 4
    n_conti_dim = 12
    spatial_dim = 3

    def __init__(self, sigma=1.0, device="cuda"):
        """
        sigma: LJ length scale
        """
        super().__init__(dim=16)

        self.sigma = float(sigma)

        self.device = device
        # Make sure eps is symmetric
        self.eps = torch.tensor(
            [
                [1.0, 0.7],
                [0.7, 0.5],
            ],
            device=self.device,
        )

        self.sample_file_name = (
            f"lj_samples_particles{self.n_particles}_types{self.n_types}.pt"
        )
        self._load_sample()

    def split_into_types_and_coords(self, x):
        *batch_dims, _ = x.shape
        return (
            x[..., self.n_conti_dim :].to(torch.long),
            x[..., : self.n_conti_dim].reshape(*batch_dims, self.n_particles, 3),
        )

    def _compute_pairwise_distances(self, coords):
        """Compute pairwise distances between all particles.

        Args:
            coords: Tensor of shape [..., N, 3] containing particle coordinates

        Returns:
            distances: Tensor of shape [..., N, N] containing pairwise distances
        """
        diff = coords[..., :, None, :] - coords[..., None, :, :]
        r_squared = torch.sum(diff**2, dim=-1) + _EPS
        r = torch.sqrt(r_squared)

        # Create a diagonal mask instead of modifying r in-place
        diag_mask = torch.eye(self.n_particles, dtype=torch.bool, device=coords.device)

        # Instead of setting diagonal to inf, create a new tensor with diagonals masked
        r_with_inf_diag = r.clone()
        r_with_inf_diag[..., diag_mask] = float("inf")

        return r_with_inf_diag

    def _remove_mean(self, coords):
        return coords - coords.mean(dim=-2, keepdim=True)

    def energy(self, x):
        """
        x: (B, *, D)
        returns U: (B, *)
        """
        types, coords = self.split_into_types_and_coords(x)

        r = self._compute_pairwise_distances(coords)
        eps_ij = self.eps[types[..., None], types[..., None, :]]

        inv_r = self.sigma / (r + _EPS)
        inv_r6 = inv_r**6
        lj = 0.5 * eps_ij * (inv_r6**2 - inv_r6)
        U = lj.triu(1).sum(dim=(-2, -1))

        coords = self._remove_mean(coords)
        osc_energies = 0.5 * (coords**2).sum(dim=(-1, -2))

        return U + osc_energies

    def _load_sample(self):
        sample_file = os.path.join(self.sample_dir, self.sample_file_name)
        try:
            self.samples = torch.load(sample_file)
        except (FileNotFoundError, EOFError):
            print(f"Generating samples for {self.sample_file_name}")
            os.makedirs(self.sample_dir, exist_ok=True)
            self.samples = self._generate_samples(100000)
            torch.save(self.samples, sample_file)

    def _generate_samples(self, batch_size):
        """Generate samples using MALA with burn-in and thinning.

        Args:
            batch_size: Total number of samples to generate

        Returns:
            samples: Tensor of shape [batch_size, D] containing generated samples
        """
        # Enumerate all possible atom-type assignments
        type_list = list(
            itertools.product(range(self.n_types), repeat=self.n_particles)
        )
        types_all = torch.tensor(
            type_list, dtype=torch.long, device=self.device
        )  # [16,4]

        # MALA hyperparameters
        burn_in = 10000
        n_chains = 100  # Number of parallel chains
        thinning = 50  # Collect every 50th sample
        step_size = 1e-3
        noise_scale = math.sqrt(2 * step_size)

        samples_list = []
        for types in tqdm.tqdm(types_all):
            # Create sample file path for each atom type
            type_str = "_".join(map(str, types.tolist()))
            type_sample_file = os.path.join(
                self.sample_dir, f"type_{type_str}_{self.sample_file_name}"
            )

            # Check if file exists
            if os.path.exists(type_sample_file):
                # Load existing file
                type_samples = torch.load(type_sample_file)
                samples_list.append(type_samples)
                continue
            else:
                type_samples = self.lj_mala(
                    types,
                    batch_size,
                    n_chains,
                    burn_in,
                    thinning,
                    step_size,
                    noise_scale,
                )
                torch.save(type_samples, type_sample_file)
                samples_list.append(type_samples)

        # Combine and shuffle samples from all types
        samples = torch.cat(samples_list, dim=0)
        perm = torch.randperm(samples.size(0), device=self.device)
        return samples[perm]

    def lj_mala(
        self,
        types,
        num_samples,
        n_chains,
        burn_in,
        thinning,
        step_size,
        noise_scale,
    ):
        samples_per_chain = num_samples // n_chains  # Samples to collect per chain

        # Initialize chains
        types_batch = types.unsqueeze(0).expand(n_chains, -1)
        coords = (
            torch.randn(n_chains, self.n_particles, 3, device=self.device) * self.sigma
        )

        # Burn-in
        for _ in range(burn_in):
            coords = self._mala_step(coords, types_batch, step_size, noise_scale)

        # Collect samples with thinning
        chain_samples = []
        steps_taken = 0
        samples_collected = 0
        while samples_collected < samples_per_chain:
            coords = self._mala_step(coords, types_batch, step_size, noise_scale)
            steps_taken += 1

            # Collect samples with thinning
            if steps_taken % thinning == 0:
                coords_flat = coords.view(n_chains, -1)
                sample = torch.cat([coords_flat, types_batch.to(torch.float32)], dim=1)
                chain_samples.append(sample)
                samples_collected += 1

        # Combine all samples from all chains for the current type
        type_samples = torch.cat(chain_samples, dim=0)
        return type_samples

    def _mala_step(self, coords, types_batch, step_size, noise_scale):
        """Perform one MALA step."""
        # Compute energy and gradients at current state
        x = coords.clone().detach().requires_grad_(True)
        x_flat = x.view(x.size(0), -1)
        x_input = torch.cat([x_flat, types_batch.to(torch.float32)], dim=1)
        E = self.energy(x_input)
        grads = torch.autograd.grad(E.sum(), x)[0]

        # Propose new state (create as separate tensor to avoid in-place issues)
        noise = torch.randn_like(x) * noise_scale
        y = x.detach() - step_size * grads.detach() + noise

        # Compute energy and gradients at proposed state
        y_det = y.clone().detach().requires_grad_(True)
        y_flat = y_det.view(y_det.size(0), -1)
        y_input = torch.cat([y_flat, types_batch.to(torch.float32)], dim=1)
        E_y = self.energy(y_input)
        grads_y = torch.autograd.grad(E_y.sum(), y_det)[0]

        # Compute transition probabilities (ensure all tensors are detached)
        m_x = x.detach() - step_size * grads.detach()
        m_y = y_det.detach() - step_size * grads_y.detach()

        # Compute log acceptance ratio
        log_alpha = (
            -E_y.detach()
            + E.detach()
            - (
                (y.detach() - m_x).pow(2).sum(dim=(-1, -2))
                - (x.detach() - m_y).pow(2).sum(dim=(-1, -2))
            )
            / (4 * step_size)
        )

        # Accept or reject
        accept = (torch.rand_like(log_alpha) < torch.exp(log_alpha)).view(-1, 1, 1)
        return torch.where(accept, y.detach(), x.detach())

    def sample(self, batch_size):
        """Sample uniformly from types and initialize positions randomly."""
        indices = torch.randint(
            0, self.samples.shape[0], (batch_size,), device=self.device
        )
        return self.samples[indices]

    def compute_distances(self, coords):
        """Compute interatomic distances for given coordinates.

        Args:
            coords: Tensor of shape [..., N, 3] containing particle coordinates

        Returns:
            dists: Tensor of shape [..., N*(N-1)/2] containing unique interatomic distances
        """
        r = self._compute_pairwise_distances(coords)
        mask = torch.eye(self.n_particles, device=self.device).bool()
        mask = mask.unsqueeze(0).expand(r.shape[0], -1, -1)
        return r[~mask].view(-1)

    def compute_type_distances(self, coords, types):
        """Compute interatomic distances for each type combination.

        Args:
            coords: Tensor of shape [..., N, 3] containing particle coordinates
            types: Tensor of shape [..., N] containing particle types

        Returns:
            type_dists: Dictionary mapping type combinations to distance tensors
        """
        r = self._compute_pairwise_distances(coords)
        type_combinations = torch.tensor(
            list(itertools.product(range(self.n_types), repeat=2)), device=self.device
        )
        type_dists = {}

        for t1, t2 in type_combinations:
            type_mask = (types[..., :, None] == t1) & (types[..., None, :] == t2)
            masked_dists = r[
                type_mask & ~torch.eye(self.n_particles, device=self.device).bool()
            ]
            if len(masked_dists) > 0:
                type_dists[f"Type {t1}-{t2}"] = masked_dists

        return type_dists

    def compute_type_counts(self, types):
        """Compute the count of each atom type.

        Args:
            types: Tensor of shape [..., N] containing particle types

        Returns:
            type_counts: Tensor of shape [n_types] containing counts for each type
        """
        return torch.bincount(types.reshape(-1).to(torch.long), minlength=self.n_types)

    def visualize(self, samples, show_ground_truth=False):
        """Visualize the samples and their properties.

        Args:
            samples: Tensor of shape [B, D] containing samples
            show_ground_truth: Whether to show ground truth samples for comparison

        Returns:
            figures: Dictionary of matplotlib figures
        """
        # Process samples
        types, coords = self.split_into_types_and_coords(samples)
        dists = self.compute_distances(coords)
        energies = self.energy(samples)

        # Process ground truth if needed
        if show_ground_truth:
            gt_samples = self.sample(len(samples))
            gt_types, gt_coords = self.split_into_types_and_coords(gt_samples)
            gt_dists = self.compute_distances(gt_coords).cpu().numpy()
            gt_energies = self.energy(gt_samples).cpu().numpy()
        else:
            gt_samples = None
            gt_dists = None
            gt_energies = None

        # 1. Energy distribution
        energy_bins = np.linspace(-10, 15, 100)
        energy_hist = plot_histogram(
            energies.cpu().numpy(),
            gt_energies,
            title="Energy Distribution",
            xlabel="Energy",
            ylabel="Density",
            bins=energy_bins,
        )

        # 2. Distance distribution
        dist_bins = np.linspace(0, 3, 100)
        dist_hist = plot_histogram(
            dists.cpu().numpy(),
            gt_dists,
            title="Distance Distribution",
            xlabel="Distance",
            ylabel="Density",
            bins=dist_bins,
        )

        # 3. Type distribution
        type_fig = type_histogram(types, self.n_particles, self.n_types, self.device)

        return {
            "energy_hist": energy_hist,
            "distance_hist": dist_hist,
            "type_hist": type_fig,
        }

    def metric(self, samples):
        # Compute energies for generated samples
        energies = self.energy(samples)
        # Get ground truth samples
        gt_samples = self.sample(len(samples))
        gt_energies = self.energy(gt_samples)
        # Compute Wasserstein-1 distance between energy distributions
        from .metric import wasserstein

        energy_w1 = wasserstein(energies.view(-1, 1), gt_energies.view(-1, 1), power=1)
        return {"energy_w1": energy_w1}


def type_histogram(types, n_particles=4, n_types=2, device="cuda"):
    type_fig, type_ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(top=0.95)  # Reduce top margin

    # Prepare data for horizontal stacked bar plot
    y = np.arange(n_particles)
    height = 0.35

    # For generated samples
    type_counts = torch.zeros((n_particles, n_types), device=device)
    for i in range(n_particles):
        atom_types = types[:, i]
        counts = torch.bincount(atom_types.to(torch.long), minlength=n_types)
        type_counts[i] = counts

    # Convert to proportions
    type_props = type_counts / type_counts.sum(dim=1, keepdim=True)

    # Plot horizontal stacked bars for generated samples
    left = np.zeros(n_particles)
    for i, label in enumerate(["A", "B"]):
        type_ax.barh(
            y - height / 2,
            type_props[:, i].cpu().numpy(),
            height,
            left=left,
            label=f"Generated {label}",
            alpha=0.5,
            color="C0" if i == 0 else "C1",
        )
        left += type_props[:, i].cpu().numpy()

    type_ax.set(
        xlabel="Proportion",
        ylabel="Atom Position",
        title="Atom Type Distribution by Position",
    )
    type_ax.set_yticks(y)
    type_ax.set_yticklabels([f"Atom {i+1}" for i in range(n_particles)])
    type_ax.set_xlim(0, 1)
    type_ax.legend()

    return type_fig
