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


class JointDWCluster(BaseSet):
    """
    Double well cluster with multiple atom types.
    The energy is the sum of DW interactions between all pairs of atoms,
    with different parameters for each pair.

    The number of particles is 4 and the number of atom types is 2.
    """

    sample_dir = "energy/dw_cluster2"
    n_particles = 4
    n_types = 2

    num_disc_tokens = 2
    n_disc_dim = 4
    n_conti_dim = 8
    spatial_dim = 2

    def __init__(self, device="cuda"):
        """
        a, b, c: Double-well potential parameters
        V(r) = a * (r^2 - b)^2 + c
        """
        super().__init__(dim=12)

        self.device = device

        self.dw_params = torch.tensor(
            [
                [[0.8, -3.0], [0.4, -2.5]],  # Type 0-0, Type 0-1
                [[0.4, -2.5], [0.6, -2.8]],  # Type 1-0, Type 1-1
            ],
            device=device,
        )

        self.sample_file_name = (
            f"dw_samples_particles{self.n_particles}_types{self.n_types}.pt"
        )
        self._load_sample()

    def split_into_types_and_coords(self, x):
        *batch_dims, _ = x.shape
        return (
            x[..., self.n_conti_dim :].to(torch.long),
            x[..., : self.n_conti_dim].reshape(*batch_dims, self.n_particles, 2),
        )

    def _compute_pairwise_distances(self, coords):
        """Compute pairwise distances between all particles.

        Args:
            coords: Tensor of shape [..., N, 2] containing particle coordinates

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

    def double_well_potential(self, r, a, b):
        """
        Double-well potential: V(r) = 0.5 * (a * (r - offset) ** 4 + b * (r - offset) ** 2)
        """
        offset = 2.0
        return 0.5 * (a * (r - offset) ** 4 + b * (r - offset) ** 2)

    def energy(self, x):
        """
        x: (B, *, D)
        returns U: (B, *)
        """
        types, coords = self.split_into_types_and_coords(x)

        r = self._compute_pairwise_distances(coords)

        a_ij = self.dw_params[types[..., None], types[..., None, :], 0]
        b_ij = self.dw_params[types[..., None], types[..., None, :], 1]

        dw_potential = self.double_well_potential(r, a_ij, b_ij)
        U = dw_potential.triu(1).sum(dim=(-2, -1))

        return U

    def _load_sample(self):
        sample_file = os.path.join(self.sample_dir, self.sample_file_name)
        try:
            self.samples = torch.load(sample_file)
        except (FileNotFoundError, EOFError):
            print(f"Generating samples for {self.sample_file_name}")
            os.makedirs(self.sample_dir, exist_ok=True)
            self.samples = self._generate_samples_gibbs(100000)
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
                type_samples = self.dw_mala(
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

    def _generate_samples_gibbs(self, batch_size):
        """Generate samples using Gibbs sampling by alternating between
        p(continuous|discrete) and p(discrete|continuous).

        Args:
            batch_size: Total number of samples to generate

        Returns:
            samples: Tensor of shape [batch_size, D] containing generated samples
        """
        # Gibbs sampling hyperparameters
        burn_in = 10000
        n_chains = 100  # Number of parallel chains
        thinning = 50  # Collect every 50th sample

        # Initialize chains with random coordinates and types
        coords = torch.randn(n_chains, self.n_particles, 2, device=self.device)
        types = torch.randint(
            0, self.n_types, (n_chains, self.n_particles), device=self.device
        )

        # Burn-in
        for _ in tqdm.tqdm(range(burn_in), desc="Burn-in"):
            coords, types = self._gibbs_step(coords, types)

        # Collect samples with thinning
        chain_samples = []
        steps_taken = 0
        samples_per_chain = (batch_size + n_chains - 1) // n_chains  # Ceiling division

        for _ in tqdm.tqdm(range(samples_per_chain * thinning), desc="Sampling"):
            coords, types = self._gibbs_step(coords, types)
            steps_taken += 1

            # Collect samples with thinning
            if steps_taken % thinning == 0:
                coords_flat = coords.view(n_chains, -1)
                sample = torch.cat([coords_flat, types.to(torch.float32)], dim=1)
                chain_samples.append(sample)

        # Combine all samples from all chains
        samples = torch.cat(chain_samples, dim=0)

        # If we collected more samples than requested, trim to the right size
        if samples.size(0) > batch_size:
            samples = samples[:batch_size]

        # Shuffle the samples
        perm = torch.randperm(samples.size(0), device=self.device)
        return samples[perm]

    def _gibbs_step(self, coords, types):
        """Perform one Gibbs sampling step by alternating between
        sampling p(continuous|discrete) and p(discrete|continuous).

        Args:
            coords: Tensor of shape [n_chains, n_particles, 2] containing coordinates
            types: Tensor of shape [n_chains, n_particles] containing types

        Returns:
            coords_new: Updated coordinates
            types_new: Updated types
        """
        # 1. Sample p(continuous|discrete) using Metropolis step
        # Create input tensor for energy computation
        coords_grad = coords.clone().detach().requires_grad_(True)
        coords_flat = coords_grad.reshape(coords.size(0), -1)
        x_input = torch.cat([coords_flat, types.to(torch.float32)], dim=1)

        # Compute energy and gradients
        energy = self.energy(x_input)
        grads = torch.autograd.grad(energy.sum(), coords_grad)[0]

        # Propose new coordinates with gradient step plus noise
        step_size = 1e-3
        noise_scale = math.sqrt(2 * step_size)
        coords_proposed = (
            coords - step_size * grads + torch.randn_like(coords) * noise_scale
        )

        # Compute energy of proposed state
        coords_flat_proposed = coords_proposed.reshape(coords.size(0), -1)
        x_input_proposed = torch.cat(
            [coords_flat_proposed, types.to(torch.float32)], dim=1
        )
        energy_proposed = self.energy(x_input_proposed)

        # Metropolis acceptance
        log_alpha = -energy_proposed + energy
        accept = (torch.rand_like(log_alpha) < torch.exp(log_alpha)).view(-1, 1, 1)
        coords_new = torch.where(accept, coords_proposed, coords)

        # 2. Sample p(discrete|continuous) using Gibbs sampling
        types_new = types.clone()

        # For each particle, sample its type conditioned on all other particles
        for i in range(self.n_particles):
            # For each possible type value
            log_probs = torch.zeros((coords.size(0), self.n_types), device=self.device)

            for t in range(self.n_types):
                # Create a copy of types and set the current particle to type t
                types_temp = types_new.clone()
                types_temp[:, i] = t

                # Compute energy with this configuration
                coords_flat = coords_new.reshape(coords.size(0), -1)
                x_input = torch.cat([coords_flat, types_temp.to(torch.float32)], dim=1)
                energy_t = self.energy(x_input)

                # Store negative energy (proportional to log probability)
                log_probs[:, t] = -energy_t

            # Normalize to get probabilities
            probs = F.softmax(log_probs, dim=1)

            # Sample new type for this particle
            types_new[:, i] = torch.multinomial(probs, 1).squeeze(-1)

        return coords_new, types_new

    def dw_mala(
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
        coords = torch.randn(n_chains, self.n_particles, 2, device=self.device)

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

        # Compute energy at proposed state
        y_flat = y.view(y.size(0), -1)
        y_input = torch.cat([y_flat, types_batch.to(torch.float32)], dim=1)
        E_y = self.energy(y_input)

        # Compute log acceptance ratio for MALA (simplified)
        log_alpha = -E_y.detach() + E.detach()

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
            coords: Tensor of shape [..., N, 2] containing particle coordinates

        Returns:
            dists: Tensor of shape [..., N*(N-1)/2] containing unique interatomic distances
        """
        r = self._compute_pairwise_distances(coords)
        mask = torch.triu(
            torch.ones(self.n_particles, self.n_particles, device=self.device),
            diagonal=1,
        ).bool()
        return r[..., mask].view(-1)

    def compute_type_distances(self, coords, types):
        """Compute interatomic distances for each type combination.

        Args:
            coords: Tensor of shape [..., N, 2] containing particle coordinates
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
        energy_bins = np.linspace(-15, 0, 100)
        energy_hist = plot_histogram(
            energies.cpu().numpy(),
            gt_energies,
            title="Energy distribution",
            xlabel="Energy",
            ylabel="Density",
            bins=energy_bins,
        )

        # 2. Distance distribution
        dist_bins = np.linspace(0, 10, 100)
        dist_hist = plot_histogram(
            dists.cpu().numpy(),
            gt_dists,
            title="Distance distribution",
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
        ylabel="Atom position",
        title="Atom type distribution by position",
    )
    type_ax.set_yticks(y)
    type_ax.set_yticklabels([f"Atom {i+1}" for i in range(n_particles)])
    type_ax.set_xlim(0, 1)
    type_ax.legend()

    return type_fig
