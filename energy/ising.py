import matplotlib.pyplot as plt
import torch
import numpy as np

from .base import BaseSet
from .metric import wasserstein
from .viz import *


# 2D Ising model
class IsingModel(BaseSet):
    num_tokens = 2  # spin up and spin down

    def __init__(
        self,
        grid_size,
        inv_T,
        J,
        h=0.0,
        burn_in=10000,
        num_samples=300000,
        use_wolff=False,
        use_block_gibbs=False,
        block_size=2,
        num_chains=4,
        load_samples=True,
    ):
        super().__init__(dim=grid_size**2)
        self.grid_size = grid_size
        self.inv_T = inv_T
        self.J = J
        self.h = h  # magnetic field strength
        self.num_samples = num_samples
        self.use_wolff = use_wolff
        self.use_block_gibbs = use_block_gibbs
        self.block_size = block_size
        self.num_chains = num_chains

        # Try to load samples if requested
        if load_samples:
            self.samples = self._load_samples()
            if self.samples is None:
                # If loading fails, generate new samples
                self.samples = self.ising_simulation(inv_T, burn_in)
                self._save_samples()
        else:
            # Generate new samples
            self.samples = self.ising_simulation(inv_T, burn_in)
            self._save_samples()

    def _get_sample_filename(self):
        """Generate filename based on model parameters."""
        algorithm = (
            "wolff"
            if self.use_wolff
            else "block_gibbs" if self.use_block_gibbs else "metropolis"
        )
        filename = f"energy/ising_samples/ising_{self.grid_size}x{self.grid_size}_T{1/self.inv_T:.3f}_J{self.J:.3f}_h{self.h:.3f}_{algorithm}"
        if self.use_block_gibbs:
            filename += f"_block{self.block_size}"
        filename += f"_chains{self.num_chains}_samples{self.num_samples}.pt"
        return filename

    def _save_samples(self):
        """Save samples to file."""
        filename = self._get_sample_filename()
        torch.save(
            {
                "samples": self.samples,
                "grid_size": self.grid_size,
                "inv_T": self.inv_T,
                "J": self.J,
                "h": self.h,
                "use_wolff": self.use_wolff,
                "use_block_gibbs": self.use_block_gibbs,
                "block_size": self.block_size,
                "num_chains": self.num_chains,
                "num_samples": self.num_samples,
            },
            filename,
        )

    def _load_samples(self):
        """Load samples from file if they exist."""
        filename = self._get_sample_filename()
        try:
            checkpoint = torch.load(filename)
            # Verify parameters match
            if (
                checkpoint["grid_size"] == self.grid_size
                and checkpoint["inv_T"] == self.inv_T
                and checkpoint["J"] == self.J
                and checkpoint["h"] == self.h
                and checkpoint["use_wolff"] == self.use_wolff
                and checkpoint["use_block_gibbs"] == self.use_block_gibbs
                and checkpoint["block_size"] == self.block_size
                and checkpoint["num_chains"] == self.num_chains
                and checkpoint["num_samples"] == self.num_samples
            ):
                return checkpoint["samples"]
            else:
                print("Parameters don't match. Generating new samples.")
                return None
        except FileNotFoundError:
            print("Sample file not found. Generating new samples.")
            return None
        except Exception as e:
            print(f"Error loading samples: {e}. Generating new samples.")
            return None

    def _calculate_energy(self, x):
        up = torch.roll(x, shifts=1, dims=-2)
        down = torch.roll(x, shifts=-1, dims=-2)
        left = torch.roll(x, shifts=1, dims=-1)
        right = torch.roll(x, shifts=-1, dims=-1)

        # Interaction energy
        interaction_energy = -self.J * x * (up + down + left + right)
        interaction_energy = interaction_energy.sum(dim=(-1, -2)) / 2

        # Magnetic field energy
        magnetic_energy = -self.h * x.sum(dim=(-1, -2))

        # Apply inverse temperature to total energy
        return (interaction_energy + magnetic_energy) * self.inv_T

    def energy(self, x):
        """
        Compute the energy of the grid configuration with periodic boundary conditions.
        Returns -log unnormalized density which includes inverse temperature.
        """
        if (x == 2).any().item():
            raise ValueError("Invalid spin value")

        x = self.assure_grid(x)

        return self._calculate_energy(x)

    def local_energy(self, x):
        """
        Compute the energy of the grid configuration with periodic boundary conditions.
        Returns -log unnormalized density which includes inverse temperature.
        """
        x = self.assure_grid(x)

        # if x is neither 1 nor -1, then it's a mask. Set the mask to zero.
        mask = (x != 1) & (x != -1)
        x[mask] = 0

        return self._calculate_energy(x)

    def sample(self, batch_size, grid_sample=False) -> torch.Tensor:
        # Randomly select indices from pre-computed samples
        indices = torch.randint(0, len(self.samples), (batch_size,))

        grid = self.samples[indices]
        if grid_sample:
            return grid
        else:
            return self.flatten(grid)

    def ising_simulation(self, inv_T, burn_in, initial_grid=None):
        """
        Run parallel MCMC chains with optimized vectorized operations.
        """
        if initial_grid is None:
            grids = (
                torch.randint(0, 2, (self.num_chains, self.grid_size, self.grid_size))
                * 2
                - 1
            )
        else:
            grids = initial_grid.clone().repeat(self.num_chains, 1, 1)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        grids = grids.to(device)

        # Pre-allocate memory for samples
        samples_per_chain = self.num_samples // self.num_chains
        samples = torch.zeros(
            (self.num_samples, self.grid_size, self.grid_size),
            dtype=grids.dtype,
            device=device,
        )

        # Burn-in period with vectorized updates
        for _ in range(burn_in):
            if self.use_wolff:
                for i in range(self.num_chains):
                    grids[i] = wolff_step(grids[i], inv_T, J=self.J, h=self.h)
            elif self.use_block_gibbs:
                for i in range(self.num_chains):
                    grids[i] = block_gibbs_step(
                        grids[i], inv_T, J=self.J, h=self.h, block_size=self.block_size
                    )
            else:
                # Process all chains in parallel for Metropolis updates
                for i in range(self.num_chains):
                    grids[i] = metropolis_step(grids[i], inv_T, J=self.J, h=self.h)

        # Collect samples with optimized memory access
        for i in range(self.num_chains):
            chain_samples = torch.zeros(
                (samples_per_chain, self.grid_size, self.grid_size),
                dtype=grids.dtype,
                device=device,
            )

            if self.use_wolff:
                for j in range(samples_per_chain):
                    grids[i] = wolff_step(grids[i], inv_T, J=self.J, h=self.h)
                    chain_samples[j] = grids[i]
            elif self.use_block_gibbs:
                for j in range(samples_per_chain):
                    grids[i] = block_gibbs_step(
                        grids[i], inv_T, J=self.J, h=self.h, block_size=self.block_size
                    )
                    chain_samples[j] = grids[i]
            else:
                # Vectorized Metropolis updates for each chain
                for j in range(samples_per_chain):
                    grids[i] = metropolis_step(grids[i], inv_T, J=self.J, h=self.h)
                    chain_samples[j] = grids[i]

            samples[i * samples_per_chain : (i + 1) * samples_per_chain] = chain_samples

        return samples

    def flatten(self, x):
        batch_shape = x.shape[:-2]
        return (x.reshape(*batch_shape, self.grid_size * self.grid_size) + 1) // 2

    def to_grid(self, x):
        batch_shape = x.shape[:-1]
        return x.reshape(*batch_shape, self.grid_size, self.grid_size) * 2 - 1

    def is_flat(self, x):
        return x.shape[-1] == self.grid_size * self.grid_size

    def assure_grid(self, x):
        if self.is_flat(x):
            x = self.to_grid(x)
        return x

    def visualize(self, samples, show_ground_truth=False):
        samples = self.assure_grid(samples.cpu().detach())

        # Plot samples
        sample_fig = plot_ising_samples(samples)

        # Calculate energies and magnetization
        energies = self.energy(samples)
        magnetization = samples.float().mean(dim=(1, 2))

        # Get ground truth samples if needed
        if show_ground_truth:
            gt_samples = self.sample(len(samples), grid_sample=True)
            gt_energies = self.energy(gt_samples).detach().cpu().numpy()
            gt_magnetization = (
                gt_samples.float().mean(dim=(1, 2)).detach().cpu().numpy()
            )
        else:
            gt_samples = None
            gt_energies = None
            gt_magnetization = None

        # Plot energy histogram
        energy_hist = plot_histogram(
            energies.detach().cpu().numpy(),
            gt_energies,
            title="Energy histogram",
            xlabel="Energy",
            ylabel="Frequency",
            color="skyblue",
        )

        # Plot magnetization histogram
        magnetization_hist = plot_histogram(
            magnetization.detach().cpu().numpy(),
            gt_magnetization,
            title="Magnetization histogram",
            xlabel="Magnetization",
            ylabel="Frequency",
            color="lightgreen",
        )

        corr = two_point_correlation(samples)
        corr_fig, corr_ax = plt.subplots(figsize=(8, 6))
        corr_mean = corr.mean(dim=0).detach().cpu().numpy()
        distances = np.arange(len(corr_mean))
        corr_ax.plot(distances, corr_mean, "o-", color="purple", label="samples")
        corr_ax.set_xlabel("Distance (r)")
        corr_ax.set_ylabel("Correlation function C(r)")
        corr_ax.set_title("Two-point correlation function")
        corr_ax.grid(True)

        if show_ground_truth:
            gt_corr = two_point_correlation(gt_samples)
            gt_corr_mean = gt_corr.mean(dim=0).detach().cpu().numpy()
            corr_ax.plot(
                distances, gt_corr_mean, "o-", color="blue", label="ground truth"
            )
            corr_ax.legend()

        return {
            "sample_fig": sample_fig,
            "energy_hist": energy_hist,
            "magnetization_hist": magnetization_hist,
            "correlation_fig": corr_fig,
        }

    def metric(self, samples):
        samples = self.assure_grid(samples.cpu().detach())

        energies = self.energy(samples)
        magnetization = samples.float().mean(dim=(1, 2))

        gt_samples = self.sample(len(samples), grid_sample=True)
        gt_energies = self.energy(gt_samples)
        gt_magnetization = gt_samples.float().mean(dim=(1, 2))

        # Energy distribution W1 distance
        energy_w1 = wasserstein(
            energies.detach().cpu().view(-1, 1),
            gt_energies.detach().cpu().view(-1, 1),
            power=1,
        )

        energy_w2 = wasserstein(
            energies.detach().cpu().view(-1, 1),
            gt_energies.detach().cpu().view(-1, 1),
            power=2,
        )

        # Magnetization distribution W1 distance
        magnetization_w1 = wasserstein(
            magnetization.detach().cpu().view(-1, 1),
            gt_magnetization.detach().cpu().view(-1, 1),
            power=1,
        )

        magnetization_w2 = wasserstein(
            magnetization.detach().cpu().view(-1, 1),
            gt_magnetization.detach().cpu().view(-1, 1),
            power=2,
        )

        return {
            "energy_w1": energy_w1,
            "magnetization_w1": magnetization_w1,
            "energy_w2": energy_w2,
            "magnetization_w2": magnetization_w2,
        }


def metropolis_step(grid, inv_T, J=1, h=0):
    """
    Perform a single Metropolis update: randomly choose a spin, calculate the energy change,
    and flip the spin based on the Metropolis acceptance criterion.
    """
    N, M = grid.shape
    # Choose a random site.
    i = torch.randint(0, N, (1,)).item()
    j = torch.randint(0, M, (1,)).item()

    # Compute the sum of the nearest neighbors (using periodic boundaries).
    up = grid[(i - 1) % N, j]
    down = grid[(i + 1) % N, j]
    left = grid[i, (j - 1) % M]
    right = grid[i, (j + 1) % M]

    neighbors = up + down + left + right
    # Energy change if the spin is flipped (interaction + magnetic field)
    deltaE = 2 * J * grid[i, j] * neighbors + 2 * h * grid[i, j]

    # Metropolis acceptance criterion.
    if deltaE <= 0 or torch.rand(1).item() < torch.exp(-deltaE * inv_T).item():
        grid[i, j] *= -1  # Flip the spin.
    return grid


def wolff_step(grid, inv_T, J=1, h=0):
    """
    Perform a single Wolff cluster update step.
    """
    N, M = grid.shape
    # Choose a random site to start the cluster
    i = torch.randint(0, N, (1,)).item()
    j = torch.randint(0, M, (1,)).item()

    # Initialize the cluster
    cluster = torch.zeros_like(grid, dtype=torch.bool)
    cluster[i, j] = True
    stack = [(i, j)]

    # Probability to add a site to the cluster
    p = 1 - np.exp(-2 * J * inv_T)

    while stack:
        i, j = stack.pop()
        spin = grid[i, j]

        # Check neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni = (i + di) % N
            nj = (j + dj) % M

            # If neighbor has same spin and not in cluster
            if grid[ni, nj] == spin and not cluster[ni, nj]:
                if torch.rand(1).item() < p:
                    cluster[ni, nj] = True
                    stack.append((ni, nj))

    # Flip the entire cluster
    grid[cluster] *= -1
    return grid


def block_gibbs_step(grid, inv_T, J=1, h=0, block_size=2):
    """
    Perform a block Gibbs update step using vectorized operations.
    """
    N, M = grid.shape
    # Choose a random block
    i = torch.randint(0, N - block_size + 1, (1,)).item()
    j = torch.randint(0, M - block_size + 1, (1,)).item()

    # Get the block
    block = grid[i : i + block_size, j : j + block_size]

    # Create shifted versions of the block for internal interactions
    block_up = torch.roll(block, shifts=1, dims=0)
    block_left = torch.roll(block, shifts=1, dims=1)

    # Calculate internal interaction energy
    internal_energy = -J * block * (block_up + block_left)
    original_internal = internal_energy.sum()
    flipped_internal = -original_internal

    # Get neighbors outside the block
    neighbors_up = torch.roll(grid, shifts=1, dims=0)[
        i : i + block_size, j : j + block_size
    ]
    neighbors_left = torch.roll(grid, shifts=1, dims=1)[
        i : i + block_size, j : j + block_size
    ]

    # Calculate external interaction energy
    external_energy = -J * block * (neighbors_up + neighbors_left)
    original_external = external_energy.sum()
    flipped_external = -original_external

    # Calculate total energies
    original_energy = original_internal + original_external - h * block.sum()
    flipped_energy = flipped_internal + flipped_external - h * (-block.sum())

    # Calculate acceptance probability
    p = 1 / (1 + torch.exp((original_energy - flipped_energy) * inv_T))

    # Flip the block with probability p
    if torch.rand(1).item() < p:
        grid[i : i + block_size, j : j + block_size] *= -1

    return grid


def compare_sampling_algorithms(
    grid_size=32,
    J=1.0,
    h=0.0,
    inv_T_range=(0.1, 1.0, 0.1),
    num_samples=100000,
    num_chains=4,
    burn_in=10000,
):
    """
    Compare different sampling algorithms by plotting average absolute magnetization
    as a function of inverse temperature.
    """
    inv_T_values = torch.arange(*inv_T_range)
    algorithms = [
        {"name": "Metropolis", "use_wolff": False, "use_block_gibbs": False},
        {"name": "Wolff", "use_wolff": True, "use_block_gibbs": False},
        {
            "name": "Block Gibbs",
            "use_wolff": False,
            "use_block_gibbs": True,
            "block_size": 2,
        },
    ]

    # Create figure
    plt.figure(figsize=(10, 6))

    for algo in algorithms:
        magnetizations = []
        for inv_T in inv_T_values:
            # Try to load samples
            model = IsingModel(
                grid_size=grid_size,
                inv_T=inv_T.item(),
                J=J,
                h=h,
                num_samples=num_samples,
                num_chains=num_chains,
                burn_in=burn_in,
                **{k: v for k, v in algo.items() if k != "name"},
            )

            # Calculate average absolute magnetization
            samples = model.samples
            magnetization = samples.float().mean(dim=(1, 2)).abs().mean().item()
            magnetizations.append(magnetization)

            print(
                f"{algo['name']} at T={1/inv_T.item():.2f}: magnetization={magnetization:.4f}"
            )

        # Plot magnetization curve
        plt.plot(1 / inv_T_values, magnetizations, "o-", label=algo["name"])

    # Add theoretical critical temperature for 2D Ising model
    Tc = 2.269
    plt.axvline(x=Tc, color="gray", linestyle="--", label="Critical Temperature")

    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Absolute Magnetization")
    plt.title(f"Ising Model: {grid_size}x{grid_size}, J={J}, h={h}")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(
        f"energy/ising_samples/magnetization_comparison_{grid_size}x{grid_size}_J{J}_h{h}.png"
    )
    plt.close()

    return magnetizations


def two_point_correlation(spins: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2-point correlation function C(r) = <s(0)s(r)>
    averaged over all sites and directions, for each distance r=0..N/2.

    Args:
        spins (torch.Tensor): Tensor of shape (..., N, N), where ... is batch dims.

    Returns:
        torch.Tensor: Correlation function of shape (..., N//2 + 1)
    """
    assert spins.ndim >= 2, "Input must have at least 2 dimensions (N, N)"
    N = spins.shape[-1]
    batch_shape = spins.shape[:-2]

    # Normalize spins to -1 and +1 if not already
    assert torch.all((spins == -1) | (spins == 1)), "Spins must be -1 or +1"

    spins = spins.float()

    # Zero-mean for correlation if needed, but for Ising usually not centered
    mean_spin = spins.mean(dim=(-2, -1), keepdim=True)
    centered = spins - mean_spin

    # Compute correlations up to distance r = N//2 (to avoid wrap-around)
    max_r = N // 2
    corr = []
    for r in range(max_r + 1):
        shift_x = torch.roll(spins, shifts=r, dims=-2)
        shift_y = torch.roll(spins, shifts=r, dims=-1)

        # Compute correlation <s(i,j)s(i+r,j)> and <s(i,j)s(i,j+r)>
        corr_x = (spins * shift_x).mean(dim=(-2, -1))
        corr_y = (spins * shift_y).mean(dim=(-2, -1))

        corr_avg = 0.5 * (corr_x + corr_y)
        corr.append(corr_avg)

    return torch.stack(corr, dim=-1)  # shape: (..., N//2 + 1)


def plot_avg_magnetization(magnetizations, inv_T_values, algo_name, grid_size, J, h):

    # Create figure
    fig = plt.figure(figsize=(10, 6))

    # Plot magnetization curve
    plt.plot(1 / inv_T_values, magnetizations, "o-", label=algo_name)

    # Add theoretical critical temperature for 2D Ising model
    Tc = 2.269
    plt.axvline(x=Tc, color="gray", linestyle="--", label="Critical Temperature")

    plt.xlabel("Temperature (T)")
    plt.ylabel("Average Absolute Magnetization")
    plt.title(f"Ising Model: {grid_size}x{grid_size}, J={J}, h={h}")
    plt.legend()
    plt.grid(True)

    return fig
