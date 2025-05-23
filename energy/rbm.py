import torch
import torch.nn.functional as F
from .base import BaseSet
import matplotlib.pyplot as plt
import numpy as np
import os

from .metric import wasserstein
from .viz import *


class GaussianBernoulliRBM(BaseSet):
    """
    Implementation of Gaussian-Bernoulli Restricted Boltzmann Machine

    The visible layer consists of Gaussian units,
    and the hidden layer consists of Bernoulli units.
    """

    num_disc_tokens = 2

    def __init__(
        self,
        n_visible,
        n_hidden,
        sigma=1.0,
        device="cuda",
        num_samples=10000,
        seed=42,
        w_coeff=0.01,
        num_chains=100,
        burn_in=10000,
        thinning=100,
        param_file_name=None,
    ):
        """
        Args:
            n_visible: Number of visible units
            n_hidden: Number of hidden units
            sigma: Standard deviation of Gaussian visible layer
            device: Device to use for computation
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            w_coeff: Coefficient for the weight matrix
            num_chains: Number of parallel Gibbs sampling chains
            burn_in: Number of burn-in steps
            thinning: Number of thinning steps
        """
        super(GaussianBernoulliRBM, self).__init__(dim=n_visible + n_hidden)

        # Save the seed for reproducibility
        self.seed = seed

        # Create PyTorch RNG
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.torch_rng = torch.Generator(device=self.device)
        self.torch_rng.manual_seed(seed)

        self.n_conti_dim = self.n_visible = n_visible
        self.n_disc_dim = self.n_hidden = n_hidden
        self.sigma = sigma
        self.num_samples = num_samples
        self.sample_cache = None
        self.w_coeff = w_coeff

        self._param_file_name = param_file_name

        # Load weights and biases
        self._load_params(self.device)

        # Initialize sample cache
        self._initialize_cache(num_chains, burn_in, thinning)

    def split_into_visible_and_hidden(self, x):
        """Split the input into visible and hidden layers"""
        return x[..., : self.n_visible], x[..., self.n_visible :]

    def extract_hidden(self, x):
        """Extract the hidden layer from the input"""
        return x[..., self.n_visible :]

    @property
    def sample_file_name(self):
        return self.param_file_name.replace(".pt", f"_samples_{self.num_samples}.pt")

    @property
    def param_file_name(self):
        if self._param_file_name is None:
            return f"energy/rbm_params/rbm_nv{self.n_visible}_nh{self.n_hidden}_seed{self.seed}_sigma{self.sigma}_wcoeff{self.w_coeff}.pt"
        else:
            return self._param_file_name

    def _initialize_cache(self, num_chains, burn_in, thinning):
        """Initialize the sample cache by loading from file or generating new samples"""
        filename = self.sample_file_name
        try:
            self.sample_cache = torch.load(filename, map_location=self.device)
            assert self.sample_cache.shape == (
                self.num_samples,
                self.n_visible + self.n_hidden,
            )
            print(f"Sample cache loaded from {filename}")
        except (FileNotFoundError, IOError):
            print(
                f"Sample cache file not found, generating new samples with seed {self.seed}..."
            )
            self.sample_cache = self._generate_samples(
                self.num_samples, num_chains, burn_in, thinning
            )
            print("Sample cache generated")

            # Save generated samples
            os.makedirs("energy/rbm_params", exist_ok=True)
            torch.save(self.sample_cache, filename)
            print(f"Sample cache saved to {filename}")

    def _generate_samples(
        self,
        batch_size,
        num_chains=100,
        burn_in=10000,
        thinning=100,
    ):
        """Generate samples using 100 parallel Gibbs sampling chains"""
        samples_per_chain = (batch_size + num_chains - 1) // num_chains  # 올림 계산

        # Initialize visible layer states for all chains in parallel
        v = (
            self.sigma
            * torch.randn(
                num_chains, self.n_visible, device=self.device, generator=self.torch_rng
            )
            + self.v_bias
        )

        # Run burn-in phase in parallel for all chains
        for _ in range(burn_in):
            _, h = self.sample_h_given_v(v)
            _, v = self.sample_v_given_h(h)

        # Collect samples with thinning in parallel
        all_samples = []

        for _ in range(samples_per_chain):
            # Perform thinning steps
            for _ in range(thinning):
                _, h = self.sample_h_given_v(v)
                _, v = self.sample_v_given_h(h)

            # Concatenate v and h for all chains
            samples = torch.cat([v, h], dim=1)
            all_samples.append(samples)

        # Stack all samples and take the first batch_size samples
        all_samples = torch.cat(all_samples, dim=0)
        return all_samples[:batch_size]

    def _load_params(self, device):
        # Try to load parameters from file
        filename = self.param_file_name
        try:
            params = torch.load(filename, map_location=device)
            self.W = params["W"]
            self.v_bias = params["v_bias"]
            self.h_bias = params["h_bias"]
            print(f"Params loaded from {filename}")
        except (FileNotFoundError, IOError):
            # Initialize with random values using the torch RNG
            self.W = (
                torch.randn(
                    self.n_visible,
                    self.n_hidden,
                    device=device,
                    generator=self.torch_rng,
                )
                * self.w_coeff
            )
            self.v_bias = torch.randn(
                self.n_visible, device=device, generator=self.torch_rng
            )
            self.h_bias = torch.randn(
                self.n_hidden, device=device, generator=self.torch_rng
            )
            print(f"Params file not found, initialized randomly with seed {self.seed}")

            # Save generated parameters
            os.makedirs("energy/rbm_params", exist_ok=True)
            torch.save(
                {"W": self.W, "v_bias": self.v_bias, "h_bias": self.h_bias}, filename
            )

    def energy(self, x):
        """
        Energy function

        Args:
            v: Visible layer state [batch_size, n_visible]
            h: Hidden layer state [batch_size, n_hidden]

        Returns:
            Energy [batch_size]
        """
        v, h = self.split_into_visible_and_hidden(x)
        v_term = 0.5 * torch.sum((v - self.v_bias) ** 2, dim=-1) / (self.sigma**2)
        h_term = -torch.sum(self.h_bias * h, dim=-1)

        # v: [batch_size, n_visible], W: [n_visible, n_hidden], h: [batch_size, n_hidden]
        w_term = -torch.sum((v / (self.sigma**2)) @ self.W * h, dim=-1)

        return v_term + h_term + w_term

    def sample_h_given_v(self, v):
        """
        Sample hidden layer given visible layer state

        Args:
            v: Visible layer state [batch_size, n_visible]

        Returns:
            h_prob: Hidden layer activation probability [batch_size, n_hidden]
            h_sample: Hidden layer sample [batch_size, n_hidden]
        """
        # v: [batch_size, n_visible]
        # W: [n_visible, n_hidden]
        # h_bias: [n_hidden]
        h_prob = torch.sigmoid(F.linear(v / (self.sigma**2), self.W.t(), self.h_bias))
        h_sample = torch.bernoulli(h_prob, generator=self.torch_rng)

        return h_prob, h_sample

    def sample_v_given_h(self, h):
        """
        Sample visible layer given hidden layer state

        Args:
            h: Hidden layer state [batch_size, n_hidden]

        Returns:
            v_mean: Visible layer mean [batch_size, n_visible]
            v_sample: Visible layer sample [batch_size, n_visible]
        """
        # h: [batch_size, n_hidden]
        # W: [n_visible, n_hidden]
        # v_bias: [n_visible]
        v_mean = F.linear(h, self.W, self.v_bias)
        v_sample = v_mean + self.sigma * torch.randn(
            v_mean.shape, device=self.device, generator=self.torch_rng
        )

        return v_mean, v_sample

    def sample(self, batch_size):
        """
        Generate samples from RBM using cached samples.

        Args:
            batch_size: Number of samples to generate

        Returns:
            samples: Generated samples [batch_size, n_visible + n_hidden]
        """
        if self.sample_cache is None:
            self._initialize_cache()

        # Randomly select samples from cache
        indices = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        return self.sample_cache[indices]

    def v_mean(self, x):
        """
        Compute the mean of the visible layer given the hidden layer state
        """
        h = self.extract_hidden(x)
        return F.linear(h, self.W, self.v_bias)

    def visualize(self, samples, show_ground_truth=False):
        """
        Visualize RBM samples and their statistics

        Args:
            samples: Generated samples [batch_size, n_visible + n_hidden]
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

        # 2. Average activation of hidden units
        h = self.extract_hidden(samples)
        hidden_activations = h.mean(dim=-1).detach().cpu().numpy()
        if show_ground_truth and gt_samples is not None:
            gt_h = self.extract_hidden(gt_samples)
            gt_hidden_activations = gt_h.mean(dim=-1).detach().cpu().numpy()
        else:
            gt_hidden_activations = None

        hidden_act_hist = plot_histogram(
            hidden_activations,
            gt_hidden_activations,
            title="Hidden Unit Activation",
            xlabel="Average Activation",
            ylabel="Density",
            color="lightgreen",
            gt_color="lightgray",
        )

        # 3. 2D Projection plot
        projection_plot1 = plot_projection(
            samples,
            ground_truth=gt_samples if show_ground_truth else None,
            title="2D Projection of RBM Samples (dim 0, 1)",
            proj_dim1=0,
            proj_dim2=1,
            lim=(-10, 10, -5, 10),
        )

        projection_plot2 = plot_projection(
            samples,
            ground_truth=gt_samples if show_ground_truth else None,
            title="2D Projection of RBM Samples (dim 1, 2)",
            proj_dim1=1,
            proj_dim2=2,
        )

        return {
            "energy_hist": energy_hist,
            "hidden_act_hist": hidden_act_hist,
            "projection_plot1": projection_plot1,
            "projection_plot2": projection_plot2,
        }

    def metric(self, samples):
        energies = self.energy(samples)

        gt_samples = self.sample(len(samples))
        gt_energies = self.energy(gt_samples)

        # Energy distribution W1 distance
        energy_w1 = wasserstein(
            energies.detach().cpu().view(-1, 1),
            gt_energies.detach().cpu().view(-1, 1),
            power=1,
        )

        if self.n_visible == 2:
            sample_w2 = wasserstein(
                samples[..., :2],
                gt_samples[..., :2],
                power=2,
            )

        return {
            "energy_w1": energy_w1,
            "sample_w2": sample_w2,
        }
