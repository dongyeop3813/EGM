import math

import numpy as np
import torch

from flow_matching.path import ProbPath, DiscretePathSample
from flow_matching.solver import ODESolver
from utils.etc import next_time_step, log_mean_exp

_EPS = 1e-8


class GeometricNoiseSchedule:
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = self.sigma_max / self.sigma_min

    def g(self, t):
        # Let sigma_d = sigma_max / sigma_min
        # Then g(t) = sigma_min * sigma_d^t * sqrt{2 * log(sigma_d)}
        # See Eq 192 in https://arxiv.org/pdf/2206.00364.pdf
        t = 1 - t
        return (
            self.sigma_min
            * (self.sigma_diff**t)
            * ((2 * np.log(self.sigma_diff)) ** 0.5)
        )

    # This is sigma squared.
    def h(self, t):
        t = 1 - t
        # Let sigma_d = sigma_max / sigma_min
        # Then h(t) = \int_0^t g(z)^2 dz = sigma_min * sqrt{sigma_d^{2t} - 1}
        # see Eq 199 in https://arxiv.org/pdf/2206.00364.pdf
        return (self.sigma_min * (((self.sigma_diff ** (2 * t)) - 1) ** 0.5)) ** 2

    def sigma_t(self, t):
        return self.h(t) ** 0.5

    def dsigma_t(self, t):
        return -0.5 * self.g(t) ** 2 / self.sigma_t(t)


class VEPath(ProbPath):
    def __init__(
        self,
        n_conti_dim: int,
        sigma_max: float,
        sigma_min: float,
    ):
        """
        sigma_max: max noise level
        sigma_min: min noise level

        N(x_t; x_1, h(t)I)
        """

        super().__init__()
        self.n_conti_dim = n_conti_dim
        self.noise_schedule = GeometricNoiseSchedule(sigma_min, sigma_max)

    def sample_x0(self, batch_size, device):
        noise = torch.randn((batch_size, self.n_conti_dim), device=device)
        return noise * self.noise_schedule.sigma_t(0)

    def sample(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> DiscretePathSample:
        eps = x_0 / self.noise_schedule.sigma_t(0)
        x_t = x_1 + self.noise_schedule.sigma_t(t).unsqueeze(-1) * eps
        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)

    def cond_vel(self, xt, t, x1):
        """
        Compute the conditional velocity.

        Args:
            xt: [batch_dim..., D]
            t: [batch_dim...]
            x1: [batch_dim..., D]
        Expected shape of the output tensor:
        cond_vel: [batch_dim..., D]
        """
        coeff = self.noise_schedule.dsigma_t(t) / self.noise_schedule.sigma_t(t)
        return coeff.unsqueeze(-1) * (xt - x1)

    def cond_vel_given_xr(self, xt, t, xr, step_size):
        """
        Compute the conditional velocity given xr.
        Args:
            xt: [batch_dim..., D]
            t: [batch_dim...]
            xr: [batch_dim..., D]
            step_size: float
        Expected shape of the output tensor:
        cond_vel: [batch_dim..., D]
        """

        r = next_time_step(t, step_size)

        transition_var = self.noise_schedule.h(t) - self.noise_schedule.h(r)
        dsigma_t = self.noise_schedule.dsigma_t(t)
        sigma_t = self.noise_schedule.sigma_t(t)

        coeff = (dsigma_t * sigma_t) / (transition_var + _EPS)

        return coeff.unsqueeze(-1) * (xt - xr)

    def naive_proposal(self, xt, t, energy=None, num_samples=1000):
        """
        xt: [B, D]
        t: [B]
        x1: [B, num_samples, D]
        """
        assert xt.dim() == 2 and t.dim() == 1

        B, D = xt.shape

        sigma_t = self.noise_schedule.sigma_t(t)

        eps = torch.randn((B, num_samples, D), device=xt.device)
        x1 = xt.unsqueeze(1) + sigma_t[..., None, None] * eps

        if energy is not None:
            logits = -energy(x1)
            weight = torch.softmax(logits, dim=-1)
            return x1, weight.detach()
        else:
            return x1

    def small_step_proposal(
        self, xt, t, energy=None, flow_model=None, num_samples=1000, step_size=1e-2
    ):
        """
        Generate samples from q_{r|t}(x_r| x_t) \propto p_{t|r}(x_t | x_r).
        Here, r is set to t + step_size.

        Args:
            xt: [B, D] - state at current time t
            t: [B] - current time
            energy: energy function
            flow_model: flow model
            num_samples: number of samples to generate
            step_size: time step size

        Returns:
            xr: [B, num_samples, D] - proposed samples
            weight: [B, num_samples] - weights for each sample
        """
        assert xt.dim() == 2 and t.dim() == 1

        B, D = xt.shape

        r = next_time_step(t, step_size)

        eps = torch.randn((B, num_samples, D), device=xt.device)
        sigma_r = self.noise_schedule.sigma_t(r)
        sigma_t = self.noise_schedule.sigma_t(t)
        std = (sigma_t**2 - sigma_r**2) ** 0.5

        xr = xt.unsqueeze(1) + std[..., None, None] * eps

        if flow_model is not None:
            expanded_r = r[..., None].expand(B, num_samples)

            # [B, num_samples, D] x [B, num_samples] -> [B, num_samples]
            flow = flow_model(xr, expanded_r)

            assert flow.shape == (B, num_samples)

            weight = torch.softmax(flow, dim=-1)

            return xr, weight.detach()
        else:
            return xr

    def estimate_vel(self, xt, t, energy, num_mc_samples=1000):
        """
        xt: [B, D]
        t: [B]
        """
        assert xt.dim() == 2 and t.dim() == 1

        B, D = xt.shape

        # x1: [B, num_samples, D]
        x1, weight = self.naive_proposal(
            xt,
            t,
            energy=energy,
            num_samples=num_mc_samples,
        )

        assert weight.shape == (B, num_mc_samples)
        xt = xt[:, None, :].expand(B, num_mc_samples, D)
        t = t[:, None].expand(B, num_mc_samples)
        weight = weight[..., None].expand(B, num_mc_samples, 1)

        return (weight * self.cond_vel(xt, t, x1)).sum(dim=1)

    def bootstrap_estimate_vel(
        self,
        xt,
        t,
        energy,
        flow_model,
        num_mc_samples=1000,
        step_size=1e-2,
    ):
        """
        xt: [B, D]
        t: [B]
        """
        assert xt.dim() == 2 and t.dim() == 1

        B, D = xt.shape

        xr, weight = self.small_step_proposal(
            xt, t, energy, flow_model, num_samples=num_mc_samples, step_size=step_size
        )

        assert weight.shape == (B, num_mc_samples)
        xt = xt[:, None, :].expand(B, num_mc_samples, D)
        t = t[:, None].expand(B, num_mc_samples)
        weight = weight[..., None].expand(B, num_mc_samples, 1)

        return (weight * self.cond_vel_given_xr(xt, t, xr, step_size)).sum(dim=1)

    def estimate_flow(self, xt, t, energy, num_mc_samples=1000):
        """
        xt: [B, D]
        t: [B]
        """
        assert xt.dim() == 2 and t.dim() == 1

        # x1: [B, num_samples, D]
        x1 = self.naive_proposal(
            xt,
            t,
            energy=None,
            num_samples=num_mc_samples,
        )

        logits = -energy(x1)
        est = log_mean_exp(logits, dim=-1)
        return est


def construct_ve_path(
    drift,
    n_conti_dim: int,
    sigma_max: float,
    sigma_min: float,
):
    path = VEPath(n_conti_dim, sigma_max, sigma_min)
    solver = ODESolver(drift)
    return path, solver
