import numpy as np
import torch

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from utils.etc import next_time_step

_EPS = 1e-8


class CondOTGaussianPath(CondOTProbPath):
    def __init__(self, n_conti_dim: int, conti_prior_sigma: float = 1.0):
        super().__init__()
        self.n_conti_dim = n_conti_dim
        self.conti_prior_sigma = conti_prior_sigma

    def sample_x0(self, batch_size, device):
        return (
            torch.randn((batch_size, self.n_conti_dim), device=device)
            * self.conti_prior_sigma
        )

    def cond_vel(self, xt, t, x1):
        return self.target_to_velocity(x1, xt, t.unsqueeze(-1))

    def sigma_t_given_r(self, t, r):
        return ((1 - t) ** 2) - (t**2) / (r**2) * ((1 - r) ** 2)

    def dsigma_t_given_r(self, t, r):
        return -2 * (1 - t) - 2 * t / (r**2) * ((1 - r) ** 2)

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
        t = t.unsqueeze(-1)
        r = next_time_step(t, step_size)

        sigma_t = self.sigma_t_given_r(t, r)
        dsigma_t = self.dsigma_t_given_r(t, r)

        cond_vel = (xr / r) + 0.5 * dsigma_t / sigma_t * (xt - (t / r) * xr)
        return cond_vel

    def naive_proposal(self, xt, t, energy=None, num_samples=1000):
        """
        xt: [B, D]
        t: [B]
        x1: [B, num_samples, D]
        """
        assert xt.dim() == 2 and t.dim() == 1

        t = t.unsqueeze(-1)

        B, D = xt.shape

        scheduler_output = self.scheduler(t)
        sigma_t = scheduler_output.sigma_t.unsqueeze(1)
        alpha_t = scheduler_output.alpha_t.unsqueeze(1)
        xt = xt.unsqueeze(1)

        x0 = torch.randn((B, num_samples, D), device=xt.device)

        x1 = xt * (1 / (alpha_t + _EPS)) + x0 * (sigma_t / (alpha_t + _EPS))

        if energy is not None:
            logits = -energy(x1)
            weight = torch.softmax(logits, dim=-1)
            return x1, weight.detach()
        else:
            return x1

    def small_step_proposal(
        self,
        xt,
        t,
        energy=None,
        flow_model=None,
        num_samples=1000,
        step_size=1e-2,
    ):
        """
        Generate samples from q_{r|t}(x_r| x_t) \propto p_{t|r}(x_t | x_r).
        Here, r is set to t + step_size.

        Args:
            xt: [B, D]
            t: [B]
            energy: energy function
            num_samples: number of samples to draw
            step_size: step size
        Returns:
            xr: [B, num_samples, D]
        """
        assert xt.dim() == 2 and t.dim() == 1

        B, D = xt.shape
        xt = xt.unsqueeze(1).expand(B, num_samples, D)

        r = next_time_step(t, step_size)
        sigma_t = self.sigma_t_given_r(t, r)[:, None, None]
        ratio = (r / t)[:, None, None]

        xr = ratio * xt + ratio * torch.sqrt(sigma_t) * torch.randn_like(xt)

        if flow_model is not None:
            expanded_r = r[..., None].expand(B, num_samples)
            weight = torch.softmax(flow_model(xr, expanded_r), dim=-1)
            return xr, weight.detach()
        else:
            return xr

    def p_t_given_1(self, xt, t, x1):
        """
        p(t | x1)

        xt: [..., D]
        t: [..., 1]
        x1: [..., D]
        """
        # Normal distirbution N(xt; alpha_t * x1, sigma_t^2 * I)
        assert t.shape[-1] == 1

        scheduler_output = self.scheduler(t)
        sigma_t = scheduler_output.sigma_t
        alpha_t = scheduler_output.alpha_t

        mean = alpha_t * x1
        var = sigma_t**2

        D = xt.shape[-1]
        log_prob = -0.5 * (
            ((xt - mean) ** 2).sum(dim=-1) / var.squeeze(-1)
            + D * torch.log(var.squeeze(-1))
            + D * np.log(2 * np.pi)
        )

        return log_prob

    def ais_proposal(self, xt, t, energy, num_samples=1000, num_steps=1000):
        """
        Annealed Importance Sampling proposal distribution.
        Samples are drawn from a distribution that interpolates between
        naive proposal and energy-based distribution using time t.

        Args:
            xt: [B, D] - current state
            t: [B, 1] - time step
            energy: callable - energy function
            num_samples: int - number of samples to draw

        Returns:
            x1: [B, num_samples, D] - proposed samples
            weight: [B, num_samples] - importance weights
        """
        # Get naive proposal samples
        dt = 1.0 / num_steps
        sigma = 1e-2
        B = xt.shape[0]

        def annealed_energy(z, i):
            # p_annealing_i(z) = p_t|1(xt|z) * p_1(z)^i
            log_p_t_given_1 = self.p_t_given_1(xt.unsqueeze(1), t.unsqueeze(1), z)
            return i * dt * energy(z) - log_p_t_given_1

        def annealed_score(z, i):
            with torch.no_grad():
                copy_z = z.detach().clone()
                copy_z.requires_grad = True
                with torch.enable_grad():
                    (-annealed_energy(copy_z, i)).sum().backward()
                grad_energy = copy_z.grad.data
            return grad_energy

        def ais_step(prev_z, i):
            # eps: [B, num_samples, D]
            eps = torch.randn_like(prev_z)

            # next_z: [B, num_samples, D]
            next_z = (
                prev_z
                + 0.5 * (sigma**2) * annealed_score(prev_z, i) * dt
                + eps * np.sqrt(dt) * sigma
            )

            # log_weight: [B, num_samples]
            log_weight = -dt * energy(next_z)

            return next_z, log_weight

        # total_log_weight: [B, num_samples]
        total_log_weight = torch.zeros(B, num_samples, device=xt.device)

        # z0: [B, num_samples, D]
        z0 = self.naive_proposal(xt, t, num_samples)
        zi = z0
        for i in range(num_steps):
            zi, log_w = ais_step(zi, i)
            total_log_weight += log_w

        # weight: [B, num_samples]
        weight = torch.softmax(total_log_weight, dim=-1)
        return zi, weight

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

        xt = xt[:, None, :].expand(B, num_mc_samples, D)
        t = t[:, None].expand(B, num_mc_samples)
        weight = weight[..., None]

        return (weight * self.cond_vel(xt, t, x1)).sum(dim=1)

    def estimate_vel_with_ais(
        self,
        xt,
        t,
        energy,
        num_samples=1000,
        num_steps=1000,
    ):
        """
        xt: [B, D]
        t: [B, 1]
        """
        # x1: [B, num_samples, D]
        x1, weight = self.ais_proposal(xt, t, energy, num_samples, num_steps)

        # cond_vel: [B, num_samples, D]
        cond_vel = self.cond_vel(
            xt[:, None, :],  # xt: [B, D] -> [B, 1, D]
            t[:, None],  # t: [B, 1] -> [B, 1, 1]
            x1,  # x1: [B, num_samples, D]
        )

        return (weight[..., None] * cond_vel).sum(dim=1)

    def perturbed_buffer_proposal(
        self,
        xt,
        t,
        buffer_data,
        num_samples=1000,
        sigma=0.01,
    ):
        """
        xt: [B, D]
        t: [B, 1]
        x1: [B, num_samples, D]
        """
        bsz, D = xt.shape

        def gmm_sample(modes, sigma):
            num_modes = modes.shape[0]
            indices = torch.randint(num_modes, (bsz, num_samples))
            x1 = modes[indices]
            x1 = x1 + torch.randn_like(x1) * sigma
            return x1

        def gmm_log_prob(x, modes, sigma):
            # x: [bsz, num_samples, D]
            # modes: [num_modes, D]

            # [bsz, num_samples, num_modes, D]
            diff = x.unsqueeze(2) - modes.unsqueeze(0).unsqueeze(0)

            # [bsz, num_samples, num_modes]
            log_probs_per_mode = -0.5 * (diff**2).sum(dim=-1) / sigma**2

            # [bsz, num_samples]
            log_prob = torch.logsumexp(log_probs_per_mode, dim=2)

            return log_prob

        x1 = gmm_sample(buffer_data, sigma)
        log_prob = gmm_log_prob(x1, buffer_data, sigma)

        return x1, log_prob

    def estimate_vel_with_buffer(
        self,
        xt,
        t,
        energy,
        buffer_data,
        num_mc_samples=1000,
        sigma=0.01,
    ):
        """
        xt: [B, D]
        t: [B, 1]
        """

        # x1: [B, num_samples, D]
        # log_prob: [B, num_samples]
        x1, log_prob = self.perturbed_buffer_proposal(
            xt,
            t,
            buffer_data,
            num_samples=num_mc_samples,
            sigma=sigma,
        )

        # log_unnormalized_prob: [B, num_samples]
        log_pt_given_1 = self.p_t_given_1(
            xt.unsqueeze(1).expand(-1, num_mc_samples, -1),
            t.unsqueeze(1).expand(-1, num_mc_samples, -1),
            x1,
        )
        log_unnormalized_prob = log_pt_given_1 - energy(x1)

        log_weight = log_unnormalized_prob - log_prob
        weight = torch.softmax(log_weight, dim=-1)

        cond_vel = self.cond_vel(xt[:, None, :], t[:, None], x1)
        return (weight[..., None] * cond_vel).sum(dim=1)


def construct_cond_ot_path(
    drift,
    n_conti_dim: int,
    conti_prior_sigma: float = 1.0,
):
    path = CondOTGaussianPath(n_conti_dim, conti_prior_sigma)
    solver = ODESolver(drift)
    return path, solver
