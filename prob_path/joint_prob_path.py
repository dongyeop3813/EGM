import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from flow_matching.path import ProbPath, DiscretePathSample
from flow_matching.path.scheduler import ConvexScheduler, PolynomialConvexScheduler
from flow_matching.solver import Solver
from flow_matching.utils.categorical_sampler import categorical

from .masked_mixture_path import MaskedMixturePath
from .cond_ot_path import CondOTGaussianPath

from utils import clip_norm, next_time_step, log_mean_exp

_EPS = 1e-8


class OTandMixtureJointPath(ProbPath):
    """
    This class is a joint path of conditional optimal transport and mixture of discrete probability.
    The continuous part is conditional OT path with the Gaussian prior and the discrete part is the masked mixture path.
    """

    def __init__(
        self,
        scheduler: ConvexScheduler,
        n_conti_dim: int,
        n_disc_dim: int,
        num_disc_tokens: int,
        conti_prior_sigma: float = 1.0,
    ):
        self.scheduler = scheduler
        self.conti_prior_sigma = conti_prior_sigma
        self.n_conti_dim = n_conti_dim
        self.n_disc_dim = n_disc_dim
        self.num_disc_tokens = num_disc_tokens

        self.mixture_path = MaskedMixturePath(
            scheduler=scheduler,
            num_tokens=num_disc_tokens,
            seq_length=n_disc_dim,
        )

        self.ot_path = CondOTGaussianPath(
            n_conti_dim=n_conti_dim,
            conti_prior_sigma=conti_prior_sigma,
        )

    def split(self, x):
        return x[..., : self.n_conti_dim], x[..., self.n_conti_dim :].long()

    def sample_x0(self, batch_size, device):
        x_conti = self.ot_path.sample_x0(batch_size, device)
        x_disc = self.mixture_path.sample_x0(batch_size, device)

        return torch.cat([x_conti, x_disc], dim=-1)

    def sample(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> DiscretePathSample:
        x0_conti, x0_disc = self.split(x_0)
        x1_conti, x1_disc = self.split(x_1)

        x_conti = self.ot_path.sample(x0_conti, x1_conti, t)
        x_disc = self.mixture_path.sample(x0_disc, x1_disc, t)

        x_t = torch.cat([x_conti.x_t, x_disc.x_t], dim=-1)

        return DiscretePathSample(
            x_1=x_1,
            x_0=x_0,
            t=t,
            x_t=x_t,
        )

    def cond_vel(self, xt, t, x1):
        """
        Compute the conditional velocity.

        Args:
            xt: [batch_dim..., D]
            t: [batch_dim...]
            x1: [batch_dim..., D]
        Expected shape of the output tensor:
            cond_vel: [batch_dim..., D]
            cond_denoiser: [batch_dim..., D, num_tokens]

        """
        assert t.shape == xt.shape[:-1] and t.shape == x1.shape[:-1]

        x1_conti, x1_disc = self.split(x1)
        xt_conti, xt_disc = self.split(xt)

        conti_vel = self.ot_path.cond_vel(xt_conti, t, x1_conti)
        disc_denoiser = self.mixture_path.cond_denoiser(
            xt_disc, t.unsqueeze(-1), x1_disc
        )

        return conti_vel, disc_denoiser

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
            cond_denoiser: [batch_dim..., D, num_tokens]
        """
        assert t.shape == xt.shape[:-1] and t.shape == xr.shape[:-1]

        xt_conti, xt_disc = self.split(xt)
        xr_conti, xr_disc = self.split(xr)

        conti_vel = self.ot_path.cond_vel_given_xr(xt_conti, t, xr_conti, step_size)
        disc_denoiser = self.mixture_path.cond_denoiser_given_xr(
            xt_disc, t.unsqueeze(-1), xr_disc, step_size
        )

        return conti_vel, disc_denoiser

    def naive_proposal(self, xt, t, num_samples=1000):
        """
        Args:
            xt: [B, Nc + Nd]
            t: [B, 1]

        Returns:
            xr: [B, num_samples, Nc + Nd]
        """
        assert t.dim() == 1
        assert xt.dim() == 2

        xt_conti, xt_disc = self.split(xt)
        x_conti = self.ot_path.naive_proposal(xt_conti, t, num_samples=num_samples)
        x_disc = self.mixture_path.naive_proposal(xt_disc, t, num_samples=num_samples)

        return torch.cat([x_conti, x_disc], dim=-1)

    def small_step_proposal(
        self, xt, t, energy, flow_model, num_samples=1000, step_size=1e-2
    ):
        """
        Args:
            xt: [B, Nc + Nd]
            t: [B]
        """
        assert t.dim() == 1
        assert xt.dim() == 2

        B, D = xt.shape

        xt_conti, xt_disc = self.split(xt)

        r = next_time_step(t, step_size)

        x_conti = self.ot_path.small_step_proposal(
            xt_conti, t, num_samples=num_samples, step_size=step_size
        )
        x_disc = self.mixture_path.small_step_proposal(
            xt_disc, t, num_samples=num_samples, step_size=step_size
        )

        xr = torch.cat([x_conti, x_disc], dim=-1)

        if flow_model is not None:
            with torch.no_grad():
                r_expanded = r[:, None].expand(B, num_samples)
                weight = torch.softmax(flow_model(xr, r_expanded), dim=-1)
            return xr, weight.detach()
        else:
            return xr

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
        Estimate the velocity using bootstrap method.
        """
        assert t.dim() == 1
        assert xt.dim() == 2

        B, D = xt.shape

        xr, weight = self.small_step_proposal(xt, t, energy, flow_model, num_mc_samples)

        xt = xt[:, None, :].expand(B, num_mc_samples, D)
        t = t[:, None].expand(B, num_mc_samples)

        conti_vel, disc_denoiser = self.cond_vel_given_xr(xt, t, xr, step_size)

        conti_vel = (weight[..., None] * conti_vel).sum(dim=1)
        disc_denoiser = (weight[..., None, None] * disc_denoiser).sum(dim=1)

        return conti_vel, disc_denoiser

    def estimate_flow(self, xt, t, energy, num_mc_samples=1000):
        """
        Estimate the flow with naive proposal.
        """
        assert t.dim() == 1
        assert xt.dim() == 2

        x1 = self.naive_proposal(xt, t, num_mc_samples)
        logits = -energy(x1)
        est = log_mean_exp(logits, dim=1)

        xt_conti, xt_disc = self.split(xt)

        est += self.mixture_path.proposal_logZ(xt_disc, t, torch.ones_like(t))

        return est

    def estimate_vel(self, xt, t, energy, num_mc_samples=1000):
        """
        xt: [B, Nc + Nd]
        t: [B, 1]
        """
        assert t.dim() == 1
        assert xt.dim() == 2

        B, D = xt.shape

        x1 = self.naive_proposal(xt, t, num_mc_samples)

        logits = -energy(x1)
        weight = torch.softmax(logits, dim=-1)

        xt = xt[:, None, :].expand(B, num_mc_samples, D)
        t = t[:, None].expand(B, num_mc_samples)

        conti_cond_vel, disc_cond_denoiser = self.cond_vel(xt, t, x1)

        conti_vel = (weight[..., None] * conti_cond_vel).sum(dim=1)
        disc_denoiser = (weight[..., None, None] * disc_cond_denoiser).sum(dim=1)

        return conti_vel, disc_denoiser


class JointEulerSolver(Solver):
    def __init__(
        self,
        drift: nn.Module,
        path: OTandMixtureJointPath,
        solver_start_time: float = 0.01,
    ):
        super().__init__()
        self.drift = drift
        self.path = path
        self.solver_start_time = solver_start_time

    def sample(
        self,
        x_init: torch.Tensor,
        step_size: Optional[float] = None,
        clip_cont_vel: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Use Euler method to sample from the joint space of continuous and discrete spaces.

        Args:
            x_0: initial state [batch_size, n_conti_dim + n_disc_dim]
            step_size: time step size

        Returns:
            x_1: final state [batch_size, n_conti_dim + n_disc_dim]
        """
        if step_size is None:
            step_size = 0.01  # default to 100 steps

        start_time = self.solver_start_time
        end_time = 1.0

        x_t = x_init
        bsz = x_init.shape[0]
        num_steps = int((end_time - start_time) / step_size)

        # Ignore t = 0 since naive OT proposal is invalid.
        time_grid = torch.linspace(start_time, end_time, num_steps, device=x_t.device)
        num_steps = len(time_grid) - 1

        num_disc_tokens = self.path.mixture_path.num_tokens

        for i, t in enumerate(time_grid[:-1]):
            # 1. predict drift
            conti_vel, disc_denoiser = self.drift(x_t, t.unsqueeze(0).expand(bsz, -1))

            x_conti, x_disc = self.path.split(x_t)
            x_disc = x_disc.long()

            # 2. update continuous space
            if clip_cont_vel is not None:
                conti_vel = clip_norm(conti_vel, clip_cont_vel)
            x_conti = x_conti + conti_vel * step_size

            # 3. update discrete space
            x_1 = categorical(disc_denoiser.to(dtype=torch.float32))
            if i == num_steps - 1:
                x_disc = x_1
            else:
                # Compute u_t(x|x_t,x_1)
                scheduler_output = self.path.mixture_path.scheduler(t=t)

                k_t = scheduler_output.alpha_t
                d_k_t = scheduler_output.d_alpha_t

                delta_1 = F.one_hot(x_1.long(), num_classes=num_disc_tokens).to(
                    k_t.dtype
                )
                u = d_k_t / (1 - k_t) * delta_1

                # Set u_t(x_t|x_t,x_1) = 0
                delta_t = F.one_hot(x_disc, num_classes=num_disc_tokens)
                u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)

                # Sample x_t ~ u_t( \cdot |x_t,x_1)
                intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0

                mask_jump = torch.rand(
                    size=x_disc.shape, device=x_disc.device
                ) < 1 - torch.exp(-step_size * intensity)

                if mask_jump.sum() > 0:
                    x_disc[mask_jump] = categorical(
                        u[mask_jump].to(dtype=torch.float32)
                    )

            # 4. combine state
            x_t = torch.cat([x_conti, x_disc], dim=-1)

        return x_t


def construct_joint_ot_path(
    drift,
    n_conti_dim,
    n_disc_dim,
    num_disc_tokens,
    conti_prior_sigma=1.0,
    solver_start_time=0.01,
):
    path = OTandMixtureJointPath(
        scheduler=PolynomialConvexScheduler(n=1.0),
        n_conti_dim=n_conti_dim,
        n_disc_dim=n_disc_dim,
        num_disc_tokens=num_disc_tokens,
        conti_prior_sigma=conti_prior_sigma,
    )

    solver = JointEulerSolver(
        drift=drift,
        path=path,
        solver_start_time=solver_start_time,
    )

    return path, solver
