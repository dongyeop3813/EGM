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
from .ve_path import VEPath
from .joint_prob_path import JointEulerSolver

from utils import clip_norm, next_time_step, log_mean_exp

_EPS = 1e-8


class VEandMixtureJointPath(ProbPath):
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
        sigma_max: float,
        sigma_min: float,
    ):
        self.scheduler = scheduler
        self.n_conti_dim = n_conti_dim
        self.n_disc_dim = n_disc_dim
        self.num_disc_tokens = num_disc_tokens

        self.mixture_path = MaskedMixturePath(
            scheduler=scheduler,
            num_tokens=num_disc_tokens,
            seq_length=n_disc_dim,
        )

        self.ve_path = VEPath(
            n_conti_dim=n_conti_dim,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

    def split(self, x):
        return x[..., : self.n_conti_dim], x[..., self.n_conti_dim :].long()

    def sample_x0(self, batch_size, device):
        x_conti = self.ve_path.sample_x0(batch_size, device)
        x_disc = self.mixture_path.sample_x0(batch_size, device)

        return torch.cat([x_conti, x_disc], dim=-1)

    def sample(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> DiscretePathSample:
        x0_conti, x0_disc = self.split(x_0)
        x1_conti, x1_disc = self.split(x_1)

        x_conti = self.ve_path.sample(x0_conti, x1_conti, t)
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

        conti_vel = self.ve_path.cond_vel(xt_conti, t, x1_conti)
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

        conti_vel = self.ve_path.cond_vel_given_xr(xt_conti, t, xr_conti, step_size)
        disc_denoiser = self.mixture_path.cond_denoiser_given_xr(
            xt_disc, t.unsqueeze(-1), xr_disc, step_size
        )

        return conti_vel, disc_denoiser

    def naive_proposal(self, xt, t, num_samples=1000):
        """
        Args:
            xt: [B, Nc + Nd]
            t: [B]

        Returns:
            xr: [B, num_samples, Nc + Nd]
        """
        assert xt.dim() == 2 and t.dim() == 1

        xt_conti, xt_disc = self.split(xt)
        x_conti = self.ve_path.naive_proposal(xt_conti, t, num_samples=num_samples)
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

        x_conti = self.ve_path.small_step_proposal(
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

        xt_conti, xt_disc = self.split(xt)

        x1 = self.naive_proposal(xt, t, num_mc_samples)
        logits = -energy(x1)
        est = log_mean_exp(logits, dim=1)

        est += self.mixture_path.proposal_logZ(xt_disc, t, torch.ones_like(t))

        return est

    def estimate_vel(self, xt, t, energy, num_mc_samples=1000):
        """
        xt: [B, Nc + Nd]
        t: [B]
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


def construct_joint_ve_mask_path(
    drift,
    n_conti_dim,
    n_disc_dim,
    num_disc_tokens,
    sigma_max,
    sigma_min,
):
    path = VEandMixtureJointPath(
        scheduler=PolynomialConvexScheduler(n=1.0),
        n_conti_dim=n_conti_dim,
        n_disc_dim=n_disc_dim,
        num_disc_tokens=num_disc_tokens,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
    )

    solver = JointEulerSolver(
        drift=drift,
        path=path,
        solver_start_time=0.00,
    )

    return path, solver
