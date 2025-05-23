from .masked_mixture_path import *
from .joint_prob_path import *
from .cond_ot_path import *
from .ve_path import *
from .joint_ve_mask import *


def construct_conti_path(cfg, drift, energy):
    if cfg.conti_prob_path == "cond_ot":
        path, solver = construct_cond_ot_path(
            drift,
            energy.data_ndim,
            cfg.conti_prior_sigma,
            cfg.solver_start_time,
        )
    elif cfg.conti_prob_path == "VE":
        path, solver = construct_ve_path(
            drift,
            energy.data_ndim,
            cfg.ve_sigma_max,
            cfg.ve_sigma_min,
        )
    else:
        raise ValueError(f"Unknown conti_prob_path: {cfg.conti_prob_path}")

    return path, solver


def construct_joint_path(cfg, drift, energy):

    if cfg.conti_prob_path == "cond_ot":
        path, solver = construct_joint_ot_path(
            drift,
            energy.n_conti_dim,
            energy.n_disc_dim,
            energy.num_disc_tokens,
            cfg.conti_prior_sigma,
            cfg.solver_start_time,
        )
    elif cfg.conti_prob_path == "VE":
        path, solver = construct_joint_ve_mask_path(
            drift,
            energy.n_conti_dim,
            energy.n_disc_dim,
            energy.num_disc_tokens,
            cfg.ve_sigma_max,
            cfg.ve_sigma_min,
        )
    else:
        raise ValueError(f"Unknown conti_prob_path: {cfg.conti_prob_path}")

    return path, solver
