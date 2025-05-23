from hydra.utils import instantiate
import yaml
from omegaconf import OmegaConf

from omegaconf import DictConfig

import os
import torch
from prob_path import *


def load_discrete_model(output_dir, device):
    config_path = os.path.join(output_dir, ".hydra/config.yaml")

    with open(config_path, "r") as f:
        config_dict = OmegaConf.load(f)

    energy = instantiate(config_dict.energy)

    model = instantiate(config_dict.model)(
        num_tokens=energy.num_tokens + 1,
        grid_size=config_dict.energy.grid_size,
        seq_length=config_dict.energy.grid_size * config_dict.energy.grid_size,
    ).to(device)

    model_path = os.path.join(output_dir, "model_it99.pt")
    model.load_state_dict(torch.load(model_path))

    path, solver = construct_masked_diffusion(
        model,
        energy.num_tokens,
        energy.grid_size**2,
        scheduler=PolynomialConvexScheduler(n=1.0),
    )

    return energy, model, path, solver


def load_joint_model(output_dir, device):
    config_path = os.path.join(output_dir, ".hydra/config.yaml")

    with open(config_path, "r") as f:
        config_dict = OmegaConf.load(f)

    energy = instantiate(config_dict.energy)

    model = instantiate(config_dict.model)(
        num_tokens=energy.num_disc_tokens + 1,
        discrete_dim=energy.n_disc_dim,
        continuous_dim=energy.n_conti_dim,
    ).to(device)

    model_path = os.path.join(output_dir, "model_it99.pt")
    model.load_state_dict(torch.load(model_path))

    path, solver = construct_joint_path(config_dict, model, energy)

    return energy, model, path, solver


def magnetization(x):
    if x.dim() == 3:
        return x.float().mean(dim=(-1, -2))
    elif x.dim() == 2:
        return ((x - 0.5) * 2).mean(dim=-1)


def choice_samples(samples, num_samples):
    return samples[torch.randint(0, len(samples), (num_samples,))]
