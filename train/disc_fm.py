import torch
import numpy as np

import wandb
import argparse
import tqdm

from hydra.utils import instantiate
from flow_matching.loss import MixturePathGeneralizedKL

from architectures import *
from prob_path import *
from utils import *
from energy import *


def train(cfg, energy: BaseSet, logger: Logger):

    prob_denoiser = instantiate(cfg.model)(
        num_tokens=energy.num_tokens + 1,
        grid_size=cfg.energy.grid_size,
        seq_length=cfg.energy.grid_size * cfg.energy.grid_size,
    ).to(cfg.device)

    cfg.num_params = count_params(prob_denoiser)

    optim = init_optimizer(prob_denoiser, cfg.optim)

    mixture_path, solver = construct_masked_diffusion(
        prob_denoiser, energy.num_tokens, energy.ndim
    )

    loss_fn = MixturePathGeneralizedKL(path=mixture_path)

    for epoch in tqdm.trange(cfg.epoch):
        x1 = energy.sample(cfg.batch_size).to(cfg.device)
        t = torch.rand(cfg.batch_size, device=cfg.device)
        x0 = mixture_path.sample_x0(cfg.batch_size, device=cfg.device)
        path_sample = mixture_path.sample(x0, x1, t)

        # Estimate the prob velocity
        logits = prob_denoiser(path_sample.x_t, path_sample.t)

        optim.zero_grad()
        loss = loss_fn(logits, x1, path_sample.x_t, path_sample.t)
        loss.backward()
        optim.step()

        logger.log_loss({"train/loss": loss.item()})

        if epoch % 1000 == 0:
            with torch.no_grad():
                prob_denoiser.eval()

                # Evaluate the model
                step_size = 1 / cfg.num_steps
                x0 = mixture_path.sample_x0(2000, device=cfg.device)
                x1 = solver.sample(x0, step_size)

                sample_fig = energy.visualize(x1, show_ground_truth=True)
                logger.log_visual(sample_fig)
                logger.log_metric(energy.metric(x1))

                prob_denoiser.train()
