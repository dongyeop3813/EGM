import torch
import tqdm
from hydra.utils import instantiate

from energy import *
from architectures import *
from prob_path import *
from utils import *


def train(cfg, energy: BaseSet, logger: Logger):

    if isinstance(energy, GMM25):
        drift = instantiate(cfg.model)(
            dim=energy.data_ndim,
        ).to(cfg.device)
    else:
        drift = instantiate(cfg.model)(
            n_particles=energy.n_particles,
            n_dimension=energy.spatial_dim,
        ).to(cfg.device)

    logger.add_hyperparams({"num_params": count_params(drift)})

    optim = init_optimizer(drift, cfg.optim)

    path, solver = construct_conti_path(cfg, drift, energy)

    buffer = make_buffer(cfg, energy)

    for outer_iter in tqdm.trange(cfg.iter):
        x1 = eval_step(outer_iter, cfg, drift, path, solver, energy, logger)

        if not cfg.use_gt_sample:
            buffer.add(x1, energy.log_reward(x1))
            eval_buffer(buffer, energy, logger)

        for innter_iter in range(cfg.inner_iter):

            # Sample the collocation points
            x1 = buffer.sample()[0].to(cfg.device)
            t = torch.rand(x1.shape[0], device=cfg.device)
            x0 = path.sample_x0(x1.shape[0], device=cfg.device)
            path_sample = path.sample(x0, x1, t)
            xt = path_sample.x_t

            estimated_vel = path.estimate_vel(
                xt,
                t,
                energy.energy,
                num_mc_samples=cfg.num_mc_samples,
            )

            estimated_vel = clip(estimated_vel, cfg.clip_est, energy)

            vel = drift(xt, t)

            optim.zero_grad()
            loss = mse_loss(vel, estimated_vel)
            loss.backward()
            optim.step()

            logger.log_loss({"loss": loss.item()})
