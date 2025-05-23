import torch
import tqdm
from hydra.utils import instantiate

from energy import *
from architectures import *
from prob_path import *
from utils import *


def train(cfg, energy: BaseSet, logger: Logger):

    drift = instantiate(cfg.model)(
        num_tokens=energy.num_disc_tokens + 1,
        discrete_dim=energy.n_disc_dim,
        continuous_dim=energy.n_conti_dim,
    ).to(cfg.device)

    logger.add_hyperparams({"num_params": count_params(drift)})

    optim = init_optimizer(drift, cfg.optim)

    path, solver = construct_joint_path(cfg, drift, energy)

    buffer = make_buffer(cfg, energy)

    lambda_conti = cfg.lambda_conti
    lambda_disc = cfg.lambda_disc

    for outer_iter in tqdm.trange(cfg.iter):
        x1 = eval_step(
            outer_iter,
            cfg,
            drift,
            path,
            solver,
            energy,
            logger,
            clip_cont_vel=cfg.clip_cont_vel,
        )

        if not cfg.use_gt_sample:
            buffer.add(x1, energy.log_reward(x1))
            eval_buffer(buffer, energy, logger)

        for innter_iter in range(cfg.inner_iter):

            # Sample the collocation points
            x1 = buffer.sample()[0].to(cfg.device)
            if cfg.conti_prob_path == "VE":
                t = torch.rand(x1.shape[0], device=cfg.device)
            else:
                t = torch.rand(x1.shape[0], device=cfg.device) * 0.97 + 0.03
            x0 = path.sample_x0(x1.shape[0], device=cfg.device)
            path_sample = path.sample(x0, x1, t)

            estimated_conti_vel, est_disc_denoiser = path.estimate_vel(
                path_sample.x_t,
                path_sample.t,
                energy.energy,
                num_mc_samples=cfg.num_mc_samples,
            )

            estimated_conti_vel = clip(estimated_conti_vel, cfg.clip_est, energy)

            model_conti_vel, model_disc_denoiser = drift(path_sample.x_t, path_sample.t)

            optim.zero_grad()
            conti_loss = vector_mse(model_conti_vel, estimated_conti_vel)
            disc_loss = matrix_mse(model_disc_denoiser, est_disc_denoiser)
            loss = (lambda_conti * conti_loss + lambda_disc * disc_loss).mean()
            loss.backward()
            optim.step()

            logger.log_loss(
                {
                    "loss": loss.item(),
                    "conti_loss": conti_loss.mean().item(),
                    "disc_loss": disc_loss.mean().item(),
                    **t_stratified_loss(disc_loss, path_sample.t, "disc"),
                    **t_stratified_loss(conti_loss, path_sample.t, "conti"),
                }
            )
