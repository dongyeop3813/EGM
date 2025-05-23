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

        flow_model = instantiate(cfg.flow_model)(
            dim=energy.data_ndim,
        ).to(cfg.device)
    else:
        drift = instantiate(cfg.model)(
            n_particles=energy.n_particles,
            n_dimension=energy.spatial_dim,
        ).to(cfg.device)

        flow_model = instantiate(cfg.flow_model)(
            n_particles=energy.n_particles,
            n_dimension=energy.spatial_dim,
        ).to(cfg.device)

    logger.add_hyperparams({"num_params": count_params(drift)})
    logger.add_hyperparams({"num_params_flow": count_params(flow_model)})

    optim = init_optimizer(drift, cfg.optim)
    optim_flow = init_optimizer(flow_model, cfg.optim_flow)

    path, solver = construct_conti_path(cfg, drift, energy)

    buffer = make_buffer(cfg, energy)

    flow_loss_fn = make_loss_fn(cfg.flow_loss_type)

    for outer_iter in tqdm.trange(cfg.iter):
        x1 = eval_step(outer_iter, cfg, drift, path, solver, energy, logger)

        if not cfg.use_gt_sample:
            buffer.add(x1, energy.log_reward(x1))
            eval_buffer(buffer, energy, logger)

        if cfg.ema_flow.use_ema and outer_iter == cfg.ema_flow.start_iter:
            ema_flow = EMA(flow_model, cfg.ema_flow.decay)

        for innter_iter in range(cfg.inner_iter):

            # Sample the collocation points
            x1 = buffer.sample()[0].to(cfg.device)
            if cfg.conti_prob_path == "VE":
                t = torch.rand(x1.shape[0], device=cfg.device)
            else:
                t = torch.rand(x1.shape[0], device=cfg.device) * 0.97 + 0.03
            x0 = path.sample_x0(x1.shape[0], device=cfg.device)
            path_sample = path.sample(x0, x1, t)
            xt = path_sample.x_t

            # Learn the flow
            est_flow = path.estimate_flow(
                xt,
                t,
                energy.energy,
                num_mc_samples=cfg.flow_mc_samples,
            ).detach()

            if cfg.clip_flow_est is not None:
                est_flow = torch.clamp_min(est_flow, min=-cfg.clip_flow_est)

            flow = flow_model(xt, t)
            flow_loss = flow_loss_fn(est_flow, flow)

            optim_flow.zero_grad()
            loss = flow_loss.mean()
            loss.backward()

            if cfg.clip_grad_norm_flow is not None:
                torch.nn.utils.clip_grad_norm_(
                    flow_model.parameters(), cfg.clip_grad_norm_flow
                )

            optim_flow.step()

            logger.log_loss(
                {
                    "flow_loss": loss.item(),
                    **t_stratified_loss(flow_loss, t, "flow_loss"),
                }
            )

            if cfg.ema_flow.use_ema and outer_iter > cfg.ema_flow.start_iter:
                ema_flow.update(flow_model)
                ema_flow.apply()
                # Now flow_model is the EMA, estimate the vel with EMA target.

            # Learn the velocity
            estimated_vel = path.bootstrap_estimate_vel(
                xt,
                t,
                energy.energy,
                flow_model,
                num_mc_samples=cfg.num_mc_samples,
                step_size=cfg.bootstrap_step_size,
            )

            estimated_vel = clip(estimated_vel, cfg.clip_est, energy)
            vel = drift(xt, t)

            optim.zero_grad()
            vel_loss = vector_mse(vel, estimated_vel)
            loss = vel_loss.mean()
            loss.backward()
            optim.step()

            logger.log_loss(
                {
                    "loss": loss.item(),
                    **t_stratified_loss(vel_loss, t, "loss"),
                }
            )

            # Restore the flow model
            if cfg.ema_flow.use_ema and outer_iter > cfg.ema_flow.start_iter:
                ema_flow.restore()
