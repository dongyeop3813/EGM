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

    flow_model = instantiate(cfg.flow_model)(
        num_tokens=energy.num_disc_tokens + 1,
        discrete_dim=energy.n_disc_dim,
        continuous_dim=energy.n_conti_dim,
    ).to(cfg.device)

    logger.add_hyperparams({"num_params": count_params(drift)})
    logger.add_hyperparams({"num_params_flow": count_params(flow_model)})

    optim = init_optimizer(drift, cfg.optim)
    optim_flow = init_optimizer(flow_model, cfg.optim_flow)

    if cfg.lr_scheduler is not None:
        # Scheduler for prob_denoiser model
        optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=cfg.iter,
            eta_min=cfg.lr_scheduler.eta_min,
        )

    path, solver = construct_joint_path(cfg, drift, energy)

    buffer = make_buffer(cfg, energy)

    flow_loss_fn = make_loss_fn(cfg.flow_loss_type)

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

        if cfg.ema_flow.use_ema and outer_iter == cfg.ema_flow.start_iter:
            ema_flow = EMA(flow_model, cfg.ema_flow.decay)

        for innter_iter in range(cfg.inner_iter):

            # Sample the collocation points
            x1 = buffer.sample()[0].to(cfg.device)
            if cfg.conti_prob_path == "VE":
                t = torch.rand(x1.shape[0], device=cfg.device)
            else:
                t = (
                    torch.rand(x1.shape[0], device=cfg.device)
                    * (1 - cfg.solver_start_time)
                    + cfg.solver_start_time
                )
            x0 = path.sample_x0(x1.shape[0], device=cfg.device)
            path_sample = path.sample(x0, x1, t)
            xt = path_sample.x_t

            # Learn the flow
            est_flow = path.estimate_flow(
                xt,
                t,
                energy.energy,
                num_mc_samples=cfg.num_mc_samples,
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
            estimated_conti_vel, est_disc_denoiser = path.bootstrap_estimate_vel(
                xt,
                t,
                energy.energy,
                flow_model,
                num_mc_samples=cfg.num_mc_samples,
                step_size=cfg.bootstrap_step_size,
            )

            estimated_conti_vel = clip(estimated_conti_vel, cfg.clip_est, energy)

            model_conti_vel, model_disc_denoiser = drift(xt, t)

            optim.zero_grad()
            conti_loss = vector_mse(model_conti_vel, estimated_conti_vel)
            disc_loss = matrix_mse(model_disc_denoiser, est_disc_denoiser)

            if "loss_scaling" in cfg and cfg.loss_scaling:
                conti_loss = cfg.bootstrap_step_size * conti_loss * (t + 1e-6)
                disc_loss = cfg.bootstrap_step_size * disc_loss * (t + 1e-6)

            loss = lambda_conti * conti_loss.mean() + lambda_disc * disc_loss.mean()
            loss.backward()
            optim.step()

            logger.log_loss(
                {
                    "loss": loss.item(),
                    "conti_loss": conti_loss.mean().item(),
                    "disc_loss": disc_loss.mean().item(),
                    **t_stratified_loss(conti_loss, t, "conti_loss"),
                    **t_stratified_loss(disc_loss, t, "disc_loss"),
                }
            )

            # Restore the flow model
            if cfg.ema_flow.use_ema and outer_iter > cfg.ema_flow.start_iter:
                ema_flow.restore()

        if cfg.lr_scheduler is not None:
            optim_scheduler.step()
            logger.log_metric({"lr": optim_scheduler.get_last_lr()[0]})
