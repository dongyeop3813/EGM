"""
Bootstrapped Energy-based discrete flow matching
"""

import torch

from energy import *
from architectures import *
from prob_path import *
from utils import *
from flow_matching.path.scheduler.scheduler import VPScheduler

import tqdm

from hydra.utils import instantiate


def train(cfg, energy: BaseSet, logger: Logger):

    prob_denoiser = instantiate(cfg.model)(
        num_tokens=energy.num_tokens + 1,
        grid_size=cfg.energy.grid_size,
        seq_length=cfg.energy.grid_size * cfg.energy.grid_size,
    ).to(cfg.device)

    flow_model = instantiate(cfg.flow_model)(
        num_tokens=energy.num_tokens + 1,
        grid_size=cfg.energy.grid_size,
        seq_length=cfg.energy.grid_size * cfg.energy.grid_size,
        energy_fn=energy.local_energy,
    ).to(cfg.device)

    logger.add_hyperparams({"num_params": count_params(prob_denoiser)})
    logger.add_hyperparams({"num_params_flow": count_params(flow_model)})

    # Initialize optimizers
    optim = init_optimizer(prob_denoiser, cfg.optim)
    optim_flow = init_optimizer(flow_model, cfg.optim_flow)

    if cfg.lr_scheduler is not None:
        # Scheduler for prob_denoiser model
        optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=cfg.iter,
            eta_min=cfg.lr_scheduler.eta_min,
        )

    if cfg.scheduler is not None and cfg.scheduler.type == "geometric":
        scheduler = GeometricScheduler(
            exponent=cfg.scheduler.exponent,
        )
    else:
        scheduler = PolynomialConvexScheduler(n=1.0)

    mixture_path, solver = construct_masked_diffusion(
        prob_denoiser,
        energy.num_tokens,
        energy.ndim,
        scheduler=scheduler,
    )

    buffer = make_buffer(cfg, energy)

    lambda_bootstrap = cfg.lambda_bootstrap
    lambda_simple = cfg.lambda_simple
    lambda_flow = cfg.lambda_flow
    flow_loss_fn = make_loss_fn(cfg.flow_loss_type)

    for outer_iter in tqdm.trange(cfg.iter):
        x1 = eval_step(
            outer_iter, cfg, prob_denoiser, mixture_path, solver, energy, logger
        )

        if not cfg.use_gt_sample:
            buffer.add(x1, energy.log_reward(x1))
            eval_buffer(buffer, energy, logger)

        if cfg.ema_flow.use_ema and outer_iter == cfg.ema_flow.start_iter:
            # Start EMA for the flow model
            ema_flow = EMA(flow_model, cfg.ema_flow.decay)

        if cfg.lambda_schedule:
            lambda_bootstrap = outer_iter / cfg.iter
            lambda_simple = 1 - outer_iter / cfg.iter

        for inner_iter in range(cfg.inner_iter):

            # Sample the collocation points
            x1 = buffer.sample()[0].to(cfg.device)
            t = torch.rand(x1.shape[0], device=cfg.device)
            x0 = mixture_path.sample_x0(x1.shape[0], device=cfg.device)
            path_sample = mixture_path.sample(x0, x1, t)
            xt = path_sample.x_t

            est_flow = mixture_path.estimate(
                xt,
                t,
                energy.energy,
                num_mc_samples=cfg.num_mc_samples,
                proposal_type=cfg.flow_proposal_type,
                estimate_type=cfg.flow_estimate_type,
                flow_model=flow_model,
                step_size=cfg.bootstrap_step_size,
            ).detach()

            flow = flow_model(xt, t)
            flow_loss = flow_loss_fn(est_flow, flow)

            if cfg.clip_grad_norm_flow is not None:
                torch.nn.utils.clip_grad_norm_(
                    flow_model.parameters(), cfg.clip_grad_norm_flow
                )

            optim_flow.zero_grad()
            loss = lambda_flow * flow_loss.mean()
            loss.backward()
            optim_flow.step()

            # Estimate the flow
            if cfg.ema_flow.use_ema and outer_iter > cfg.ema_flow.start_iter:
                ema_flow.update(flow_model)
                ema_flow.apply()
                # Now flow_model is the EMA, estimate the vel with EMA target.

            # Estimate the prob denoiser
            est_vel = mixture_path.estimate(
                xt,
                t,
                energy.energy,
                num_mc_samples=cfg.num_mc_samples,
                proposal_type="naive",
                estimate_type=cfg.estimate_type,
            ).detach()

            bootstrap_vel = mixture_path.estimate(
                xt,
                t,
                energy.energy,
                num_mc_samples=cfg.num_mc_samples,
                proposal_type=cfg.bootstrap_proposal_type,
                estimate_type=cfg.bootstrap_estimate_type,
                flow_model=flow_model,
                step_size=cfg.bootstrap_step_size,
                learned_prob_denoiser=prob_denoiser,
            ).detach()

            if cfg.estimate_type == "vel":
                vel = prob_denoiser.prob_vel(xt, t)
            elif cfg.estimate_type == "denoiser":
                vel = prob_denoiser.prob_x1_given_xt(xt, t)
            else:
                raise ValueError(f"Invalid estimate type: {cfg.estimate_type}")

            optim.zero_grad()
            simple_loss = matrix_mse(vel, est_vel)
            bootstrap_loss = matrix_mse(vel, bootstrap_vel)

            if "loss_scaling" in cfg and cfg.loss_scaling:
                bootstrap_loss = cfg.bootstrap_step_size * (t + 1e-6) * bootstrap_loss

            loss = (
                lambda_bootstrap * bootstrap_loss.mean()
                + lambda_simple * simple_loss.mean()
            )
            loss.backward()

            if cfg.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    prob_denoiser.parameters(), cfg.clip_grad_norm
                )

            optim.step()

            logger.log_loss(
                {
                    "flow_loss": flow_loss.mean().item(),
                    "vel_loss": loss.item(),
                    "simple_loss": simple_loss.mean().item(),
                    "bootstrap_loss": bootstrap_loss.mean().item(),
                    **t_stratified_loss(flow_loss, t, "flow_loss"),
                    **t_stratified_loss(simple_loss, t, "simple_loss"),
                    **t_stratified_loss(bootstrap_loss, t, "bootstrap_loss"),
                }
            )

            if inner_iter == 0:
                _, weight = mixture_path.small_step_proposal(
                    xt, t, None, flow_model, 1000, 0.01
                )
                ess = 1 / torch.sum(weight**2, dim=-1)
                logger.log_metric({"ESS": ess.mean().item()})

            # Restore the flow model
            if cfg.ema_flow.use_ema and outer_iter > cfg.ema_flow.start_iter:
                ema_flow.restore()

        if cfg.lr_scheduler is not None:
            optim_scheduler.step()
            logger.log_metric({"lr": optim_scheduler.get_last_lr()[0]})
