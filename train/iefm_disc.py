import torch
from omegaconf import OmegaConf

from energy import *
from architectures import *
from prob_path import *
from utils import *
import tqdm

from hydra.utils import instantiate


def train(cfg, energy: BaseSet, logger: Logger):

    prob_denoiser = instantiate(cfg.model)(
        num_tokens=energy.num_tokens + 1,
        grid_size=cfg.energy.grid_size,
        seq_length=cfg.energy.grid_size * cfg.energy.grid_size,
    ).to(cfg.device)

    logger.add_hyperparams({"num_params": count_params(prob_denoiser)})

    optim = init_optimizer(prob_denoiser, cfg.optim)

    mixture_path, solver = construct_masked_diffusion(
        prob_denoiser, energy.num_tokens, energy.ndim
    )

    buffer = make_buffer(cfg, energy)

    loss_fn = make_denoiser_loss_fn(cfg.loss_type)

    for outer_iter in tqdm.trange(cfg.iter):
        x1 = eval_step(
            outer_iter,
            cfg,
            prob_denoiser,
            mixture_path,
            solver,
            energy,
            logger,
        )

        if not cfg.use_gt_sample:
            buffer.add(x1, energy.log_reward(x1))
            eval_buffer(buffer, energy, logger)

        for inner_iter in range(cfg.inner_iter):

            # Sample the collocation points
            x1 = buffer.sample()[0].to(cfg.device)
            t = torch.rand(x1.shape[0], device=cfg.device)
            x0 = mixture_path.sample_x0(x1.shape[0], device=cfg.device)
            path_sample = mixture_path.sample(x0, x1, t)
            xt = path_sample.x_t

            # Estimate the prob velocity
            v = mixture_path.estimate(
                xt,
                t,
                energy.energy,
                num_mc_samples=cfg.num_mc_samples,
                proposal_type="naive",
                estimate_type="denoiser",
            )
            u = prob_denoiser.prob_x1_given_xt(xt, t)

            optim.zero_grad()
            loss = loss_fn(u, v, xt=xt, t=t)
            loss.backward()
            optim.step()

            logger.log_loss({"loss": loss.item()})

            if inner_iter == 0:
                x1 = mixture_path.naive_proposal(xt, t, None, 1000)
                weight = torch.exp(-energy.energy(x1))
                ess = effective_sample_size(weight)
                logger.log_metric({"ESS": ess.mean().item()})


def make_denoiser_loss_fn(name):
    if name == "ce":
        return ce_loss
    elif name == "mse":
        return mse_loss
    else:
        raise ValueError(f"Unknown denoiser loss type: {name}")
