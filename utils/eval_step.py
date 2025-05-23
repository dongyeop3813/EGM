import torch
from .etc import save_model


def eval_step(iter, cfg, drift, path, solver, energy, logger, **kwargs):
    drift.eval()

    with torch.no_grad():
        x0 = path.sample_x0(cfg.num_sample_to_collect, cfg.device)
        x1 = solver.sample(x0, 1 / cfg.num_steps, **kwargs)

        sample_fig = energy.visualize(x1, show_ground_truth=True)
        logger.log_visual(sample_fig)
        logger.log_metric(energy.metric(x1))

    if (iter + 1) % 10 == 0:
        save_model(logger.output_dir, drift, iter)

    drift.train()

    return x1


def eval_buffer(buffer, energy, logger):
    x1, _ = buffer.sample()

    buffer_fig = energy.visualize(x1, show_ground_truth=True)
    logger.log_visual(buffer_fig, prefix="buffer")
    return x1
