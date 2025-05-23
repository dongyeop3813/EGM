import torch
import random
import numpy as np
from typing import Union
from flow_matching.path.scheduler.scheduler import ConvexScheduler, SchedulerOutput


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def extract_model_name(class_path):
    return class_path.split(".")[-1]


def save_fig(output_dir, fig, name):
    fig.savefig(
        f"{output_dir}/{name}.png",
        format="png",
        bbox_inches="tight",
        dpi=100,
    )


def save_model(output_dir, model, step, name=""):
    torch.save(model.state_dict(), f"{output_dir}/model{name}_it{step}.pt")


def next_time_step(t, step_size):
    return torch.min(t + step_size, torch.ones_like(t))


def clip(x, clip_cfg, energy):
    if clip_cfg.type == "norm":
        return clip_norm(x, clip_cfg.value)
    elif clip_cfg.type == "vel":
        return clip_vel(x, clip_cfg.value, energy.n_particles, energy.spatial_dim)
    else:
        raise ValueError(f"Unknown clip type: {clip_cfg.type}")


def clip_norm(x, max_norm):
    norm = torch.norm(x, dim=-1, keepdim=True)
    mask = norm > max_norm
    scale_factor = torch.ones_like(norm)
    scale_factor[mask] = max_norm / norm[mask]
    return x * scale_factor


def clip_vel(x, max_vel, num_particles, spatial_dim):
    x = x.view(-1, num_particles, spatial_dim)
    norm = torch.norm(x, dim=-1, keepdim=True)
    mask = norm > max_vel
    scale_factor = torch.ones_like(norm)
    scale_factor[mask] = max_vel / norm[mask]
    return (x * scale_factor).view(-1, num_particles * spatial_dim)


def t_stratified_loss(loss, t, prefix=""):
    intervals = torch.linspace(0, 1.0, 6)
    stratified_losses = {}

    for i in range(len(intervals) - 1):
        start, end = intervals[i], intervals[i + 1]
        mask = (t >= start) & (t < end)
        if i == len(intervals) - 2:
            mask = (t >= start) & (t <= end)

        if mask.sum() > 0:
            interval_loss = loss[mask].mean().item()
        else:
            interval_loss = 0.0

        stratified_losses[f"{prefix}_{start:.1f}<=t<{end:.1f}"] = interval_loss

    return stratified_losses


def log_mean_exp(x, dim=1):
    return torch.logsumexp(x, dim=dim) - np.log(x.shape[dim])


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class EMA(torch.nn.Module):
    def __init__(self, model, decay):
        super().__init__()
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.stored_model = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                self.stored_model[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.shadow[name] * self.decay + param.data * (
                    1 - self.decay
                )

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.stored_model[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.stored_model[name]


class GeometricScheduler(ConvexScheduler):
    def __init__(self, exponent: float = 1e-3) -> None:
        self.exponent = exponent

    def __call__(self, t: torch.Tensor) -> SchedulerOutput:
        alpha_t = 1 - (self.exponent**t) / (1 - self.exponent)
        d_alpha_t = -np.log(self.exponent) * (self.exponent**t) / (1 - self.exponent)

        return SchedulerOutput(
            alpha_t=alpha_t,
            sigma_t=1 - alpha_t,
            d_alpha_t=d_alpha_t,
            d_sigma_t=-d_alpha_t,
        )

    def kappa_inverse(self, kappa: torch.Tensor) -> torch.Tensor:
        return torch.log(1 - kappa * (1 - self.exponent)) / np.log(self.exponent)


def effective_sample_size(w):
    normalized_w = w / w.sum(dim=-1, keepdim=True)
    return 1 / torch.sum(normalized_w**2, dim=-1)
