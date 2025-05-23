import torch


def init_optimizer(models, optim_cfg):
    if not isinstance(models, list):
        models = [models]

    if optim_cfg.name == "adam":
        return torch.optim.Adam(
            [p for model in models for p in model.parameters()],
            lr=optim_cfg.lr,
            betas=(optim_cfg.beta1, optim_cfg.beta2),
            weight_decay=optim_cfg.weight_decay,
        )
    elif optim_cfg.name == "adamw":
        return torch.optim.AdamW(
            [p for model in models for p in model.parameters()],
            lr=optim_cfg.lr,
            weight_decay=optim_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Invalid optimizer: {optim_cfg.name}")
