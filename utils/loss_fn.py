import torch


def mse_loss(u, v, **kwargs):
    loss = ((u - v) ** 2).sum(dim=-1).mean()
    return loss


def ce_loss(u, v, **kwargs):
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    u = u + epsilon
    u = u / u.sum(dim=-1, keepdim=True)
    loss = -(v * torch.log(u)).sum(dim=-1).mean()

    return loss


def matrix_mse(u, v):
    loss = ((u - v) ** 2).sum(dim=-1).mean(dim=-1)
    return loss


def vector_mse(u, v):
    loss = ((u - v) ** 2).sum(dim=-1)
    return loss


def scalar_mse(u, v):
    loss = (u - v) ** 2
    return loss


def cosh_loss(u, v):
    loss = 2 * torch.cosh(u - v) - 2
    return loss


def lineax1_loss(u, v):
    t = u - v
    loss = torch.exp(t) - t - 1
    return loss


def lineax_half_loss(u, v):
    t = u - v
    loss = 4 * torch.exp(t / 2) - 2 * t - 4
    return loss


def make_loss_fn(name):
    if name == "cosh":
        return cosh_loss
    elif name == "mse":
        return scalar_mse
    elif name == "lineax1":
        return lineax1_loss
    elif name == "lineax_half":
        return lineax_half_loss
    else:
        raise ValueError(f"Unknown flow loss type: {name}")
