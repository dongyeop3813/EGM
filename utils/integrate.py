import torch
import torch.nn.functional as F
from typing import Union


def euler_step_disc(
    prob_vel,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: Union[float, torch.Tensor],
    num_tokens: int,
    seq_length: int,
) -> torch.Tensor:
    """
    Input shapes:
        x: [B, L] with values in {0, 1, ..., num_tokens - 1 (Mask token)}
        t: [B]
        dt: float or [B]

    Output shape:
        x: [B, L] with values in {0, 1, ..., num_tokens - 1 (Mask token)}
    """
    # rates: [B, L, num_tokens]
    rates = prob_vel(x, t)

    # One-hot encoding of current state
    current_state = F.one_hot(x, num_classes=num_tokens)

    # Calculate transition probabilities: P(x_t+dt | x_t) = x_t + rates * dt
    # Since rates already satisfy the rate matrix conditions,
    # adding rates * dt to current_state gives correct transition probabilities
    transition_probs = current_state + rates * dt

    # Ensure transition probabilities are non-negative
    transition_probs = F.relu(transition_probs)

    # Normalize transition probabilities to sum to 1 for each position
    transition_probs = transition_probs / transition_probs.sum(dim=-1, keepdim=True)

    # Sample new states for each position
    x = torch.multinomial(transition_probs.view(-1, num_tokens), num_samples=1)
    x = x.view(x.shape[0] // seq_length, seq_length)

    return x


def simulate_disc(prob_vel, x0: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """
    Simulate the CTMC model for a given number of steps.
    """

    num_samples, seq_length = x0.shape
    device = x0.device

    dt = 1.00 / 100
    xt = x0
    for i in range(100):
        t = torch.ones((num_samples,), dtype=torch.float, device=device) * (i * dt)
        xt = euler_step_disc(prob_vel, xt, t, dt, num_tokens, seq_length)
    return xt


def euler_step_conti(
    drift,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: Union[float, torch.Tensor],
) -> torch.Tensor:
    return x + drift(x, t) * dt


def simulate_conti(drift, x0: torch.Tensor) -> torch.Tensor:
    """
    Simulate the CTMC model for a given number of steps.
    """

    # Fully masked initial state
    xt = x0

    num_steps = 100
    dt = 1.00 / num_steps
    for i in range(1, num_steps):
        t = torch.ones((x0.shape[0],), dtype=torch.float, device=x0.device) * (i * dt)
        xt = euler_step_conti(drift, xt, t, dt)
    return xt
