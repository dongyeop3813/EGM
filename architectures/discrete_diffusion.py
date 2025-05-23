import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Callable

import math

from .emb import SinusoidalPositionEmbeddings


class FactorizedVelocity(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        seq_length: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length  # sequence length =: L

        self.token_embedding = nn.Embedding(num_tokens, hidden_dim)

        # Timestep embedding with sinusoidal encoding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Neural drift module (time-dependent)
        # Input: [L, H] x [H] -> [L, num_tokens]
        self.vel_module = nn.Sequential(
            nn.Linear(seq_length * hidden_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)],
            *[nn.GELU() for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, seq_length * num_tokens),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, *, L] shape of discrete state tensor
            t: [B, *] shape of time tensor

        Returns:
            [B, *, L, num_tokens] shape of velocity tensor
        """

        *batch_dim, L = x.shape

        x_emb = self.token_embedding(x).view(*batch_dim, L * self.hidden_dim)
        t_emb = self.time_embedding(t)
        h = torch.cat([x_emb, t_emb], dim=-1)

        # [B, *, L, num_tokens]
        logits = self.vel_module(h)
        logits = logits.view(*batch_dim, L, self.num_tokens)

        rates = F.softplus(logits)

        # is_mask [B, *, L]
        mask_token_idx = self.num_tokens - 1
        is_mask = x == mask_token_idx
        rates[..., :, mask_token_idx] = (-1 / (1 - t)).unsqueeze(-1) * is_mask

        rates[(~is_mask).unsqueeze(-1).expand_as(rates)] = 0.0

        # Make the diagonal rates as minus the sum of the rates.
        diag_mask = F.one_hot(x, num_classes=self.num_tokens).bool()
        rates[diag_mask] = 0.0

        rates = torch.where(
            diag_mask,
            -rates.sum(dim=-1).unsqueeze(-1).expand_as(rates),
            rates,
        )

        return rates

    def prob_vel(self, xt, t):
        return self(xt, t)

    def prob_x1_given_xt(self, xt, t):
        rates = self(xt, t)

        xt = F.one_hot(xt, num_classes=self.num_tokens)

        probs = (1 - t[..., None, None]) * rates + xt

        return probs


class FactorizedProbDenoiser(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        seq_length: int,
        tok_emb_dim: int = 4,
        hidden_dim: int = 32,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length  # sequence length =: L
        self.tok_emb_dim = tok_emb_dim

        self.token_embedding = nn.Embedding(num_tokens, tok_emb_dim)

        # Timestep embedding with sinusoidal encoding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Neural drift module (time-dependent)
        # Input: [L, H] x [H] -> [L, num_tokens]
        self.denoiser_module = nn.Sequential(
            nn.Linear(seq_length * tok_emb_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)],
            *[nn.GELU() for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, seq_length * num_tokens),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, *, L] shape of discrete state tensor
            t: [B, *] shape of time tensor

        Returns:
            [B, *, L, num_tokens] shape of velocity tensor
        """

        *batch_dim, L = x.shape

        x_emb = self.token_embedding(x).view(*batch_dim, L * self.tok_emb_dim)
        t_emb = self.time_embedding(t)
        h = torch.cat([x_emb, t_emb], dim=-1)

        logits = self.denoiser_module(h)
        logits = logits.view(*batch_dim, L, self.num_tokens)

        mask_token_idx = self.num_tokens - 1
        logits[..., mask_token_idx] = float("-inf")

        return logits

    def prob_x1_given_xt(self, xt, t):
        logits = self(xt, t)
        probs = F.softmax(logits, dim=-1)
        return probs


_EPS = 1e-8


class NoiseScaledVelocity(nn.Module):
    def __init__(self, model: FactorizedVelocity, noise_schedule):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule

        self.num_tokens = model.num_tokens
        self.hidden_dim = model.hidden_dim
        self.seq_length = model.seq_length

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        coeff = self.noise_schedule.diff(t) / (1 - self.noise_schedule(t) + _EPS)
        return coeff[:, None, None] * self.model(x, t)

    def prob_vel(self, xt, t):
        return self(xt, t)


class CNNProbDenoiser(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        grid_size: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        use_batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size

        # Token embedding: [B, H, W] -> [B, hidden_dim, H, W]
        self.token_embedding = nn.Embedding(num_tokens, hidden_dim)

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # CNN layers
        layers = []
        in_channels = 2 * hidden_dim  # additional hidden_dim for time embedding

        # First layer
        layers.extend(
            [
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.GELU(),
            ]
        )

        # Middle layers
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim) if use_batch_norm else nn.Identity(),
                    nn.GELU(),
                ]
            )

        # Final layer
        layers.append(nn.Conv2d(hidden_dim, num_tokens, kernel_size=3, padding=1))

        self.cnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, *, L] shape of discrete state tensor
            t: [B, *] shape of time tensor

        Returns:
            [B, *, L, num_tokens] shape of velocity tensor
        """

        *batch_dim, L = x.shape

        # Reshape input to 2D grid: [B, H*W] -> [B, H, W]
        x = x.view(*batch_dim, self.grid_size, self.grid_size)

        # Token embedding: [B, *, H, W] -> [B, *, hidden_dim, H, W]
        x_emb = self.token_embedding(x).permute(*batch_dim, -1, -3, -2)

        # Time embedding: [B, *] -> [B, *, hidden_dim, 1, 1]
        t_emb = self.time_embedding(t).view(*batch_dim, self.hidden_dim, 1, 1)
        t_emb = t_emb.expand(*batch_dim, -1, self.grid_size, self.grid_size)

        # Concatenate embeddings: [B, *, hidden_dim+1, H, W]
        h = torch.cat([x_emb, t_emb], dim=-3)

        # CNN forward pass: [B, *, num_tokens, H, W]
        logits = self.cnn(h)

        # Reshape back to original format: [B, *, num_tokens, H, W]
        logits = logits.permute(*batch_dim, -1, -3, -2)

        mask_token_idx = self.num_tokens - 1
        logits[..., mask_token_idx] = float("-inf")

        return logits

    def prob_x1_given_xt(self, xt, t):
        logits = self(xt, t)
        probs = F.softmax(logits, dim=-1)
        return probs


class TransformerFactorizedProbDenoiser(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        seq_length: int,
        tok_emb_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.grid_size = int(math.sqrt(seq_length))
        self.num_layers = num_layers

        # Token embedding with scale factor
        self.token_embedding = nn.Embedding(num_tokens, tok_emb_dim)
        self.embed_projection = nn.Linear(tok_emb_dim, hidden_dim)
        self.embed_scale = math.sqrt(hidden_dim)

        # Initialize sinusoidal positional encoding
        self._init_positional_encoding()

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(
                            hidden_dim, num_heads, dropout=dropout, batch_first=True
                        ),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                        ),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_tokens)

        # Initialize weights
        self._init_weights()

    def _init_positional_encoding(self):
        # Create positional embedding for each grid position
        self.pos_embedding = nn.Embedding(
            self.grid_size * self.grid_size, self.hidden_dim
        )

        # Initialize positional indices
        position_indices = torch.arange(self.grid_size * self.grid_size)
        self.register_buffer("position_indices", position_indices)

        # Initialize weights
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        # Scale parameter for positional embeddings
        self.pos_scale = nn.Parameter(torch.ones(1))

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize output projection with smaller weights
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Discrete Ising grid state and time as input, return single GFlowNet flow value.

        Args:
            x: [B, *, L] shape of discrete state tensor
            t: [B, *] shape of time tensor

        Returns:
            [B, *] shape of flow value tensor
        """

        *batch_dim, L = x.shape

        # Token embedding with scale, [B, *, L] -> [B, *, L, H]
        x_emb = self.token_embedding(x) * self.embed_scale
        x_emb = self.embed_projection(x_emb)

        # Add positional embeddings
        pos_emb = self.pos_embedding(self.position_indices).unsqueeze(0)  # [1, L, H]
        h = x_emb + self.pos_scale * pos_emb

        # Time embedding, [B, *] -> [B, *, L, H]
        t_emb = self.time_embedding(t)
        t_emb = t_emb.unsqueeze(-2).expand(*batch_dim, L, -1)
        h = h + t_emb

        # Transformer layers
        for layer in self.layers:
            # Self-attention with residual connection
            h_norm = layer["norm1"](h)
            attn_output, _ = layer["self_attn"](h_norm, h_norm, h_norm)
            h = h + attn_output

            # Feed-forward with residual connection
            h_norm = layer["norm2"](h)
            ffn_output = layer["ffn"](h_norm)
            h = h + ffn_output

        # Output projection
        logits = self.output_proj(h)  # [B, *, L, num_tokens]

        # Mask token handling
        mask_token_idx = self.num_tokens - 1
        logits[..., mask_token_idx] = float("-inf")

        return logits

    def prob_x1_given_xt(self, xt, t):
        logits = self(xt, t)
        probs = F.softmax(logits, dim=-1)
        return probs


class DiscreteStateFlow(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        seq_length: int,
        tok_emb_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.tok_emb_dim = tok_emb_dim

        self.token_embedding = nn.Embedding(num_tokens, tok_emb_dim)

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = [
            nn.Linear(seq_length * tok_emb_dim + hidden_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.flow_network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Discrete Ising grid state and time as input, return single GFlowNet flow value.

        Args:
            x: [B, *, L] shape of discrete state tensor
            t: [B, *] shape of time tensor

        Returns:
            [B, *] shape of flow value tensor
        """

        *batch_dim, L = x.shape

        x_emb = self.token_embedding(x).view(*batch_dim, L * self.tok_emb_dim)
        t_emb = self.time_embedding(t)
        combined = torch.cat([x_emb, t_emb], dim=-1)

        flow_value = self.flow_network(combined).squeeze(-1)

        return flow_value


class DiscreteStateFlowSymmetric(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        seq_length: int,
        tok_emb_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.tok_emb_dim = tok_emb_dim

        self.token_embedding = nn.Embedding(num_tokens, tok_emb_dim)

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = [
            nn.Linear(seq_length * tok_emb_dim + hidden_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.flow_network = nn.Sequential(*layers)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the grid 90, 180, 270 degrees.

        Args:
            x: [B.., L]

        Returns:
            x: [4, B.., L]
        """
        *batch_dim, L = x.shape

        grid_size = int(math.sqrt(L))
        x_grid = x.view(*batch_dim, grid_size, grid_size)

        x_rot90 = torch.rot90(x_grid, k=1, dims=(-2, -1))
        x_rot180 = torch.rot90(x_grid, k=2, dims=(-2, -1))
        x_rot270 = torch.rot90(x_grid, k=3, dims=(-2, -1))

        x_rot90 = x_rot90.reshape(*batch_dim, L)
        x_rot180 = x_rot180.reshape(*batch_dim, L)
        x_rot270 = x_rot270.reshape(*batch_dim, L)

        x = torch.stack([x, x_rot90, x_rot180, x_rot270], dim=0)

        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Discrete Ising grid state and time as input, return single GFlowNet flow value.

        Args:
            x: [B, *, L] shape of discrete state tensor
            t: [B, *] shape of time tensor

        Returns:
            [B, *] shape of flow value tensor
        """

        *batch_dim, L = x.shape

        x = self.rotate(x)

        x_emb = self.token_embedding(x).view(4, *batch_dim, L * self.tok_emb_dim)
        t_emb = self.time_embedding(t).unsqueeze(0).expand(4, *batch_dim, -1)
        combined = torch.cat([x_emb, t_emb], dim=-1)

        flow_value = self.flow_network(combined).squeeze(-1)

        return flow_value.sum(dim=0)


class FLDiscreteStateFlowIsing(nn.Module):
    """
    If some tokens are known, the flow considering already determined local
    interaction of Ising model will be benefical.
    This class is a flow network that considers local interaction of Ising model.

    Args:
        num_tokens: number of tokens in the grid
        seq_length: length of the grid
        tok_emb_dim: dimension of the token embedding
        hidden_dim: dimension of the hidden state
        num_layers: number of layers in the flow network
    """

    def __init__(
        self,
        num_tokens: int,
        seq_length: int,
        tok_emb_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        energy_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.tok_emb_dim = tok_emb_dim

        self.token_embedding = nn.Embedding(num_tokens, tok_emb_dim)

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = [
            nn.Linear(seq_length * tok_emb_dim + hidden_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.flow_network = nn.Sequential(*layers)
        self.energy_fn = energy_fn

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Discrete Ising grid state and time as input, return single GFlowNet flow value.

        Args:
            x: [B, *, L] shape of discrete state tensor
            t: [B, *] shape of time tensor

        Returns:
            [B, *] shape of flow value tensor
        """

        *batch_dim, L = x.shape

        intermediate_energy = self.energy_fn(x)

        x_emb = self.token_embedding(x).view(*batch_dim, L * self.tok_emb_dim)
        t_emb = self.time_embedding(t)
        combined = torch.cat([x_emb, t_emb], dim=-1)

        flow_value = self.flow_network(combined).squeeze(-1)

        return -intermediate_energy + flow_value


class FLDiscreteStateFlowIsingResidual(nn.Module):
    """
    If some tokens are known, the flow considering already determined local
    interaction of Ising model will be benefical.
    This class is a flow network that considers local interaction of Ising model.

    Args:
        num_tokens: number of tokens in the grid
        seq_length: length of the grid
        tok_emb_dim: dimension of the token embedding
        hidden_dim: dimension of the hidden state
        num_layers: number of layers in the flow network
    """

    def __init__(
        self,
        num_tokens: int,
        seq_length: int,
        tok_emb_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        energy_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.tok_emb_dim = tok_emb_dim

        self.token_embedding = nn.Embedding(num_tokens, tok_emb_dim)

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_layer = nn.Sequential(
            nn.Linear(seq_length * tok_emb_dim + hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.residual_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.residual_layers.append(
                nn.ModuleList(
                    [
                        nn.Linear(hidden_dim + hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                    ]
                )
            )

        self.output_layer = nn.Linear(hidden_dim, 1)
        self.energy_fn = energy_fn

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Discrete Ising grid state and time as input, return single GFlowNet flow value.

        Args:
            x: [B, *, L] shape of discrete state tensor
            t: [B, *] shape of time tensor

        Returns:
            [B, *] shape of flow value tensor
        """

        *batch_dim, L = x.shape

        intermediate_energy = self.energy_fn(x)

        x_emb = self.token_embedding(x).view(*batch_dim, L * self.tok_emb_dim)
        t_emb = self.time_embedding(t)
        combined = torch.cat([x_emb, t_emb], dim=-1)

        h = self.input_layer(combined)

        for layer_modules in self.residual_layers:
            h_combined = torch.cat([h, t_emb], dim=-1)
            h_res = layer_modules[0](h_combined)
            h_res = layer_modules[1](h_res)
            h_res = layer_modules[2](h_res)
            h_res = layer_modules[3](h_res)
            h = h + h_res

        flow_value = self.output_layer(h).squeeze(-1)

        return -intermediate_energy + flow_value


class CNNStateFlow(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        seq_length: int,
        tok_emb_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        grid_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.tok_emb_dim = tok_emb_dim
        self.grid_size = grid_size or int(math.sqrt(seq_length))

        self.token_embedding = nn.Embedding(num_tokens, tok_emb_dim)

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # CNN layers for translational invariance
        self.conv_layers = nn.ModuleList()
        # feature channels after concatenating token and time embeddings
        self.emb_dim = tok_emb_dim + hidden_dim
        in_channels = self.emb_dim
        for _ in range(num_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=0),
                    nn.GELU(),
                    nn.BatchNorm2d(hidden_dim),
                )
            )
            in_channels = hidden_dim

        # Global average pooling for transformation invariance
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Flow value prediction with time information
        self.flow_head = nn.Sequential(
            # take concatenated [features, time_embedding] input of size hidden_dim*2
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the grid by 90, 180, 270 degrees.

        Args:
            x: [B.., L]

        Returns:
            x: [4, B.., L]
        """
        *batch_dim, L = x.shape

        grid_size = int(math.sqrt(L))
        x_grid = x.view(*batch_dim, grid_size, grid_size)

        x_rot90 = torch.rot90(x_grid, k=1, dims=(-2, -1))
        x_rot180 = torch.rot90(x_grid, k=2, dims=(-2, -1))
        x_rot270 = torch.rot90(x_grid, k=3, dims=(-2, -1))

        x_rot90 = x_rot90.reshape(*batch_dim, L)
        x_rot180 = x_rot180.reshape(*batch_dim, L)
        x_rot270 = x_rot270.reshape(*batch_dim, L)

        x = torch.stack([x, x_rot90, x_rot180, x_rot270], dim=0)

        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Takes discrete Ising grid state and time as input, returns single GFlowNet flow value.

        Args:
            x: [B, *, L] shape of discrete state tensor
            t: [B, *] shape of time tensor

        Returns:
            [B, *] shape of flow value tensor
        """

        # Create 4 rotated versions for rotational invariance
        x_rotated = self.rotate(x)  # [4, B, *, L]
        rot_dim, *batch_dims, L = x_rotated.shape

        # Token embedding and reshape to grid form
        x_emb = self.token_embedding(x_rotated)  # [4, B, *, L, tok_emb_dim]

        # Prepare time embedding for each grid cell
        t_emb = self.time_embedding(t)  # [B, *, hidden_dim]
        t_emb_expanded = t_emb.unsqueeze(0).unsqueeze(-2)  # [1, B, *, 1, hidden_dim]
        t_emb_expanded = t_emb_expanded.expand(
            rot_dim, *batch_dims, L, -1
        )  # [4, B, *, L, hidden_dim]

        # Concatenate token and time embeddings: [4, B, *, L, tok_emb_dim+hidden_dim]
        emb = torch.cat([x_emb, t_emb_expanded], dim=-1)
        emb_dim = self.tok_emb_dim + self.hidden_dim

        # Reshape to 2D grid: [4, B, *, H, W, emb_dim]
        emb = emb.view(rot_dim, *batch_dims, self.grid_size, self.grid_size, emb_dim)

        # Move channel dimension for CNN: [4, B, *, emb_dim, H, W]
        # Permute only the last 3 dimensions while keeping all batch dimensions intact
        perm = list(range(len(emb.shape) - 3)) + [-1, -3, -2]
        emb = emb.permute(*perm)

        # Prepare for CNN: combine rotation and batch dims
        features = emb.reshape(-1, emb_dim, self.grid_size, self.grid_size)

        for conv_layer in self.conv_layers:
            # Apply circular padding for periodic boundary conditions
            padded = F.pad(features, (1, 1, 1, 1), mode="circular")
            features = conv_layer(padded)

        features = features.view(
            rot_dim, *batch_dims, self.hidden_dim, self.grid_size, self.grid_size
        )

        # Global average pooling for transformation invariance
        features = (
            self.global_pool(features).squeeze(-1).squeeze(-1)
        )  # [4, B, *, hidden_dim]

        # Average the features from rotated versions
        features = features.mean(dim=0)  # [B, *, hidden_dim]

        # Combine with time embedding
        combined = torch.cat([features, t_emb], dim=-1)

        # Final flow value prediction
        flow_value = self.flow_head(combined).squeeze(-1)

        return flow_value


class TransformerStateFlow(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        seq_length: int,
        tok_emb_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        grid_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.tok_emb_dim = tok_emb_dim
        self.grid_size = grid_size or int(math.sqrt(seq_length))

        # Token embedding
        self.token_embedding = nn.Embedding(num_tokens, tok_emb_dim)
        self.embed_projection = nn.Linear(tok_emb_dim, hidden_dim)
        self.embed_scale = math.sqrt(hidden_dim)

        # Initialize positional encoding
        self._init_positional_encoding()

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(
                            hidden_dim, num_heads, dropout=dropout, batch_first=True
                        ),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                        ),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # Flow prediction head
        self.flow_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_positional_encoding(self):
        # Create positional embedding for each sequence position
        self.pos_embedding = nn.Embedding(self.seq_length, self.hidden_dim)

        # Initialize positional indices
        position_indices = torch.arange(self.seq_length)
        self.register_buffer("position_indices", position_indices)

        # Initialize weights
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        # Scale parameter for positional embeddings
        self.pos_scale = nn.Parameter(torch.ones(1))

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize output projection with smaller weights
        nn.init.normal_(self.flow_head[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.flow_head[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Discrete state and time as input, return single flow value.

        Args:
            x: [B, *, L] shape of discrete state tensor
            t: [B, *] shape of time tensor

        Returns:
            [B, *] shape of flow value tensor
        """

        *batch_dim, L = x.shape
        x = x.view(-1, L)
        t = t.reshape(-1)

        # Token embedding, [B, L] -> [B, L, H]
        x_emb = self.token_embedding(x)
        x_emb = self.embed_projection(x_emb) * self.embed_scale

        # Add positional embeddings
        pos_emb = self.pos_embedding(self.position_indices).unsqueeze(0)  # [1, L, H]
        h = x_emb + self.pos_scale * pos_emb

        # Time embedding, [B] -> [B, H]
        t_emb = self.time_embedding(t)

        # Add time embeddings as a global token
        t_emb_expanded = t_emb.unsqueeze(-2)  # [B, 1, H]
        t_expanded = t_emb_expanded.expand(x.shape[0], L, -1)  # [B, L, H]
        h = h + t_expanded

        # Transformer layers
        for layer in self.layers:
            # Self-attention with residual connection
            h_norm = layer["norm1"](h)
            attn_output, _ = layer["self_attn"](h_norm, h_norm, h_norm)
            h = h + attn_output

            # Feed-forward with residual connection
            h_norm = layer["norm2"](h)
            ffn_output = layer["ffn"](h_norm)
            h = h + ffn_output

        # Global pooling - take mean across sequence dimension
        h_pooled = h.mean(dim=-2)  # [B, H]

        # Flow prediction
        flow_value = self.flow_head(torch.cat([h_pooled, t_emb], dim=-1)).squeeze(
            -1
        )  # [B, *]

        return flow_value.view(*batch_dim)
