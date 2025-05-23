import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable

from .emb import SinusoidalPositionEmbeddings


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, act_fn):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act_fn = act_fn()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return x + self.linear2(self.act_fn(self.linear1(x)))


class MultiModalDiffusionDrift(nn.Module):
    def __init__(
        self,
        discrete_dim: int,
        continuous_dim: int,
        num_tokens: int,
        disc_embedding_dim: int = 16,
        cont_embedding_dim: int = 32,
        t_emb_dim: int = 32,
        hidden_dim: int = 64,
        use_residual_blocks: bool = False,
        use_layer_norm: bool = False,
        num_hidden_layers: int = 2,
        num_res_blocks: int = 2,
        num_pred_layers: int = 2,
        act_fn: str = "SiLU",
        **kwargs,
    ):
        super().__init__()
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim
        self.num_tokens = num_tokens
        self.disc_embedding_dim = disc_embedding_dim
        self.cont_embedding_dim = cont_embedding_dim
        self.t_emb_dim = t_emb_dim
        self.hidden_dim = hidden_dim
        self.use_residual_blocks = use_residual_blocks
        self.num_hidden_layers = num_hidden_layers
        self.num_res_blocks = num_res_blocks
        self.num_pred_layers = num_pred_layers

        if act_fn == "SiLU":
            self.act_fn = nn.SiLU
        elif act_fn == "GELU":
            self.act_fn = nn.GELU
        else:
            raise ValueError(f"Invalid activation function: {act_fn}")

        # Token embeddings for discrete variables
        self.discrete_embedding = nn.Embedding(num_tokens, disc_embedding_dim)

        # Continuous variable embedding
        self.continuous_projection = nn.Sequential(
            nn.Linear(continuous_dim, cont_embedding_dim),
            self.act_fn(),
            nn.Linear(cont_embedding_dim, cont_embedding_dim),
        )

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim),
            self.act_fn(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        input_dim = discrete_dim * disc_embedding_dim + cont_embedding_dim + t_emb_dim
        self.mlp = self._build_mlp(input_dim=input_dim, use_layer_norm=use_layer_norm)

        # Drift prediction heads
        layers = []
        for _ in range(num_pred_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act_fn()])
            hidden_dim = hidden_dim
        self.continuous_head = nn.Sequential(
            *layers, nn.Linear(hidden_dim, continuous_dim)
        )

        layers = []
        for _ in range(num_pred_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.act_fn()])
            hidden_dim = hidden_dim
        self.discrete_head = nn.Sequential(
            *layers, nn.Linear(hidden_dim, discrete_dim * num_tokens)
        )

    def _build_mlp(self, input_dim, use_layer_norm):

        layers = []
        hidden_dim = self.hidden_dim

        if self.use_residual_blocks:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(self.num_res_blocks):
                layers.append(ResidualBlock(hidden_dim, self.act_fn))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
        else:
            for _ in range(self.num_hidden_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(self.act_fn())
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                input_dim = hidden_dim

        return nn.Sequential(*layers)

    def split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split the input into continuous and discrete variables"""
        return x[:, : self.continuous_dim], x[:, self.continuous_dim :]

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, ret_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Combined input [batch_size, continuous_dim + discrete_dim]
            t: Time [batch_size, 1] (or [batch_size])

        Returns:
            Tuple of:
                - Continuous drift [batch_size, continuous_dim]
                - Discrete drift [batch_size, discrete_dim, num_tokens]
        """
        if x.dim() == t.dim():
            t = t.squeeze(-1)

        x_continuous, x_discrete = self.split(x)

        # Time embedding
        t_emb = self.time_embedding(t)

        # Process continuous variables
        continuous_emb = self.continuous_projection(
            x_continuous
        )  # [batch, conti_emb_dim]

        # Process discrete variables
        discrete_emb = self.discrete_embedding(
            x_discrete.long()
        )  # [batch, discrete_dim, embedding_dim]

        # Flatten embeddings
        discrete_flat = discrete_emb.view(
            discrete_emb.size(0), -1
        )  # [batch, discrete_dim * embedding_dim]

        # Combine with time
        combined = torch.cat([continuous_emb, discrete_flat, t_emb], dim=-1)

        # Process through MLP
        h = self.mlp(combined)

        # Predict drifts
        continuous_drift = self.continuous_head(h)
        discrete_drift = self.discrete_head(h).reshape(
            -1, self.discrete_dim, self.num_tokens
        )

        mask_token_idx = self.num_tokens - 1
        discrete_drift[..., mask_token_idx] = float("-inf")

        if ret_logits:
            return continuous_drift, discrete_drift
        else:
            return continuous_drift, torch.softmax(discrete_drift, dim=-1)


class MultiModalStateFlow(nn.Module):
    """
    State flow model for multimodal input that contains both discrete and continuous components.

    Args:
        discrete_dim: Number of discrete variables
        continuous_dim: Number of continuous variables
        num_tokens: Number of possible values for each discrete variable
        disc_embedding_dim: Dimension of discrete token embeddings
        cont_embedding_dim: Dimension for processing continuous variables
        t_emb_dim: Dimension of time embedding
        hidden_dim: Hidden dimension of the network
        use_residual_blocks: Whether to use residual blocks in MLP
        use_layer_norm: Whether to use layer normalization
        num_hidden_layers: Number of hidden layers if not using residual blocks
        num_res_blocks: Number of residual blocks if using them
        energy_fn: Function to compute local energy (optional)
        act_fn: Activation function to use
    """

    def __init__(
        self,
        discrete_dim: int,
        continuous_dim: int,
        num_tokens: int,
        disc_embedding_dim: int = 4,
        cont_embedding_dim: int = 32,
        t_emb_dim: int = 32,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        num_res_blocks: Optional[int] = None,
        use_residual_blocks: bool = False,
        act_fn: str = "SiLU",
        **kwargs,
    ):
        super().__init__()
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim
        self.num_tokens = num_tokens
        self.disc_embedding_dim = disc_embedding_dim
        self.cont_embedding_dim = cont_embedding_dim
        self.t_emb_dim = t_emb_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_res_blocks = num_res_blocks
        self.use_residual_blocks = use_residual_blocks
        if act_fn == "SiLU":
            self.act_fn = nn.SiLU
        elif act_fn == "GELU":
            self.act_fn = nn.GELU
        else:
            raise ValueError(f"Invalid activation function: {act_fn}")

        # Token embeddings for discrete variables
        self.discrete_embedding = nn.Embedding(num_tokens, disc_embedding_dim)

        # Continuous variable embedding
        self.continuous_projection = nn.Sequential(
            nn.Linear(continuous_dim, cont_embedding_dim),
            self.act_fn(),
            nn.Linear(cont_embedding_dim, cont_embedding_dim),
        )

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.GELU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        # Calculate input dimension for the MLP
        input_dim = discrete_dim * disc_embedding_dim + cont_embedding_dim + t_emb_dim

        # Build the MLP
        self.mlp = self._build_mlp(input_dim=input_dim)

        # Final flow value prediction layer
        self.flow_head = nn.Linear(hidden_dim, 1)

    def _build_mlp(self, input_dim):
        layers = []
        hidden_dim = self.hidden_dim

        if self.use_residual_blocks:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(self.num_res_blocks):
                layers.append(ResidualBlock(hidden_dim, self.act_fn))
        else:
            for _ in range(self.num_hidden_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(self.act_fn())
                input_dim = hidden_dim

        return nn.Sequential(*layers)

    def split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split the input into continuous and discrete variables"""
        return x[..., : self.continuous_dim], x[..., self.continuous_dim :].long()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Multimodal state input and time as input, return flow value.

        Args:
            x: [B, continuous_dim + discrete_dim] combined input tensor
            t: [B, 1] time tensor (or [B])

        Returns:
            [B] flow value tensor
        """
        if x.dim() == t.dim():
            t = t.squeeze(-1)

        batch_dims = x.shape[:-1]
        x_continuous, x_discrete = self.split(x)

        # Time embedding
        t_emb = self.time_embedding(t)

        # Process continuous variables
        continuous_emb = self.continuous_projection(
            x_continuous
        )  # [batch, cont_emb_dim]

        # Process discrete variables, [B, discrete_dim] -> [B, discrete_dim, embedding_dim]
        discrete_emb = self.discrete_embedding(x_discrete)

        # Flatten discrete embeddings
        discrete_flat = discrete_emb.view(*batch_dims, -1)

        # Combine all embeddings
        combined = torch.cat([continuous_emb, discrete_flat, t_emb], dim=-1)

        # Process through MLP
        h = self.mlp(combined)

        # Predict flow value
        flow_value = self.flow_head(h).squeeze(-1)

        return flow_value
