from typing import Dict, Any

from .discrete_diffusion import (
    FactorizedProbDenoiser,
    CNNProbDenoiser,
    TransformerFactorizedProbDenoiser,
)
from .multi_modal import MultiModalDiffusionDrift


def create_mlp_model(
    num_tokens: int,
    grid_size: int,
    hidden_dim: int = 32,
    num_layers: int = 3,
    **kwargs: Dict[str, Any],
) -> FactorizedProbDenoiser:
    """
    Create a MLP-based denoiser model.

    Args:
        num_tokens: Number of tokens in the vocabulary
        grid_size: Size of the grid (unused for MLP)
        hidden_dim: Hidden dimension size
        num_layers: Number of layers in the model
        **kwargs: Additional parameters (unused)

    Returns:
        FactorizedProbDenoiser: Created MLP model
    """
    return FactorizedProbDenoiser(
        num_tokens=num_tokens,
        seq_length=grid_size * grid_size,  # Convert grid size to sequence length
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )


def create_cnn_model(
    num_tokens: int,
    grid_size: int,
    hidden_dim: int = 32,
    num_layers: int = 3,
    use_batch_norm: bool = True,
    **kwargs: Dict[str, Any],
) -> CNNProbDenoiser:
    """
    Create a CNN-based denoiser model.

    Args:
        num_tokens: Number of tokens in the vocabulary
        grid_size: Size of the grid
        hidden_dim: Hidden dimension size
        num_layers: Number of layers in the model
        use_batch_norm: Whether to use batch normalization
        **kwargs: Additional parameters (unused)

    Returns:
        CNNProbDenoiser: Created CNN model
    """
    return CNNProbDenoiser(
        num_tokens=num_tokens,
        grid_size=grid_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_batch_norm=use_batch_norm,
    )


def create_transformer_model(
    num_tokens: int,
    grid_size: int,
    hidden_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    **kwargs: Dict[str, Any],
) -> TransformerFactorizedProbDenoiser:
    """
    Create a Transformer-based denoiser model.

    Args:
        num_tokens: Number of tokens in the vocabulary
        grid_size: Size of the grid
        hidden_dim: Hidden dimension size (default: 512 for transformer)
        num_layers: Number of transformer layers (default: 6)
        num_heads: Number of attention heads
        dropout: Dropout rate
        **kwargs: Additional parameters (unused)

    Returns:
        TransformerFactorizedProbDenoiser: Created transformer model
    """
    return TransformerFactorizedProbDenoiser(
        num_tokens=num_tokens,
        seq_length=grid_size * grid_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )


def create_multimodal_mlp(
    num_tokens: int,
    grid_size: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    disc_embedding_dim: int = 16,
    cont_embedding_dim: int = 64,
    t_emb_dim: int = 64,
    act_fn: str = "SiLU",
    **kwargs: Dict[str, Any],
) -> MultiModalDiffusionDrift:
    """
    Create a multimodal MLP-based diffusion drift model.
    This is kept for backward compatibility.
    """
    return MultiModalDiffusionDrift(
        discrete_dim=grid_size * grid_size,
        continuous_dim=grid_size * grid_size,
        num_tokens=num_tokens,
        disc_embedding_dim=disc_embedding_dim,
        cont_embedding_dim=cont_embedding_dim,
        t_emb_dim=t_emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        act_fn=act_fn,
    )
