import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_ising_samples(samples, figsize=(8, 8)):
    """
    Plot Ising model samples in a 2x2 grid.

    Args:
        samples: Tensor of shape [batch_size, grid_size, grid_size]
        figsize: Figure size tuple

    Returns:
        fig: Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    # Create colorbar axis with adjusted position
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])

    for i in range(4):
        im = axes[i].imshow(samples[i], cmap="binary", vmin=-1, vmax=1)
        axes[i].grid(True, which="both", color="gray", linewidth=0.5)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Add colorbar
    fig.colorbar(im, cax=cbar_ax, ticks=[-1, 0, 1])
    cbar_ax.set_ylabel("Spin Value")

    # Adjust layout to prevent overlap
    plt.subplots_adjust(right=0.9)

    return fig


def plot_histogram(
    values,
    gt_values=None,
    title="Histogram",
    xlabel="Value",
    ylabel="Frequency",
    color="skyblue",
    gt_color="lightgray",
    bins=20,
    figsize=(8, 4),
):
    """
    Plot a histogram of values, optionally comparing with ground truth values.

    Args:
        values: Array of values to plot
        gt_values: Optional array of ground truth values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Color for main histogram
        gt_color: Color for ground truth histogram
        bins: Number of histogram bins or bin edges
        figsize: Figure size tuple

    Returns:
        fig: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if gt_values is not None:
        # Use the prescribed binning directly if provided
        if isinstance(bins, (list, np.ndarray)):
            specified_bins = bins
        else:
            # Calculate common bins for both distributions
            all_values = np.concatenate([values, gt_values])
            specified_bins = np.histogram_bin_edges(all_values, bins=bins)

        ax.hist(
            gt_values,
            bins=specified_bins,
            color=gt_color,
            edgecolor="black",
            alpha=0.8,
            label="Ground Truth",
            density=True,
        )

        ax.hist(
            values,
            bins=specified_bins,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.7,
            label="Generated",
            density=True,
        )

    else:
        ax.hist(
            values,
            bins=bins,
            color=color,
            alpha=0.7,
            label="Generated",
            density=True,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()

    return fig


def plot_projection(
    samples,
    ground_truth=None,
    title="2D Projection",
    proj_dim1=0,
    proj_dim2=1,
    lim=None,
):
    """
    Visualize the projection of data onto two dimensions using KDE plot for ground truth
    and scatter plot for generated samples.

    Args:
        samples: Samples to visualize [batch_size, ndim]
        ground_truth: Ground truth samples for comparison (optional)
        title: Plot title
        proj_dim1: First projection dimension
        proj_dim2: Second projection dimension

    Returns:
        matplotlib figure object
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))

    if lim is not None:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])

    # Extract dimensions from samples
    x_samples = samples[:, proj_dim1].detach().cpu().numpy()
    y_samples = samples[:, proj_dim2].detach().cpu().numpy()

    # First plot KDE of ground truth if provided
    if ground_truth is not None:
        x_gt = ground_truth[:, proj_dim1].detach().cpu().numpy()
        y_gt = ground_truth[:, proj_dim2].detach().cpu().numpy()

        sns.kdeplot(
            x=x_gt,
            y=y_gt,
            fill=True,
            alpha=1.0,
            levels=10,
            cmap="viridis",
            ax=ax,
            label="Ground Truth KDE",
            thresh=0,
        )

    # Plot scatter for generated samples
    ax.scatter(
        x_samples, y_samples, alpha=0.3, label="Generated Samples", color="red", s=20
    )

    return fig


def plot_contour_and_sample(
    samples,
    energy,
    device="cuda",
    proj_dim1=0,
    proj_dim2=1,
    title="Energy Contour and Samples",
):
    """
    Visualize energy function contours and samples with scatter plot.

    Args:
        samples: Samples to visualize [batch_size, ndim]
        energy: Energy function object
        device: Device to use for computation
        proj_dim1: First projection dimension
        proj_dim2: Second projection dimension
        title: Plot title

    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract dimensions from samples
    x_samples = samples[:, proj_dim1].detach().cpu().numpy()
    y_samples = samples[:, proj_dim2].detach().cpu().numpy()

    # Create grid for energy contours
    x = torch.linspace(-20, 20, 100, device=device)
    y = torch.linspace(-20, 20, 100, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Calculate energy function
    with torch.no_grad():
        Z = energy(grid).reshape(100, 100).cpu().numpy()

    # Draw contours
    contour = ax.contourf(
        X.cpu().numpy(), Y.cpu().numpy(), Z, levels=50, cmap="viridis", alpha=0.9
    )
    plt.colorbar(contour, label="Energy")

    # Plot samples as scatter
    ax.scatter(
        x_samples, y_samples, alpha=0.5, label="Generated Samples", color="red", s=20
    )

    ax.set_title(title)
    ax.set_xlabel(f"Dimension {proj_dim1}")
    ax.set_ylabel(f"Dimension {proj_dim2}")

    return fig
