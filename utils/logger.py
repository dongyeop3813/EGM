import abc
import os
import math
import matplotlib.pyplot as plt
import wandb
from omegaconf import DictConfig, OmegaConf


def extract_algorithm(target_str):
    """Extracts the algorithm name from the target string."""
    if target_str is None:
        return "unknown"
    parts = target_str.split(".")
    return parts[-1] if parts else "unknown"


class Logger(abc.ABC):
    @abc.abstractmethod
    def log_loss(self, loss: dict):
        """
        Logs training losses.

        Args:
            loss: Dictionary containing loss values
        """
        pass

    @abc.abstractmethod
    def log_grad(self, grad):
        """
        Logs gradient information.

        Args:
            grad: Object containing gradient information
        """
        pass

    @abc.abstractmethod
    def log_metric(self, metrics: dict):
        """
        Logs evaluation metrics.

        Args:
            metrics: Dictionary containing metric values
        """
        pass

    @abc.abstractmethod
    def log_visual(self, visuals: dict):
        """
        Logs visualization results.

        Args:
            visuals: Dictionary containing visualization objects
        """
        pass

    @abc.abstractmethod
    def finish(self):
        """Finishes the logging session."""
        pass


class NullLogger(Logger):
    """
    Do nothing logger.
    """

    def __init__(self, cfg: DictConfig, output_dir: str):
        pass

    def add_hyperparams(self, cfg: DictConfig):
        pass

    def log_loss(self, loss: dict):
        pass

    def log_grad(self, grad):
        pass

    def log_metric(self, metrics: dict):
        pass

    def log_visual(self, visuals: dict, prefix: str = ""):
        plt.close("all")

    def finish(self):
        pass


class WandbLogger(Logger):
    def __init__(
        self,
        cfg: DictConfig,
        output_dir: str,
    ):
        # Convert OmegaConf object to python dictionary.

        self.run = wandb.init(
            project=cfg.wandb.project,
            config={
                **OmegaConf.to_container(cfg, resolve=True),
                "output_dir": output_dir,
            },
            tags=cfg.wandb.tags,
            group=cfg.wandb.get("group", None),
            name=cfg.wandb.get("name", None),
        )

        self.output_dir = output_dir

    def add_hyperparams(self, cfg: dict):
        self.run.config.update(cfg)

    def log_loss(self, loss: dict):
        loss = {
            f"train/{k}": (v if not math.isinf(v) and not math.isnan(v) else 1e20)
            for k, v in loss.items()
        }
        self.run.log(loss)

    def log_grad(self, grad):
        grad = {f"grad/{k}": v for k, v in grad.items()}
        self.run.log(grad)

    def log_metric(self, metrics: dict):
        metrics = {
            f"eval/{k}": (v if not math.isinf(v) and not math.isnan(v) else 1e20)
            for k, v in metrics.items()
        }

        self.run.log(metrics)

    def log_visual(self, visuals: dict, prefix: str = ""):
        for visual_name, fig in visuals.items():
            fig.savefig(
                f"{self.output_dir}/{prefix}{visual_name}.pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=1000,
            )

            self.run.log({f"visuals/{prefix}{visual_name}": wandb.Image(fig)})

        # Prevent too many plt objects from remaining open
        plt.close("all")

    def finish(self):
        if hasattr(self, "run"):
            self.run.finish()
            self.run = None
