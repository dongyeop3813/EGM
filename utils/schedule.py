import torch


class LossScalerSchedule:
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearLossScaler(LossScalerSchedule):
    def __init__(
        self,
        max_lambda: float = 10.0,
        max_step: int = int(1e8),
    ):
        self.max_lambda = max_lambda
        self.max_step = max_step

    def __call__(self, step: int) -> float:
        return self.max_lambda * (1.0 - step / self.max_step)
