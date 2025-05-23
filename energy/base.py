import abc
from torch.utils.data import Dataset
import torch


class BaseSet(abc.ABC, Dataset):
    def __init__(self, dim):
        self.data_ndim = dim

    @property
    def gt_logz(self):
        raise NotImplementedError

    @abc.abstractmethod
    def energy(self, x):
        return

    def unnorm_pdf(self, x):
        return torch.exp(-self.energy(x))

    def score(self, x):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                (-self.energy(copy_x)).sum().backward()
                grad_energy = copy_x.grad.data
            return grad_energy

    @property
    def ndim(self):
        return self.data_ndim

    def sample(self, batch_size):
        del batch_size
        raise NotImplementedError

    def log_reward(self, x):
        return -self.energy(x)
