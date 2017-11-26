import numpy as np

from .base import Initializer


def _normal(shape, mean=0, std=1, dtype='float32'):
    return np.random.normal(mean, std, shape).astype(dtype)


class Normal(Initializer):
    def __init__(self, mean=0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, shape, dtype='float32', meaning=None):
        return _normal(shape, self.mean, self.std, dtype)


normal = Normal
