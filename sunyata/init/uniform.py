import numpy as np

from .base import Initializer


def _uniform(shape, min, max, dtype):
    return np.random.uniform(min, max, shape).astype(dtype)


class Uniform(Initializer):
    def __init__(self, min=-0.05, max=0.05):
        self.min = min
        self.max = max

    def __call__(self, shape, dtype='float32', meaning=None):
        return _uniform(shape, self.min, self.max, dtype)


uniform = Uniform
