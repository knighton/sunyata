import numpy as np

from .base import Initializer


def _eye(dim, scale=1, dtype='float32'):
    x = scale * np.eye(dim)
    return x.astype(dtype)


class Eye(Initializer):
    def __init__(self, dim, scale=1):
        self.dim = dim
        self.scale = scale

    def __call__(self, shape, dtype='float32', meaning=None):
        return _eye(shape, self.scale, dtype)


eye = Eye
