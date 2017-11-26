import numpy as np

from .base import Initializer


def _ones(shape, dtype='float32'):
    return np.ones(shape, dtype)


class Ones(Initializer):
    def __call__(self, shape, dtype='float32', meaning=None):
        return _ones(shape, dtype)


ones = Ones
