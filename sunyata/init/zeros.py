import numpy as np

from .base import Initializer


def _zeros(shape, dtype='float32'):
    return np.zeros(shape, dtype)


class Zeros(Initializer):
    def __call__(self, shape, dtype='float32', meaning=None):
        return _zeros(shape, dtype)


zeros = Zeros
