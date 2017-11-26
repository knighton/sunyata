import numpy as np

from .base import Initializer


def _constant(shape, value, dtype='float32'):
    return np.full(shape, value, dtype)


class Constant(Initializer):
    def __init__(self, value):
        self.value = value

    def __call__(self, shape, dtype='float32', meaning=None):
        return _constant(shape, self.value, dtype)


constant = Constant
