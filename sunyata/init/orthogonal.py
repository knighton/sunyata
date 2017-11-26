import numpy as np

from .base import Initializer


def _orthogonal(shape, scale=1, dtype='float32'):
    shape_2d = np.prod(shape[:-1]), shape[-1]
    x = np.random.normal(0, 1, shape_2d)
    u, s, v = np.linalg.svd(x, full_matrices=False)
    x = u if u.shape == shape_2d else v
    return scale * x.reshape(shape).astype(dtype)


class Orthogonal(Initializer):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, shape, dtype='float32', meaning=None):
        return _orthogonal(shape, self.scale, dtype)


orthogonal = Orthogonal
