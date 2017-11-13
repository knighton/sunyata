import numpy as np

from ..base import APIMixin


class BaseTensorAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def zeros(self, shape, dtype=None, device=None):
        x = np.zeros(shape, self.dtype(dtype))
        return self.numpy_to_device(x, None, device)

    def zeros_like(self, like, dtype=None, device=None):
        return self.zeros(self.shape(like), dtype, device)

    def ones(self, shape, dtype=None, device=None):
        x = np.ones(shape, self.dtype(dtype))
        return self.numpy_to_device(x, device)

    def ones_like(self, like, dtype=None, device=None):
        return self.ones(self.shape(like), dtype, device)

    def full(self, shape, value, dtype=None, device=None):
        x = np.full(shape, value, self.dtype(dtype))
        return self.numpy_to_device(x, device)

    def full_like(self, like, value, dtype=None, device=None):
        return self.full(self.shape(like), value, dtype, device)

    def arange(self, begin, end, step=1, dtype=None, device=None):
        x = np.arange(begin, end, step, dtype=dtype)
        return self.numpy_to_device(x, device)

    def eye(self, dim, dtype=None, device=None):
        x = np.eye(dim, dtype=self.dtype(dtype))
        return self.numpy_to_device(x, device)

    def random_uniform(self, shape, min=0, max=1, dtype=None, device=None):
        x = np.random.uniform(min, max, shape)
        return self.cast_numpy_to_device(x, dtype, device)

    def random_normal(self, shape, mean=0, std=1, dtype=None, device=None):
        x = np.random.normal(mean, std, shape)
        return self.cast_numpy_to_device(x, dtype, device)
