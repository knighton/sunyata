import tensorflow as tf

from ...base.core.tensor import BaseTensorAPI


class TensorFlowTensorAPI(BaseTensorAPI):
    def __init__(self):
        BaseTensorAPI.__init__(self)

    def zeros(self, shape, dtype=None, device=None):
        with self.device_scope(device):
            return tf.zeros(shape, self.dtype(dtype))

    def zeros_like(self, like, dtype=None, device=None):
        with self.device_scope(device):
            return tf.zeros_like(like, self.dtype(dtype))

    def ones(self, shape, dtype=None, device=None):
        with self.device_scope(device):
            return tf.ones(shape, self.dtype(dtype))

    def ones_like(self, like, dtype=None, device=None):
        with self.device_scope(device):
            return tf.ones_like(like, self.dtype(dtype))

    def full(self, shape, value, dtype=None, device=None):
        with self.device_scope(device):
            return tf.fill(shape, value, self.dtype(dtype))

    def full_like(self, like, value, dtype=None, device=None):
        with self.device_scope(device):
            return tf.fill(self.shape(like), value, self.dtype(dtype))

    def arange(self, begin, end, step=1, dtype=None, device=None):
        with self.device_scope(device):
            return tf.range(begin, end, step, self.dtype(dtype))

    def eye(self, dim, dtype=None, device=None):
        with self.device_scope(device):
            return tf.eye(dim, dtype=self.dtype(dtype))

    def random_uniform(self, shape, min=0, max=1, dtype=None, device=None):
        with self.device_scope(device):
            return tf.random_uniform(shape, min, max, self.dtype(dtype))

    def random_normal(self, shape, mean=0, std=1, dtype=None, device=None):
        with self.device_scope(device):
            return tf.random_normal(shape, mean, std, self.dtype(dtype))
