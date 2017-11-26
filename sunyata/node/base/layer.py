import numpy as np

from ... import backend as Z


class Layer(object):
    def __init__(self):
        self.params = []

    def add_param(self, x, trainable=True):
        if isinstance(x, np.ndarray):
            x = Z.numpy_to_device(x)
        if trainable:
            x = Z.variable(x)
            self.params.append(x)
        else:
            x = Z.constant(x)
        return x

    def get_params(self):
        return self.params

    def forward_multi(self, xx, is_training):
        raise NotImplementedError


class MergeLayer(Layer):
    pass


class TransformLayer(Layer):
    def __init__(self, ndim=None):
        super().__init__()
        self.in_ndim = ndim

    def forward_one(self, x, is_training):
        raise NotImplementedError

    def forward_multi(self, xx, is_training):
        assert len(xx) == 1
        x, = xx
        if self.in_ndim is not None:
            assert Z.ndim(x) == self.in_ndim
        x = self.forward_one(x, is_training)
        return [x]
