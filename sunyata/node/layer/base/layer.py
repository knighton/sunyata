import numpy as np

from .... import backend as Z


class Layer(object):
    def __init__(self, x_ndim):
        self._x_ndim = x_ndim
        self._params = []

    def x_ndim(self):
        return self._x_ndim

    def batch_ndim(self):
        return self._x_ndim - 1

    def spatial_ndim(self):
        return self._x_ndim - 2

    def params(self):
        return self._params

    def add_param(self, x, train=True):
        if isinstance(x, np.ndarray):
            x = Z.numpy_to_device(x)
        if train:
            x = Z.variablde(x)
            self._params.append(x)
        else:
            x = Z.constant(x)
        return x

    def forward_inner(self, xx, is_training):
        raise NotImplementedError

    def forward(self, xx, is_training):
        if self._x_ndim is not None:
            for x in xx:
                assert Z.ndim(x) == self._x_ndim
        return self.forward_inner(xx, is_training)


class TransformLayer(Layer):
    def transform(self, x, is_training):
        raise NotImplementedError

    def forward_inner(self, xx, is_training):
        assert len(xx) == 1
        x, = xx
        y = self.transform(x, is_training)
        return [y]


class MergeLayer(Layer):
    def merge(self, xx, is_training):
        raise NotImplementedError

    def forward_inner(self, xx, is_training):
        y = self.merge(xx, is_training)
        return [y]


class ForkLayer(Layer):
    def fork(self, x, is_training):
        raise NotImplementedError

    def forward_inner(self, xx, is_training):
        assert len(xx) == 1
        x, = xx
        return self.fork(x, is_training)


class FlexLayer(Layer):
    def flex(self, xx, is_training):
        raise NotImplementedError

    def forward_inner(self, xx, is_training):
        return self.flex(xx, is_training)
