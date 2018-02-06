import numpy as np

from ... import backend as Z


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

    def add_param(self, x, trainable=True):
        if isinstance(x, np.ndarray):
            x = Z.numpy_to_device(x)
        if trainable:
            x = Z.variable(x)
            self._params.append(x)
        else:
            x = Z.constant(x)
        return x

    def forward_inner(self, xx, train):
        raise NotImplementedError

    def forward(self, xx, train):
        if self._x_ndim is not None:
            for x in xx:
                assert Z.ndim(x) == self._x_ndim
        return self.forward_inner(xx, train)


class TransformLayer(Layer):
    def transform(self, x, train):
        raise NotImplementedError

    def forward_inner(self, xx, train):
        assert len(xx) == 1
        x, = xx
        y = self.transform(x, train)
        return [y]


class MergeLayer(Layer):
    def merge(self, xx, train):
        raise NotImplementedError

    def forward_inner(self, xx, train):
        y = self.merge(xx, train)
        return [y]


class ForkLayer(Layer):
    def fork(self, x, train):
        raise NotImplementedError

    def forward_inner(self, xx, train):
        assert len(xx) == 1
        x, = xx
        return self.fork(x, train)


class FlexLayer(Layer):
    def flex(self, xx, train):
        raise NotImplementedError

    def forward_inner(self, xx, train):
        return self.flex(xx, train)
