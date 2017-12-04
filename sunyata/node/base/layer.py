import numpy as np

from ... import backend as Z


class Layer(object):
    def __init__(self):
        self._input_ndim = None
        self._params = []

    def initialize_input_ndim(self, ndim):
        assert isinstance(ndim, int)
        assert 1 <= ndim
        assert self._input_ndim is None
        self._input_ndim = ndim

    def add_param(self, x, train=True):
        if isinstance(x, np.ndarray):
            x = Z.numpy_to_device(x)
        if train:
            x = Z.variablde(x)
            self._params.append(x)
        else:
            x = Z.constant(x)
        return x

    def params(self):
        return self._params

    def forward_inner(self, xx, is_training):
        raise NotImplementedError

    def forward(self, xx, is_training):
        if self._input_ndim is not None:
            for x in xx:
                assert Z.ndim(x) == self._input_ndim
        return self.forward_inner(xx, is_training)


class Transform(Layer):
    def transform(self, x, is_training):
        raise NotImplementedError

    def forward_inner(self, xx, is_training):
        assert len(xx) == 1
        x, = xx
        y = self.transform(x, is_training)
        return [y]


class Merge(Layer):
    def merge(self, xx, is_training):
        raise NotImplementedError

    def forward_inner(self, xx, is_training):
        y = self.merge(xx, is_training)
        return [y]


class Fork(Layer):
    def fork(self, x, is_training):
        raise NotImplementedError

    def forward_inner(self, xx, is_training):
        assert len(xx) == 1
        x, = xx
        return self.fork(x, is_training)


class Flex(Layer):
    def flex(self, xx, is_training):
        raise NotImplementedError

    def forward_inner(self, xx, is_training):
        return self.flex(xx, is_training)
