from ... import backend as Z
from ..base import TransformLayer, TransformSpec


class GaussianDropoutLayer(TransformLayer):
    def __init__(self, rate, keep_axis, ndim):
        self.rate = rate
        self.keep_axis = keep_axis
        self.ndim = ndim

    def forward_one(self, x, is_training):
        assert Z.ndim(x) == self.ndim
        return Z.gaussian_dropout(x, is_training, self.rate, self.keep_axis)


class GaussianDropoutSpec(TransformSpec):
    def __init__(self, rate=0.5, keep_axis=None, ndim=None):
        self.rate = rate
        self.keep_axis = keep_axis
        self.ndim = ndim

    def build_one(self, form):
        if self.ndim is None:
            ndim = len(form.shape) + 1
        else:
            ndim = self.ndim
            assert len(form.shape) + 1 == ndim
        layer = GaussianDropoutLayer(self.rate, self.keep_axis, ndim)
        return layer, form
