from ... import backend as Z
from ..base import TransformLayer, TransformSpec


class GaussianDropoutLayer(TransformLayer):
    def __init__(self, rate, keep_axis, ndim):
        super().__init__(ndim)
        self.rate = rate
        self.keep_axis = keep_axis

    def forward_one(self, x, is_training):
        return Z.gaussian_dropout(x, is_training, self.rate, self.keep_axis)


class GaussianDropoutSpec(TransformSpec):
    def __init__(self, rate=0.5, keep_axis=None, ndim=None):
        super().__init__(ndim)
        self.rate = rate
        self.keep_axis = keep_axis

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        layer = GaussianDropoutLayer(self.rate, self.keep_axis, ndim)
        return layer, form
