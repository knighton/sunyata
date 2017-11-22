import numpy as np

from ... import backend as Z
from ..base import Form, TransformLayer, TransformSpec


class FlattenLayer(TransformLayer):
    def __init__(self, ndim):
        super().__init__(ndim)

    def forward_one(self, x, is_training):
        return Z.reshape(x, (Z.shape(x)[0], -1))


class FlattenSpec(TransformSpec):
    def __init__(self, ndim=None):
        super().__init__(ndim)

    def build_one(self, form):
        ndim = self.in_ndim(form.shape)
        out_shape = (int(np.prod(form.shape)),)
        form = Form(out_shape, form.dtype)
        return FlattenLayer(ndim), form
