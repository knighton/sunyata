import numpy as np

from .... import backend as Z
from ..base import Form, node_wrap, TransformLayer, TransformSpec


class FlattenLayer(TransformLayer):
    def __init__(self, x_ndim):
        super().__init__(x_ndim)

    def transform(self, x, train):
        return Z.reshape(x, (Z.shape(x)[0], -1))


class FlattenSpec(TransformSpec):
    def __init__(self, spatial_ndim=None):
        super().__init__(spatial_ndim)

    def build_transform(self, form):
        out_shape = (int(np.prod(form.batch_shape)),)
        form = Form(out_shape, form.dtype)
        return FlattenLayer(self.x_ndim()), form


node_wrap(FlattenSpec)
