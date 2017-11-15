import numpy as np

from ... import backend as Z
from ..base import Form, TransformLayer, TransformSpec


class FlattenLayer(TransformLayer):
    def forward_one(self, x, is_training):
        return Z.reshape(x, (Z.shape(x)[0], -1))


class FlattenSpec(TransformSpec):
    def build_one(self, form):
        out_shape = (int(np.prod(form.shape)),)
        form = Form(out_shape, form.dtype)
        return FlattenLayer(), form
