from .... import backend as Z
from ...base import Form, TransformLayer, TransformSpec


class UpsampleLayer(TransformLayer):
    def __init__(self, scale, ndim):
        super().__init__(ndim)
        self.scale = scale
        self.ndim = ndim


class UpsampleSpec(TransformSpec):
    def __init__(self, scale, ndim):
        super().__init__(ndim)
        self.scale = scale
        self.ndim = ndim

    def make_layer(self, form):
        raise NotImplementedError

    def build_one(self, form):
        layer = self.make_layer(form)
        out_shape = Z.upsample_out_shape(form.shape, self.scale)
        form = Form(out_shape, form.dtype)
        return layer, form
