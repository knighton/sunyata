from ..... import backend as Z
from ...base import Form, TransformLayer, TransformSpec


class UpsampleLayer(TransformLayer):
    def __init__(self, scale, x_ndim):
        super().__init__(x_ndim)
        self.scale = scale


class UpsampleSpec(TransformSpec):
    def __init__(self, scale, spatial_ndim):
        super().__init__(spatial_ndim)
        self.scale = scale

    def make_layer(self, form):
        raise NotImplementedError

    def build_transform(self, form):
        layer = self.make_layer(form)
        out_shape = Z.upsample_out_shape(form.batch_shape, self.scale)
        form = Form(out_shape, form.dtype)
        return layer, form
