from ..... import backend as Z
from ...base import Form, TransformLayer, TransformSpec


class PadLayer(TransformLayer):
    def __init__(self, pad, x_ndim=None):
        super().__init__(x_ndim)
        self.pad = pad


class PadSpec(TransformSpec):
    def __init__(self, pad, spatial_ndim=None):
        super().__init__(spatial_ndim)
        self.pad = pad

    def make_layer(self, form):
        raise NotImplementedError

    def build_transform(self, form):
        layer = self.make_layer(form)
        out_shape = Z.pad_out_shape(form.shape, self.pad)
        form = Form(out_shape, form.dtype)
        return layer, form
