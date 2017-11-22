from .... import backend as Z
from ...base import TransformLayer, TransformSpec


class PadLayer(TransformLayer):
    def __init__(self, pad, ndim):
        super().__init__(ndim)
        self.pad = pad
        self.ndim = ndim


class PadSpec(TransformSpec):
    def __init__(self, pad, ndim):
        super().__init__(ndim)
        self.pad = pad
        self.ndim = ndim

    def make_layer(self, form):
        raise NotImplementedError

    def build_one(self, form):
        layer = self.make_layer(form)
        form = Z.pad_out_shape(form.shape, self.pad)
        return layer, form
