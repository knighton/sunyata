from .... import backend as Z
from ...base import TransformLayer, TransformSpec


class PadLayer(TransformLayer):
    def __init__(self, pad, ndim):
        self.pad = pad
        self.ndim = ndim


class PadSpec(TransformSpec):
    def __init__(self, pad, ndim):
        self.pad = pad
        self.ndim = ndim

    def make_layer(self, ndim):
        raise NotImplementedError

    def build_one(self, form):
        if self.ndim is None:
            ndim = len(form.shape) + 1
        else:
            ndim = self.ndim + 2
            assert len(form.shape) + 1 == ndim
        layer = self.make_layer(ndim)
        form = Z.pad_out_shape(form.shape, self.pad)
        return layer, form
