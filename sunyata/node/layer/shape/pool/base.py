from .... import backend as Z
from ...base import Form, TransformLayer, TransformSpec


class PoolLayer(TransformLayer):
    def __init__(self, face, stride, pad, ndim):
        super().__init__(ndim)
        self.face = face
        self.stride = stride
        self.pad = pad


class PoolSpec(TransformSpec):
    def __init__(self, face=2, stride=None, pad=0, ndim=None):
        super().__init__(ndim)
        self.face = face
        self.stride = face if stride is None else stride
        self.pad = pad

    def make_layer(self, form):
        raise NotImplementedError

    def build_one(self, form):
        layer = self.make_layer(form)
        out_shape = Z.pool_out_shape(
            form.shape, self.face, self.stride, self.pad)
        form = Form(out_shape, form.dtype)
        return layer, form
