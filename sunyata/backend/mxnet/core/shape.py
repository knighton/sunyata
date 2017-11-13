import mxnet as mx

from ...base.core.shape import BaseShapeAPI


class MXNetShapeAPI(BaseShapeAPI):
    def __init__(self):
        BaseShapeAPI.__init__(self)

    def ndim(self, x):
        return len(x.ndim)

    def shape(self, x):
        return x.shape

    def size(self, x):
        return x.size

    def reshape(self, x, shape):
        return mx.nd.reshape(x, shape)

    def expand_dims(self, x, axis):
        return mx.nd.expand_dims(x, axis)
