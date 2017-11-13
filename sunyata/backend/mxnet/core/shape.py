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

    def squeeze(self, x, axis=None):
        from_shape = tuple(x.shape)
        assert -len(from_shape) < axis < len(from_shape)
        axis %= len(from_shape)
        assert from_shape[axis] == 1
        to_shape = from_shape[:axis] + from_shape[axis + 1:]
        return mx.nd.reshape(x, to_shape)

    def tile(self, x, reps):
        return mx.nd.tile(x, reps)

    def transpose(self, x, axes):
        return mx.nd.transpose(x, axes)

    def concat(self, xx, axis):
        return mx.nd.concat(*xx, dim=axis)

    def stack(self, xx, axis=0):
        return mx.nd.stack(*xx, axis=axis)
