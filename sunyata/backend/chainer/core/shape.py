from chainer import functions as F

from ...base.core.shape import BaseShapeAPI


class ChainerShapeAPI(BaseShapeAPI):
    def ndim(self, x):
        return x.ndim

    def shape(self, x):
        return x.shape

    def size(self, x):
        return x.size

    def reshape(self, x, shape):
        return x.reshape(shape)

    def expand_dims(self, x, axis):
        return F.array.expand_dims.expand_dims(x, axis)
