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
        return F.expand_dims(x, axis)

    def squeeze(self, x, axis=None):
        return F.squeeze(x, axis)

    def tile(self, x, reps):
        return F.tile(x, reps)

    def transpose(self, x, axes):
        return F.transpose(x, axes)

    def concat(self, xx, axis):
        return F.concat(xx, axis)

    def stack(self, xx, axis=0):
        return F.stack(xx, axis)
