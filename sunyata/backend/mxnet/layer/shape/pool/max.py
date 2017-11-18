import mxnet as mx

from .....base.layer.shape.pool.max import BaseMaxPoolAPI


class MXNetMaxPoolAPI(BaseMaxPoolAPI):
    def __init__(self):
        BaseMaxPoolAPI.__init__(self)

    def _max_pool(self, x, face, stride, pad, ndim):
        if ndim is None:
            ndim = self.ndim(x)
        else:
            assert self.ndim(x) == ndim
        face = self.to_shape(face, ndim)
        stride = self.to_shape(stride, ndim)
        pad = self.to_shape(pad, ndim)
        return mx.nd.Pooling(data=x, kernel=face, stride=stride, pad=pad,
                             pool_type='max')

    def max_pool(self, x, face, stride, pad):
        return self._max_pool(x, face, stride, pad, None)

    def max_pool1d(self, x, face, stride, pad):
        return self._max_pool(x, face, stride, pad, 1)

    def max_pool2d(self, x, face, stride, pad):
        return self._max_pool(x, face, stride, pad, 2)

    def max_pool3d(self, x, face, stride, pad):
        return self._max_pool(x, face, stride, pad, 3)
