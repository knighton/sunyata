import mxnet as mx

from .....base.layer.shape.pool.avg import BaseAvgPoolAPI


class MXNetAvgPoolAPI(BaseAvgPoolAPI):
    def __init__(self):
        BaseAvgPoolAPI.__init__(self)

    def _avg_pool(self, x, face, stride, pad, ndim):
        if ndim is None:
            ndim = self.ndim(x)
        else:
            assert self.ndim(x) == ndim
        face = self.to_shape(face, ndim)
        stride = self.to_shape(stride, ndim)
        pad = self.to_shape(pad, ndim)
        return mx.nd.Pooling(data=x, kernel=face, stride=stride, pad=pad,
                             pool_type='avg')

    def avg_pool(self, x, face, stride, pad):
        return self._avg_pool(x, face, stride, pad, None)

    def avg_pool1d(self, x, face, stride, pad):
        return self._avg_pool(x, face, stride, pad, 1)

    def avg_pool2d(self, x, face, stride, pad):
        return self._avg_pool(x, face, stride, pad, 2)

    def avg_pool3d(self, x, face, stride, pad):
        return self._avg_pool(x, face, stride, pad, 3)
