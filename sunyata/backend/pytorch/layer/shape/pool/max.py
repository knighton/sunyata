from torch.nn import functional as F

from .....base.layer.shape.pool.max import BaseMaxPoolAPI


class PyTorchMaxPoolAPI(BaseMaxPoolAPI):
    def __init__(self):
        BaseMaxPoolAPI.__init__(self)
        self._ndim2max_pool = {
            1: self.max_pool1d,
            2: self.max_pool2d,
            3: self.max_pool3d,
        }

    def max_pool(self, x, face, stride, pad):
        ndim = self.ndim(x) - 2
        return self._ndim2max_pool[ndim](x, face, stride, pad)

    def max_pool1d(self, x, face, stride, pad):
        face = self.to_one(face)
        stride = self.to_one(stride)
        pad = self.to_one(pad)
        return F.max_pool1d(x, face, stride, pad)

    def max_pool2d(self, x, face, stride, pad):
        return F.max_pool2d(x, face, stride, pad)

    def max_pool3d(self, x, face, stride, pad):
        return F.max_poold3d(x, face, stride, pad)
