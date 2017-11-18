from torch.nn import functional as F

from .....base.layer.shape.pool.avg import BaseAvgPoolAPI


class PyTorchAvgPoolAPI(BaseAvgPoolAPI):
    def __init__(self):
        BaseAvgPoolAPI.__init__(self)
        self._ndim2avg_pool = {
            1: self.avg_pool1d,
            2: self.avg_pool2d,
            3: self.avg_pool3d,
        }

    def avg_pool(self, x, face, stride, pad):
        ndim = self.ndim(x) - 2
        return self._ndim2avg_pool[ndim](x, face, stride, pad)

    def avg_pool1d(self, x, face, stride, pad):
        face = self.to_one(face)
        stride = self.to_one(stride)
        pad = self.to_one(pad)
        return F.avg_pool1d(x, face, stride, pad)

    def avg_pool2d(self, x, face, stride, pad):
        return F.avg_pool2d(x, face, stride, pad)

    def avg_pool3d(self, x, face, stride, pad):
        x = self.constant_pad3d(x, pad, 0)
        return F.avg_poold3d(x, face, stride)
