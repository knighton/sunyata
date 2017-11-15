from torch.nn import functional as F

from .....base.layer.shape.pad.constant import BaseConstantPadAPI


class PyTorchConstantPadAPI(BaseConstantPadAPI):
    def __init__(self):
        BaseConstantPadAPI.__init__(self)
        self._ndim2constant_pad = {
            1: self.constant_pad1d,
            2: self.constant_pad2d,
            3: self.constant_pad3d,
        }

    def constant_pad(self, x, pad, value):
        ndim = x.dim() - 2
        return self._ndim2constant_pad[ndim](x, pad, value)

    def constant_pad1d(self, x, pad, value):
        x = x.unsqueeze(2)
        (left, right), = self.unpack_int_pad(pad, 1)
        pad = 0, 0, left, right
        x = self.constant_pad2d(x, pad, value)
        return x.squeeze(2)

    def constant_pad2d(self, x, pad, value):
        (top, bottom), (left, right) = self.unpack_int_pad(pad, 2)
        pad = top, bottom, left, right
        return F.pad(x, pad, 'constant', value)

    def constant_pad3d(self, x, pad, value):
        (front, back), (top, bottom), (left, right) = \
            self.unpack_int_pad(pad, 3)
        pad = front, back, top, bottom, left, right
        return F.pad(x, pad, 'constant', value)
