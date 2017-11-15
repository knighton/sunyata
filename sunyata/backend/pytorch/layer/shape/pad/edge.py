from torch.nn import functional as F

from .....base.layer.shape.pad.edge import BaseEdgePadAPI


class PyTorchEdgePadAPI(BaseEdgePadAPI):
    def __init__(self):
        BaseEdgePadAPI.__init__(self)
        self._ndim2edge_pad = {
            1: self.edge_pad1d,
            2: self.edge_pad2d,
            3: self.edge_pad3d,
        }

    def edge_pad(self, x, pad):
        ndim = x.dim() - 2
        return self._ndim2edge_pad[ndim](x, pad)

    def edge_pad1d(self, x, pad):
        x = x.unsqueeze(2)
        (left, right), = self.unpack_pad(pad, 1)
        pad = 0, 0, left, right
        x = self.edge_pad2d(x, pad)
        return x.squeeze(2)

    def edge_pad2d(self, x, pad):
        (top, bottom), (left, right) = self.unpack_pad(pad, 2)
        pad = top, bottom, left, right
        return F.pad(x, pad, 'replicate')

    def edge_pad3d(self, x, pad):
        (front, back), (top, bottom), (left, right) = self.unpack_pad(pad, 3)
        pad = front, back, top, bottom, left, right
        return F.pad(x, pad, 'replicate')
