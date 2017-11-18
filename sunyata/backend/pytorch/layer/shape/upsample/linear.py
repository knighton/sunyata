from torch.nn import functional as F

from .....base.layer.shape.upsample.linear import BaseLinearUpsampleAPI


class PyTorchLinearUpsampleAPI(BaseLinearUpsampleAPI):
    def __init__(self):
        BaseLinearUpsampleAPI.__init__(self)
        self._ndim2linear_upsample = {
            1: self.linear_upsample1d,
            2: self.linear_upsample2d,
            3: self.linear_upsample3d,
        }

    def linear_upsample(self, x, scale):
        ndim = self.ndim(x) - 2
        return self._ndim2linear_upsample[ndim](x, scale)

    def linear_upsample1d(self, x, scale):
        x = x.unsqueeze(2)
        x = self.upsample_nearest2d(x, scale)
        return x.squeeze(2)

    def linear_upsample2d(self, x, scale):
        return F.upsample(x, None, scale, 'bilinear')

    def linear_upsample3d(self, x, scale):
        return F.upsample(x, None, scale, 'trilinear')
