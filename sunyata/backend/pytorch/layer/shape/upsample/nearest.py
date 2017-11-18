from torch.nn import functional as F

from .....base.layer.shape.upsample.nearest import BaseNearestUpsampleAPI


class PyTorchNearestUpsampleAPI(BaseNearestUpsampleAPI):
    def __init__(self):
        BaseNearestUpsampleAPI.__init__(self)
        self._ndim2nearest_upsample = {
            1: self.nearest_upsample1d,
            2: self.nearest_upsample2d,
            3: self.nearest_upsample3d,
        }

    def nearest_upsample(self, x, scale):
        ndim = self.ndim(x) - 2
        return self._ndim2nearest_upsample[ndim](x, scale)

    def nearest_upsample1d(self, x, scale):
        x = x.unsqueeze(2)
        x = self.upsample_nearest2d(x, scale)
        return x.squeeze(2)

    def nearest_upsample2d(self, x, scale):
        assert self.ndim(x) == 4
        return F.upsample(x, None, scale, 'nearest')

    def nearest_upsample3d(self, x, scale):
        assert self.ndim(x) == 5
        return F.upsample(x, None, scale, 'nearest')
