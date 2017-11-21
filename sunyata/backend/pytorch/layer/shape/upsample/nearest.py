from torch.autograd import Variable
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

    def _nearest_upsample(self, x, scale):
        scale = self.to_shape(scale, self.ndim(x) - 2)
        scale_uniq = list(set(scale))
        if scale_uniq == [1]:
            pass
        elif len(scale_uniq) == 1 and isinstance(x, Variable):
            x = F.upsample(x, None, scale[0], 'nearest')
        else:
            x = self.repeat(x, (1, 1) + scale)
        return x

    def nearest_upsample(self, x, scale):
        ndim = self.ndim(x) - 2
        return self._ndim2nearest_upsample[ndim](x, scale)

    def nearest_upsample1d(self, x, scale):
        x = x.unsqueeze(2)
        x = self.upsample_nearest2d(x, scale)
        return x.squeeze(2)

    def nearest_upsample2d(self, x, scale):
        assert self.ndim(x) == 4
        return self._nearest_upsample(x, scale)

    def nearest_upsample3d(self, x, scale):
        assert self.ndim(x) == 5
        return self._nearest_upsample(x, scale)
