from torch.autograd import Variable
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

    def _linear_upsample(self, x, scale, method):
        scale = self.to_shape(scale, self.ndim(x) - 2)
        if 1 < len(set(scale)):
            raise NotImplementedError
        if isinstance(x, Variable):
            x = F.upsample(x, None, scale[0], method)
        else:
            x = F.upsample(Variable(x), None, scale[0], method).data
        return x

    def linear_upsample(self, x, scale):
        ndim = self.ndim(x) - 2
        return self._ndim2linear_upsample[ndim](x, scale)

    def linear_upsample1d(self, x, scale):
        scale = self.to_shape(scale, 1)
        scale = (1,) + scale
        x = x.unsqueeze(2)
        x = self.upsample_nearest2d(x, scale)
        return x.squeeze(2)

    def linear_upsample2d(self, x, scale):
        assert self.ndim(x) == 4
        return self._linear_upsample(x, scale, 'bilinear')

    def linear_upsample3d(self, x, scale):
        assert self.ndim(x) == 5
        return self._linear_upsample(x, scale, 'trilinear')
