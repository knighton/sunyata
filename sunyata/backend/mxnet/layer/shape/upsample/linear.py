import mxnet as mx

from .....base.layer.shape.upsample.linear import BaseLinearUpsampleAPI


class MXNetLinearUpsampleAPI(BaseLinearUpsampleAPI):
    def __init__(self):
        BaseLinearUpsampleAPI.__init__(self)
        self._ndim2linear_upsample = {
            1: self.linear_upsample1d,
            2: self.linear_upsample2d,
            3: self.linear_upsample3d,
        }

    def linear_upsample(self, x, scale):
        ndim = self.ndim(x) - 2
        return self._ndim2linear_upsample[ndim]

    def linear_upsample1d(self, x, scale):
        scale = self.to_shape(scale, 1)
        scale = (1,) + scale
        x = x.unsqueeze(2)
        x = self.upsample_linear2d(x, scale)
        return x.squeeze(2)

    def linear_upsample2d(self, x, scale):
        assert self.ndim(x) == 4
        scale = self.to_shape(scale, 2)
        if 1 < len(set(scale)):
            raise NotImplementedError
        elif 1 < scale[0]:
            x = mx.nd.UpSampling(x, scale=scale[0], sample_type='bilinear')
        return x

    def linear_upsample3d(self, x, scale):
        raise NotImplementedError
