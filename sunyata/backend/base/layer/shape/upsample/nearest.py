from ....base import APIMixin


class BaseNearestUpsampleAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def nearest_upsample(self, x, scale):
        ndim = self.ndim(x) - 2
        assert ndim in {1, 2, 3}
        scale = self.to_shape(scale, ndim)
        return self.repeat(x, (1, 1) + scale)

    def nearest_upsample1d(self, x, scale):
        assert self.ndim(x) == 3
        scale = self.to_shape(scale, 1)
        return self.repeat(x, (1, 1) + scale)

    def nearest_upsample2d(self, x, scale):
        assert self.ndim(x) == 4
        scale = self.to_shape(scale, 2)
        return self.repeat(x, (1, 1) + scale)

    def nearest_upsample3d(self, x, scale):
        assert self.ndim(x) == 5
        scale = self.to_shape(scale, 3)
        return self.repeat(x, (1, 1) + scale)
