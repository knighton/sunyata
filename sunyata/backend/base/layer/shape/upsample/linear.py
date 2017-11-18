from ....base import APIMixin


class BaseLinearUpsampleAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def linear_upsample(self, x, scale):
        raise NotImplementedError

    def linear_upsample1d(self, x, scale):
        raise NotImplementedError

    def linear_upsample2d(self, x, scale):
        raise NotImplementedError

    def linear_upsample3d(self, x, scale):
        raise NotImplementedError
