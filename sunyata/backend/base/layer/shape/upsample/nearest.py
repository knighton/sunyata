from ....base import APIMixin


class BaseNearestUpsampleAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def nearest_upsample(self, x, scale):
        raise NotImplementedError

    def nearest_upsample1d(self, x, scale):
        raise NotImplementedError

    def nearest_upsample2d(self, x, scale):
        raise NotImplementedError

    def nearest_upsample3d(self, x, scale):
        raise NotImplementedError
