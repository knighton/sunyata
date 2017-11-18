from ....base import APIMixin


class BaseAvgPoolAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def avg_pool(self, x, face, stride, pad):
        raise NotImplementedError

    def avg_pool1d(self, x, face, stride, pad):
        raise NotImplementedError

    def avg_pool2d(self, x, face, stride, pad):
        raise NotImplementedError

    def avg_pool3d(self, x, face, stride, pad):
        raise NotImplementedError
