from ....base import APIMixin


class BaseMaxPoolAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def max_pool(self, x, face, stride, pad):
        raise NotImplementedError

    def max_pool1d(self, x, face, stride, pad):
        raise NotImplementedError

    def max_pool2d(self, x, face, stride, pad):
        raise NotImplementedError

    def max_pool3d(self, x, face, stride, pad):
        raise NotImplementedError
