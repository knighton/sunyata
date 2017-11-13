from ...base import APIMixin


class BaseCumulativeAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def cumsum(self, x, axis):
        raise NotImplementedError

    def cumprod(self, x, axis):
        raise NotImplementedError
