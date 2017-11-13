from ....base.core.map.cumulative import BaseCumulativeAPI


class ChainerCumulativeAPI(BaseCumulativeAPI):
    def __init__(self):
        BaseCumulativeAPI.__init__(self)

    def cumsum(self, x, axis):
        raise NotImplementedError

    def cumprod(self, x, axis):
        raise NotImplementedError
