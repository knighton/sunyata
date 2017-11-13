from ....base.core.map.cumulative import BaseCumulativeAPI


class PyTorchCumulativeAPI(BaseCumulativeAPI):
    def __init__(self):
        BaseCumulativeAPI.__init__(self)

    def cumsum(self, x, axis):
        return x.cumsum(axis)

    def cumprod(self, x, axis):
        return x.cumprod(axis)
