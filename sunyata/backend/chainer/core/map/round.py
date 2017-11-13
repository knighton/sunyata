from chainer import functions as F

from ....base.core.map.round import BaseRoundAPI


class ChainerRoundAPI(BaseRoundAPI):
    def __init__(self):
        BaseRoundAPI.__init__(self)

    def ceil(self, x):
        return F.ceil(x)

    def floor(self, x):
        return F.floor(x)
