from chainer import functions as F

from ....base.core.map.sign import BaseSignAPI


class ChainerSignAPI(BaseSignAPI):
    def __init__(self):
        BaseSignAPI.__init__(self)

    def abs(self, x):
        return F.math.basic_math.absolute(x)

    def neg(self, x):
        return F.math.basic_math.neg(x)
