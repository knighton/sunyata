from chainer import functions as F

from ....base.core.map.hyperbolic import BaseHyperbolicAPI


class ChainerHyperbolicAPI(BaseHyperbolicAPI):
    def __init__(self):
        BaseHyperbolicAPI.__init__(self)

    def sinh(self, x):
        return F.sinh(x)

    def cosh(self, x):
        return F.cosh(x)

    def tanh(self, x):
        return F.tanh(x)
