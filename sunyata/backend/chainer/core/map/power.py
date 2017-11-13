from chainer import functions as F

from ....base.core.map.power import BasePowerAPI


class ChainerPowerAPI(BasePowerAPI):
    def __init__(self):
        BasePowerAPI.__init__(self)

    def pow(self, x, a):
        return F.math.basic_math.pow(x, a)

    def rsqrt(self, x):
        return F.rsqrt(x)

    def sqrt(self, x):
        return F.sqrt(x)

    def square(self, x):
        return F.square(x)
