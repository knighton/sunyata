import mxnet as mx

from ....base.core.map.power import BasePowerAPI


class MXNetPowerAPI(BasePowerAPI):
    def __init__(self):
        BasePowerAPI.__init__(self)

    def pow(self, x, a):
        return mx.nd.power(x, a)

    def rsqrt(self, x):
        return mx.nd.rsqrt(x)

    def sqrt(self, x):
        return mx.nd.sqrt(x)

    def square(self, x):
        return mx.nd.square(x)
