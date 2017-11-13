import mxnet as mx

from ....base.core.map.hyperbolic import BaseHyperbolicAPI


class MXNetHyperbolicAPI(BaseHyperbolicAPI):
    def __init__(self):
        BaseHyperbolicAPI.__init__(self)

    def sinh(self, x):
        return mx.nd.sinh(x)

    def cosh(self, x):
        return mx.nd.cosh(x)

    def tanh(self, x):
        return mx.nd.tanh(x)

    def arcsinh(self, x):
        return mx.nd.arcsinh(x)

    def arccosh(self, x):
        return mx.nd.arccosh(x)

    def arctanh(self, x):
        return mx.nd.arctanh(x)
