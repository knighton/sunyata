from ....base.core.map.hyperbolic import BaseHyperbolicAPI


class PyTorchHyperbolicAPI(BaseHyperbolicAPI):
    def __init__(self):
        BaseHyperbolicAPI.__init__(self)

    def sinh(self, x):
        return x.sinh()

    def cosh(self, x):
        return x.cosh()

    def tanh(self, x):
        return x.tanh()
