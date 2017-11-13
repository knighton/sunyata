from ....base.core.map.power import BasePowerAPI


class PyTorchPowerAPI(BasePowerAPI):
    def __init__(self):
        BasePowerAPI.__init__(self)

    def pow(self, x, a):
        return x.pow(a)

    def rsqrt(self, x):
        return x.rsqrt()

    def sqrt(self, x):
        return x.sqrt()
