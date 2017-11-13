from ...base import APIMixin


class BasePowerAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def pow(self, x, a):
        raise NotImplementedError

    def rsqrt(self, x):
        return 1 / self.sqrt(x)

    def sqrt(self, x):
        return self.pow(x, 0.5)

    def square(self, x):
        return self.pow(x, 2)
