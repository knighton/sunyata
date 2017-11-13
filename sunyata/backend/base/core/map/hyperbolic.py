from ...base import APIMixin


class BaseHyperbolicAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def sinh(self, x):
        return (self.exp(x) - self.exp(-x)) / 2

    def cosh(self, x):
        return (self.exp(x) + self.exp(-x)) / 2

    def tanh(self, x):
        e_x = self.exp(x)
        e_nx = self.exp(-x)
        return (e_x - e_nx) / (e_x + e_nx)

    def arcsinh(self, x):
        return self.log(x + self.sqrt(self.square(x) + 1))

    def arccosh(self, x):
        return self.log(x + self.sqrt(self.square(x) - 1))

    def arctanh(self, x):
        return 0.5 * self.log((1 + x) / (1 - x))
