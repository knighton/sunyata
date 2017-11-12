import numpy as np

from ..base import APIBase


class BaseMapAPI(APIBase):
    def abs(self, x):
        return self.maximum(x, -1 * x)

    def neg(self, x):
        return -1 * x

    def sign(self, x):
        return self.less(0, x) * 2 - 1

    def clip(self, x, min=-np.inf, max=np.inf):
        raise NotImplementedError

    def log(self, x):
        raise NotImplementedError

    def pow(self, x, a):
        raise NotImplementedError

    def square(self, x):
        return self.pow(x, 2)

    def sqrt(self, x):
        return self.pow(x, 0.5)
