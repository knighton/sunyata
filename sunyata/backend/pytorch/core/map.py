import numpy as np

from ...base.core.map import BaseMapAPI


class PyTorchMapAPI(BaseMapAPI):
    def __init__(self):
        BaseMapAPI.__init__(self)

    def abs(self, x):
        return x.abs()

    def neg(self, x):
        return x.neg()

    def sign(self, x):
        return x.sign()

    def clip(self, x, min=-np.inf, max=np.inf):
        return x.clamp(min, max)

    def log(self, x):
        return x.log()

    def pow(self, x, a):
        return x ** a
