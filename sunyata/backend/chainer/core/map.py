from chainer import functions as F
import numpy as np

from ...base.core.map import BaseMapAPI


class ChainerMapAPI(BaseMapAPI):
    def __init__(self):
        BaseMapAPI.__init__(self)

    def abs(self, x):
        return F.absolute(x)

    def clip(self, x, min=-np.inf, max=np.inf):
        return F.clip(x, float(min), float(max))

    def log(self, x):
        return F.math.exponential.log(x)

    def pow(self, x, a):
        return F.math.basic_math.pow(x, a)
