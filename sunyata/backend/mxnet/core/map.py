import mxnet as mx
import numpy as np

from ...base.core.map import BaseMapAPI


class MXNetMapAPI(BaseMapAPI):
    def __init__(self):
        BaseMapAPI.__init__(self)

    def abs(self, x):
        return mx.nd.abs(x)

    def neg(self, x):
        return mx.nd.neg(x)

    def sign(self, x):
        return mx.nd.sign(x)

    def clip(self, x, min=-np.inf, max=np.inf):
        return mx.nd.clip(x, min, max)

    def log(self, x):
        return mx.nd.log(x)

    def pow(self, x, a):
        return mx.nd.power(x, a)
