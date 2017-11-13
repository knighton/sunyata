import mxnet as mx

from ....base.core.map.log_exp import BaseLogExpAPI


class MXNetLogExpAPI(BaseLogExpAPI):
    def __init__(self):
        BaseLogExpAPI.__init__(self)

    def exp(self, x):
        return mx.nd.exp(x)

    def expm1(self, x):
        return mx.nd.expm1(x)

    def log(self, x):
        return mx.nd.log(x)

    def log2(self, x):
        return mx.nd.log2(x)

    def log10(self, x):
        return mx.nd.log10(x)

    def log1p(self, x):
        return mx.nd.log1p(x)
