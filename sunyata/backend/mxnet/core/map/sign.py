import mxnet as mx

from ....base.core.map.sign import BaseSignAPI


class MXNetSignAPI(BaseSignAPI):
    def __init__(self):
        BaseSignAPI.__init__(self)

    def abs(self, x):
        return mx.nd.abs(x)

    def neg(self, x):
        return mx.nd.negative(x)

    def sign(self, x):
        return mx.nd.sign(x)
