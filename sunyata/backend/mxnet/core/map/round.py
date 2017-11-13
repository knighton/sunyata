import mxnet as mx

from ....base.core.map.round import BaseRoundAPI


class MXNetRoundAPI(BaseRoundAPI):
    def __init__(self):
        BaseRoundAPI.__init__(self)

    def ceil(self, x):
        return mx.nd.ceil(x)

    def floor(self, x):
        return mx.nd.floor(x)

    def round(self, x):
        return mx.nd.round(x)

    def trunc(self, x):
        return mx.nd.trunc(x)
