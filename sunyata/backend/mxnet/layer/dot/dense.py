import mxnet as mx

from ....base.layer.dot.dense import BaseDenseAPI


class MXNetDenseAPI(BaseDenseAPI):
    def __init__(self):
        BaseDenseAPI.__init__(self)

    def dense(self, x, kernel, bias):
        return mx.nd.FullyConnected(x, kernel, bias, bias.shape[0])
