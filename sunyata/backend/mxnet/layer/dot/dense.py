import mxnet as mx

from ....base.layer.dot.dense import BaseDenseAPI


class MXNetDenseAPI(BaseDenseAPI):
    def __init__(self):
        BaseDenseAPI.__init__(self)

    def dense(self, x, kernel, bias):
        x = mx.nd.dot(x, kernel)
        if bias is not None:
            x = x + bias
        return x
