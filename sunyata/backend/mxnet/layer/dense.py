import mxnet as mx

from ...base.layer.dense import BaseDenseAPI


class MXNetDenseAPI(BaseDenseAPI):
    def dense(self, x, kernel, bias):
        return mx.nd.dot(x, kernel) + bias
