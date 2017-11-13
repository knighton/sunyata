import mxnet as mx

from ...base.layer.activation import BaseActivationAPI


class MXNetActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        return mx.nd.softmax(x)
