from chainer import functions as F

from ...base.layer.activation import BaseActivationAPI


class ChainerActivationAPI(BaseActivationAPI):
    def softmax(self, x):
        return F.softmax(x)
