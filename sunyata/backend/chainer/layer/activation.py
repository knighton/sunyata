from chainer import functions as F

from ...base.layer.activation import BaseActivationAPI


class ChainerActivationAPI(BaseActivationAPI):
    def elu(self, x, alpha=1):
        return F.elu(x, alpha)

    def hard_sigmoid(self, x):
        return F.hard_sigmoid(x)

    def log_softmax(self, x):
        return F.log_softmax(x)

    def selu(self, x):
        return F.selu(x)

    def sigmoid(self, x):
        return F.sigmoid(x)

    def softmax(self, x):
        return F.softmax(x)
