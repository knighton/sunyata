from torch.nn import functional as F

from ...base.layer.activation import BaseActivationAPI


class PyTorchActivationAPI(BaseActivationAPI):
    def __init__(self):
        BaseActivationAPI.__init__(self)

    def elu(self, x, alpha=1):
        return F.elu(x, alpha)

    def leaky_relu(self, x, alpha=0.1):
        return F.leaky_relu(x, alpha)

    def log_sigmoid(self, x):
        return F.logsigmoid(x)

    def log_softmax(self, x):
        return F.log_softmax(x)

    def selu(self, x):
        return F.selu(x)

    def sigmoid(self, x):
        return F.sigmoid(x)

    def softmax(self, x):
        x_shape = x.size()
        tp = x.transpose(1, len(x_shape) - 1)
        tp_shape = tp.size()
        input_2d = tp.contiguous().view(-1, tp_shape[-1])
        _2d = F.softmax(input_2d)
        nd = _2d.view(*tp_shape)
        return nd.transpose(1, len(x_shape) - 1)

    def softmin(self, x):
        x_shape = x.size()
        tp = x.transpose(1, len(x_shape) - 1)
        tp_shape = tp.size()
        input_2d = tp.contiguous().view(-1, tp_shape[-1])
        _2d = F.softmin(input_2d)
        nd = _2d.view(*tp_shape)
        return nd.transpose(1, len(x_shape) - 1)

    def softplus(self, x, beta=1, threshold=20):
        return F.softplus(x, beta, threshold)

    def softshrink(self, x, lambd=0.5):
        return F.softshrink(x, lambd)

    def softsign(self, x):
        return F.softsign(x)

    def tanh_shrink(self, x):
        return F.tanhshrink(x)
