from torch.nn import functional as F

from ...base.layer.activation import BaseActivationAPI


class PyTorchActivationAPI(BaseActivationAPI):
    def __init__(self):
        BaseActivationAPI.__init__(self)

    def softmax(self, x):
        x_shape = x.size()
        tran = x.transpose(1, len(x_shape) - 1)
        tran_shape = tran.size()
        tran_2d = tran.contiguous().view(-1, tran_shape[-1])
        tran_2d = F.softmax(tran_2d)
        tran = tran_2d.view(*tran_shape)
        return tran.transpose(1, len(x_shape) - 1)
