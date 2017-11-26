from torch.nn import functional as F

from ....base.layer.dot.dense import BaseDenseAPI


class PyTorchDenseAPI(BaseDenseAPI):
    def __init__(self):
        BaseDenseAPI.__init__(self)

    def dense(self, x, kernel, bias):
        return F.linear(x, kernel, bias)
