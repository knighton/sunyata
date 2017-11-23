from ....base.layer.dot.dense import BaseDenseAPI


class PyTorchDenseAPI(BaseDenseAPI):
    def __init__(self):
        BaseDenseAPI.__init__(self)

    def dense(self, x, kernel, bias):
        x = x.mm(kernel)
        if bias is not None:
            x += bias
        return x
