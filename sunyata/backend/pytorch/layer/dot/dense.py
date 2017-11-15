from ....base.layer.dot.dense import BaseDenseAPI


class PyTorchDenseAPI(BaseDenseAPI):
    def dense(self, x, kernel, bias):
        return x.mm(kernel) + bias
