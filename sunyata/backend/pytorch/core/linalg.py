from ...base.core.linalg import BaseLinearAlgebraAPI


class PyTorchLinearAlgebraAPI(BaseLinearAlgebraAPI):
    def matmul(self, a, b):
        return a.matmul(b)
