from chainer import functions as F

from ...base.core.linalg import BaseLinearAlgebraAPI


class ChainerLinearAlgebraAPI(BaseLinearAlgebraAPI):
    def matmul(self, a, b):
        return F.matmul(a, b)
