import mxnet as mx

from ...base.core.linalg import BaseLinearAlgebraAPI


class MXNetLinearAlgebraAPI(BaseLinearAlgebraAPI):
    def matmul(self, a, b):
        return mx.nd.dot(a, b)
