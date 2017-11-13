from chainer import functions as F

from ...base.layer.dense import BaseDenseAPI


class ChainerDenseAPI(BaseDenseAPI):
    def dense(self, x, kernel, bias):
        return F.connection.linear.linear(x, kernel, bias)
