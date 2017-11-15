from chainer import functions as F

from ....base.layer.dot.dense import BaseDenseAPI


class ChainerDenseAPI(BaseDenseAPI):
    def __init__(self):
        BaseDenseAPI.__init__(self)

    def dense(self, x, kernel, bias):
        return F.connection.linear.linear(x, kernel, bias)
