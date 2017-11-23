from chainer import functions as F

from ...base.layer.batch_norm import BaseBatchNormAPI


class ChainerBatchNormAPI(BaseBatchNormAPI):
    def __init__(self):
        BaseBatchNormAPI.__init__(self)

    def do_batch_norm(self, x, beta, gamma, mean, var):
        x, beta, gamma, mean, var = F.broadcast(x, beta, gamma, mean, var)
        return gamma * (x - mean) / self.sqrt(var + self.epsilon()) + beta
