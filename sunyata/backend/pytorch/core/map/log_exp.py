from ....base.core.map.log_exp import BaseLogExpAPI


class PyTorchLogExpAPI(BaseLogExpAPI):
    def __init__(self):
        BaseLogExpAPI.__init__(self)

    def exp(self, x):
        return x.exp()

    def expm1(self, x):
        return x.expm1()

    def log(self, x):
        return x.log()

    def log1p(self, x):
        return x.log1p()
