from ....base.core.map.sign import BaseSignAPI


class PyTorchSignAPI(BaseSignAPI):
    def __init__(self):
        BaseSignAPI.__init__(self)

    def abs(self, x):
        return x.abs()

    def neg(self, x):
        return x.neg()

    def sign(self, x):
        return x.sign()
