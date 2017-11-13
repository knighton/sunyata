from ....base.core.map.round import BaseRoundAPI


class PyTorchRoundAPI(BaseRoundAPI):
    def __init__(self):
        BaseRoundAPI.__init__(self)

    def ceil(self, x):
        return x.ceil()

    def floor(self, x):
        return x.floor()

    def round(self, x):
        return x.round()

    def trunc(self, x):
        return x.trunc()
