from .....base import APIMixin


class BaseConstantPadAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def constant_pad(self, x, pad, value):
        raise NotImplementedError

    def constant_pad1d(self, x, pad, value):
        raise NotImplementedError

    def constant_pad2d(self, x, pad, value):
        raise NotImplementedError

    def constant_pad3d(self, x, pad, value):
        raise NotImplementedError
