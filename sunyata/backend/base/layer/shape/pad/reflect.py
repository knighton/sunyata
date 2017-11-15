from .....base import APIMixin


class BaseReflectPadAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def reflect_pad(self, x, pad):
        raise NotImplementedError

    def reflect_pad1d(self, x, pad):
        raise NotImplementedError

    def reflect_pad2d(self, x, pad):
        raise NotImplementedError

    def reflect_pad3d(self, x, pad):
        raise NotImplementedError
