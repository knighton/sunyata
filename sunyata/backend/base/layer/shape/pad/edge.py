from ....base import APIMixin


class BaseEdgePadAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def edge_pad(self, x, pad):
        raise NotImplementedError

    def edge_pad1d(self, x, pad):
        raise NotImplementedError

    def edge_pad2d(self, x, pad):
        raise NotImplementedError

    def edge_pad3d(self, x, pad):
        raise NotImplementedError
