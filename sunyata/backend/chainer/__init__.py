from ..base import BaseBackend
from .core import ChainerCoreAPI
from .layer import ChainerLayerAPI


class ChainerBackend(BaseBackend, ChainerCoreAPI, ChainerLayerAPI):
    def __init__(self):
        BaseBackend.__init__(self)
        ChainerCoreAPI.__init__(self)
        ChainerLayerAPI.__init__(self)
        self.name = 'chainer'
