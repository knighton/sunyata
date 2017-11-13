from ..base import BaseBackend
from .core import MXNetCoreAPI
from .layer import MXNetLayerAPI


class MXNetBackend(BaseBackend, MXNetCoreAPI, MXNetLayerAPI):
    def __init__(self):
        BaseBackend.__init__(self)
        MXNetCoreAPI.__init__(self)
        MXNetLayerAPI.__init__(self)
        self.name = 'mxnet'
