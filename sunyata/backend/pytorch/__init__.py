from ..base import BaseBackend
from .core import PyTorchCoreAPI
from .layer import PyTorchLayerAPI


class PyTorchBackend(BaseBackend, PyTorchCoreAPI, PyTorchLayerAPI):
    def __init__(self):
        BaseBackend.__init__(self)
        PyTorchCoreAPI.__init__(self)
        PyTorchLayerAPI.__init__(self)
        self.name = 'pytorch'
