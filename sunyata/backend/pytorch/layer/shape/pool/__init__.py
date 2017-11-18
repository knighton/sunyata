from .....base.layer.shape.pool import BasePoolAPI
from .avg import PyTorchAvgPoolAPI
from .max import PyTorchMaxPoolAPI


class PyTorchPoolAPI(BasePoolAPI, PyTorchAvgPoolAPI, PyTorchMaxPoolAPI):
    def __init__(self):
        BasePoolAPI.__init__(self)
        PyTorchAvgPoolAPI.__init__(self)
        PyTorchMaxPoolAPI.__init__(self)
