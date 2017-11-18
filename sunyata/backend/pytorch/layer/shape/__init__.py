from ....base.layer.shape import BaseShapeAPI
from .pad import PyTorchPadAPI
from .pool import PyTorchPoolAPI


class PyTorchShapeAPI(BaseShapeAPI, PyTorchPadAPI, PyTorchPoolAPI):
    def __init__(self):
        BaseShapeAPI.__init__(self)
        PyTorchPadAPI.__init__(self)
        PyTorchPoolAPI.__init__(self)
