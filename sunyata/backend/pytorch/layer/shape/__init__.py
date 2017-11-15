from ....base.layer.shape import BaseShapeAPI
from .pad import PyTorchPadAPI


class PyTorchShapeAPI(BaseShapeAPI, PyTorchPadAPI):
    def __init__(self):
        BaseShapeAPI.__init__(self)
        PyTorchPadAPI.__init__(self)
