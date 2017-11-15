from ....base.layer.dot import BaseDotAPI
from .conv import PyTorchConvAPI
from .dense import PyTorchDenseAPI


class PyTorchDotAPI(BaseDotAPI, PyTorchConvAPI, PyTorchDenseAPI):
    def __init__(self):
        BaseDotAPI.__init__(self)
        PyTorchConvAPI.__init__(self)
        PyTorchDenseAPI.__init__(self)
