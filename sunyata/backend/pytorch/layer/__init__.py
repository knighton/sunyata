from .activation import PyTorchActivationAPI
from .dense import PyTorchDenseAPI


class PyTorchLayerAPI(PyTorchActivationAPI, PyTorchDenseAPI):
    def __init__(self):
        PyTorchActivationAPI.__init__(self)
        PyTorchDenseAPI.__init__(self)
