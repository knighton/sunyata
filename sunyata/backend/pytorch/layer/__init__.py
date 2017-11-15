from ...base.layer import BaseLayerAPI
from .activation import PyTorchActivationAPI
from .dot import PyTorchDotAPI


class PyTorchLayerAPI(BaseLayerAPI, PyTorchActivationAPI, PyTorchDotAPI):
    def __init__(self):
        BaseLayerAPI.__init__(self)
        PyTorchActivationAPI.__init__(self)
        PyTorchDotAPI.__init__(self)
