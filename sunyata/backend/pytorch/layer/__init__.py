from ...base.layer import BaseLayerAPI
from .activation import PyTorchActivationAPI
from .dot import PyTorchDotAPI
from .embed import PyTorchEmbedAPI
from .shape import PyTorchShapeAPI


class PyTorchLayerAPI(BaseLayerAPI, PyTorchActivationAPI, PyTorchDotAPI,
                      PyTorchEmbedAPI, PyTorchShapeAPI):
    def __init__(self):
        BaseLayerAPI.__init__(self)
        PyTorchActivationAPI.__init__(self)
        PyTorchDotAPI.__init__(self)
        PyTorchEmbedAPI.__init__(self)
        PyTorchShapeAPI.__init__(self)
