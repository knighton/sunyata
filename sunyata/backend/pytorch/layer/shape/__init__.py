from ....base.layer.shape import BaseShapeAPI
from .pad import PyTorchPadAPI
from .pool import PyTorchPoolAPI
from .upsample import PyTorchUpsampleAPI


class PyTorchShapeAPI(BaseShapeAPI, PyTorchPadAPI, PyTorchPoolAPI,
                      PyTorchUpsampleAPI):
    def __init__(self):
        BaseShapeAPI.__init__(self)
        PyTorchPadAPI.__init__(self)
        PyTorchPoolAPI.__init__(self)
        PyTorchUpsampleAPI.__init__(self)
