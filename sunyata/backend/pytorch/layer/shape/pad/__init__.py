from .....base.layer.shape.pad import BasePadAPI
from .constant import PyTorchConstantPadAPI
from .edge import PyTorchEdgePadAPI
from .reflect import PyTorchReflectPadAPI


class PyTorchPadAPI(BasePadAPI, PyTorchConstantPadAPI, PyTorchEdgePadAPI,
                    PyTorchReflectPadAPI):
    def __init__(self):
        BasePadAPI.__init__(self)
        PyTorchConstantPadAPI.__init__(self)
        PyTorchEdgePadAPI.__init__(self)
        PyTorchReflectPadAPI.__init__(self)
