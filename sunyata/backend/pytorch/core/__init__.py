from ...base.core import BaseCoreAPI
from .device_dtype import PyTorchDeviceDataTypeAPI
from .linalg import PyTorchLinearAlgebraAPI
from .logic import PyTorchLogicAPI
from .map import PyTorchMapAPI
from .shape import PyTorchShapeAPI
from .reduce import PyTorchReduceAPI
from .tensor import PyTorchTensorAPI
from .variable import PyTorchVariableAPI


class PyTorchCoreAPI(BaseCoreAPI, PyTorchDeviceDataTypeAPI,
                     PyTorchLinearAlgebraAPI, PyTorchLogicAPI, PyTorchMapAPI,
                     PyTorchReduceAPI, PyTorchShapeAPI, PyTorchTensorAPI,
                     PyTorchVariableAPI):
    def __init__(self):
        BaseCoreAPI.__init__(self)
        PyTorchDeviceDataTypeAPI.__init__(self)
        PyTorchLinearAlgebraAPI.__init__(self)
        PyTorchLogicAPI.__init__(self)
        PyTorchMapAPI.__init__(self)
        PyTorchReduceAPI.__init__(self)
        PyTorchShapeAPI.__init__(self)
        PyTorchTensorAPI.__init__(self)
        PyTorchVariableAPI.__init__(self)
