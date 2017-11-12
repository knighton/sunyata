from .device_dtype import PyTorchDeviceDataTypeAPI
from .logic import PyTorchLogicAPI
from .map import PyTorchMapAPI
from .shape import PyTorchShapeAPI
from .reduce import PyTorchReduceAPI
from .variable import PyTorchVariableAPI


class PyTorchCoreAPI(PyTorchDeviceDataTypeAPI, PyTorchLogicAPI, PyTorchMapAPI,
                     PyTorchReduceAPI, PyTorchShapeAPI, PyTorchVariableAPI):
    def __init__(self):
        PyTorchDeviceDataTypeAPI.__init__(self)
        PyTorchLogicAPI.__init__(self)
        PyTorchMapAPI.__init__(self)
        PyTorchReduceAPI.__init__(self)
        PyTorchShapeAPI.__init__(self)
        PyTorchVariableAPI.__init__(self)
