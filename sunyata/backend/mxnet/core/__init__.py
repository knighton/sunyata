from ...base.core import BaseCoreAPI
from .device_dtype import MXNetDeviceDataTypeAPI
from .logic import MXNetLogicAPI
from .map import MXNetMapAPI
from .reduce import MXNetReduceAPI
from .shape import MXNetShapeAPI
from .tensor import MXNetTensorAPI
from .variable import MXNetVariableAPI


class MXNetCoreAPI(BaseCoreAPI, MXNetDeviceDataTypeAPI, MXNetLogicAPI,
                   MXNetMapAPI, MXNetReduceAPI, MXNetShapeAPI, MXNetTensorAPI,
                   MXNetVariableAPI):
    def __init__(self):
        BaseCoreAPI.__init__(self)
        MXNetDeviceDataTypeAPI.__init__(self)
        MXNetLogicAPI.__init__(self)
        MXNetMapAPI.__init__(self)
        MXNetReduceAPI.__init__(self)
        MXNetShapeAPI.__init__(self)
        MXNetTensorAPI.__init__(self)
        MXNetVariableAPI.__init__(self)
