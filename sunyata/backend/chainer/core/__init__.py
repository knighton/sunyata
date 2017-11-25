from ...base.core import BaseCoreAPI
from .device_dtype import ChainerDeviceDataTypeAPI
from .linalg import ChainerLinearAlgebraAPI
from .logic import ChainerLogicAPI
from .map import ChainerMapAPI
from .shape import ChainerShapeAPI
from .reduce import ChainerReduceAPI
from .variable import ChainerVariableAPI


class ChainerCoreAPI(BaseCoreAPI, ChainerDeviceDataTypeAPI,
                     ChainerLinearAlgebraAPI, ChainerLogicAPI, ChainerMapAPI,
                     ChainerReduceAPI, ChainerShapeAPI, ChainerVariableAPI):
    def __init__(self):
        BaseCoreAPI.__init__(self)
        ChainerDeviceDataTypeAPI.__init__(self)
        ChainerLinearAlgebraAPI.__init__(self)
        ChainerLogicAPI.__init__(self)
        ChainerMapAPI.__init__(self)
        ChainerReduceAPI.__init__(self)
        ChainerShapeAPI.__init__(self)
        ChainerVariableAPI.__init__(self)
