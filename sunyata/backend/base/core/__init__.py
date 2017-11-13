from .device_dtype import BaseDeviceDataTypeAPI
from .epsilon import BaseEpsilonAPI
from .logic import BaseLogicAPI
from .map import BaseMapAPI
from .shape import BaseShapeAPI
from .reduce import BaseReduceAPI
from .tensor import BaseTensorAPI
from .variable import BaseVariableAPI


class BaseCoreAPI(BaseDeviceDataTypeAPI, BaseEpsilonAPI, BaseLogicAPI,
                  BaseMapAPI, BaseReduceAPI, BaseShapeAPI, BaseTensorAPI,
                  BaseVariableAPI):
    def __init__(self):
        BaseDeviceDataTypeAPI.__init__(self)
        BaseEpsilonAPI.__init__(self)
        BaseLogicAPI.__init__(self)
        BaseMapAPI.__init__(self)
        BaseReduceAPI.__init__(self)
        BaseShapeAPI.__init__(self)
        BaseTensorAPI.__init__(self)
        BaseVariableAPI.__init__(self)
