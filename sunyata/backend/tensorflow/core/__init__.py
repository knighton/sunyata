from ...base.core import BaseCoreAPI
from .device_dtype import TensorFlowDeviceDataTypeAPI
from .logic import TensorFlowLogicAPI
from .map import TensorFlowMapAPI
from .reduce import TensorFlowReduceAPI
from .shape import TensorFlowShapeAPI
from .variable import TensorFlowVariableAPI


class TensorFlowCoreAPI(BaseCoreAPI, TensorFlowDeviceDataTypeAPI,
                        TensorFlowLogicAPI, TensorFlowMapAPI,
                        TensorFlowReduceAPI, TensorFlowShapeAPI,
                        TensorFlowVariableAPI):
    def __init__(self):
        BaseCoreAPI.__init__(self)
        TensorFlowDeviceDataTypeAPI.__init__(self)
        TensorFlowLogicAPI.__init__(self)
        TensorFlowMapAPI.__init__(self)
        TensorFlowReduceAPI.__init__(self)
        TensorFlowShapeAPI.__init__(self)
        TensorFlowVariableAPI.__init__(self)
