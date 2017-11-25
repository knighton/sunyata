from ...base.core import BaseCoreAPI
from .device_dtype import TensorFlowDeviceDataTypeAPI
from .linalg import TensorFlowLinearAlgebraAPI
from .logic import TensorFlowLogicAPI
from .map import TensorFlowMapAPI
from .reduce import TensorFlowReduceAPI
from .shape import TensorFlowShapeAPI
from .tensor import TensorFlowTensorAPI
from .variable import TensorFlowVariableAPI


class TensorFlowCoreAPI(BaseCoreAPI, TensorFlowDeviceDataTypeAPI,
                        TensorFlowLinearAlgebraAPI, TensorFlowLogicAPI,
                        TensorFlowMapAPI, TensorFlowReduceAPI,
                        TensorFlowShapeAPI, TensorFlowTensorAPI,
                        TensorFlowVariableAPI):
    def __init__(self):
        BaseCoreAPI.__init__(self)
        TensorFlowDeviceDataTypeAPI.__init__(self)
        TensorFlowLinearAlgebraAPI.__init__(self)
        TensorFlowLogicAPI.__init__(self)
        TensorFlowMapAPI.__init__(self)
        TensorFlowReduceAPI.__init__(self)
        TensorFlowShapeAPI.__init__(self)
        TensorFlowTensorAPI.__init__(self)
        TensorFlowVariableAPI.__init__(self)
