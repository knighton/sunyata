import numpy as np

from .base import APIBase
from .device_dtype import BaseDeviceAPI, BaseDataTypeAPI, BaseDeviceDataTypeAPI
from .epsilon import BaseEpsilonAPI
from .layer import BaseLayerAPI
from .layer.activation import BaseActivationAPI
from .layer.dense import BaseDenseAPI
from .logic import BaseLogicAPI
from .map import BaseMapAPI
from .metric import BaseMetricAPI
from .shape import BaseShapeAPI
from .reduce import BaseReduceAPI
from .variable import BaseVariableAPI


class BaseBackend(BaseDeviceDataTypeAPI, BaseEpsilonAPI, BaseLogicAPI,
                  BaseMapAPI, BaseLayerAPI, BaseMetricAPI, BaseReduceAPI,
                  BaseShapeAPI, BaseVariableAPI):
    def __init__(self):
        BaseDeviceDataTypeAPI.__init__(self)
        BaseEpsilonAPI.__init__(self)
        BaseLayerAPI.__init__(self)
        BaseLogicAPI.__init__(self)
        BaseMapAPI.__init__(self)
        BaseMetricAPI.__init__(self)
        BaseReduceAPI.__init__(self)
        BaseShapeAPI.__init__(self)
        BaseVariableAPI.__init__(self)

    def zeros_like(self, x):
        return self.cast_numpy_to(np.zeros(self.shape(x), self.dtype_of(x)))
