import numpy as np

from .activation import BaseActivationAPI
from .base import APIBase
from .device_dtype import BaseDeviceAPI, BaseDataTypeAPI, BaseDeviceDataTypeAPI
from .epsilon import BaseEpsilonAPI
from .logic import BaseLogicAPI
from .map import BaseMapAPI
from .metric import BaseMetricAPI
from .reduce import BaseReduceAPI
from .variable import BaseVariableAPI


class BaseDenseAPI(APIBase):
    def dense(self, x, kernel, bias):
        raise NotImplementedError


class BaseShapeAPI(APIBase):
    def ndim(self, x):
        raise NotImplementedError

    def shape(self, x):
        raise NotImplementedError

    def size(self, x):
        raise NotImplementedError

    def reshape(self, x, shape):
        raise NotImplementedError

    def expand_dims(self, x, axis):
        raise NotImplementedError


class BaseBackend(BaseActivationAPI, BaseDeviceDataTypeAPI, BaseEpsilonAPI,
                  BaseLogicAPI, BaseMapAPI, BaseMetricAPI, BaseReduceAPI,
                  BaseDenseAPI, BaseShapeAPI, BaseVariableAPI):
    def __init__(self):
        BaseActivationAPI.__init__(self)
        BaseDeviceDataTypeAPI.__init__(self)
        BaseEpsilonAPI.__init__(self)
        BaseLogicAPI.__init__(self)
        BaseMapAPI.__init__(self)
        BaseMetricAPI.__init__(self)
        BaseReduceAPI.__init__(self)
        BaseDenseAPI.__init__(self)
        BaseShapeAPI.__init__(self)
        BaseVariableAPI.__init__(self)

    def zeros_like(self, x):
        return self.cast_numpy_to(np.zeros(self.shape(x), self.dtype_of(x)))
