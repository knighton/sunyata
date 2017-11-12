import numpy as np

from .activation import BaseActivationAPI
from .base import APIBase
from .device_dtype import BaseDeviceAPI, BaseDataTypeAPI, BaseDeviceDataTypeAPI
from .epsilon import BaseEpsilonAPI
from .logic import BaseLogicAPI
from .map import BaseMapAPI
from .metric import BaseMetricAPI
from .reduce import BaseReduceAPI


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


class BaseVariableAPI(APIBase):
    def constant(self, x):
        raise NotImplementedError

    def _variable_name(self, name=None):
        if name is None:
            name = str(1 << 30)
        else:
            assert isinstance(name, str)
            assert name
        return name

    def variable(self, x):
        raise NotImplementedError

    def _aux_scores(self, aux_judges, yy_true, yy_pred):
        if aux_judges is None:
            return None
        aux_scores = []
        for y_aux_judges, y_true, y_pred in zip(aux_judges, yy_true, yy_pred):
            y_aux_scores = []
            for judge in y_aux_judges:
                result = self.mean(judge(y_true, y_pred))
                y_aux_scores.append(self.result_to_tensor(result))
            aux_scores.append(y_aux_scores)
        return aux_scores

    def gradients(self, params, forward, judges, aux_judges, xx, yy_true):
        raise NotImplementedError

    def variable_to_tensor(self, x):
        raise NotImplementedError

    def result_to_tensor(self, x):
        raise NotImplementedError

    def assign(self, x, new_value):
        raise NotImplementedError

    def incr(self, x, incr):
        self.assign(x, self.variable_to_tensor(x) + incr)

    def decr(self, x, decr):
        self.assign(x, self.variable_to_tensor(x) - decr)

    def numpy(self, x):
        raise NotImplementedError

    def list(self, x):
        return self.numpy(x).tolist()

    def scalar(self, x):
        assert self.size(x) == 1
        return self.numpy(x).flatten()[0]


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
