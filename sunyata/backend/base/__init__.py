from .core import BaseCoreAPI
from .core.device_dtype import \
    BaseDeviceAPI, BaseDataTypeAPI, BaseDeviceDataTypeAPI
from .core.epsilon import BaseEpsilonAPI
from .core.logic import BaseLogicAPI
from .core.map import BaseMapAPI
from .core.shape import BaseShapeAPI
from .core.reduce import BaseReduceAPI
from .core.variable import BaseVariableAPI
from .layer import BaseLayerAPI
from .layer.activation import BaseActivationAPI
from .layer.dense import BaseDenseAPI
from .metric import BaseMetricAPI


class BaseBackend(BaseCoreAPI, BaseLayerAPI, BaseMetricAPI):
    def __init__(self):
        BaseCoreAPI.__init__(self)
        BaseLayerAPI.__init__(self)
        BaseMetricAPI.__init__(self)
