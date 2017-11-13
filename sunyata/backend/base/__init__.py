from .core import BaseCoreAPI
from .layer import BaseLayerAPI
from .metric import BaseMetricAPI


class BaseBackend(BaseCoreAPI, BaseLayerAPI, BaseMetricAPI):
    def __init__(self):
        BaseCoreAPI.__init__(self)
        BaseLayerAPI.__init__(self)
        BaseMetricAPI.__init__(self)
