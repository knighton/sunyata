from .....base.layer.shape.pool import BasePoolAPI
from .avg import MXNetAvgPoolAPI
from .max import MXNetMaxPoolAPI


class MXNetPoolAPI(BasePoolAPI, MXNetAvgPoolAPI, MXNetMaxPoolAPI):
    def __init__(self):
        BasePoolAPI.__init__(self)
        MXNetAvgPoolAPI.__init__(self)
        MXNetMaxPoolAPI.__init__(self)
