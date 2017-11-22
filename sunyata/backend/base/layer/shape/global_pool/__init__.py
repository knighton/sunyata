from .avg import BaseGlobalAvgPoolAPI
from .max import BaseGlobalMaxPoolAPI


class BaseGlobalPoolAPI(BaseGlobalAvgPoolAPI, BaseGlobalMaxPoolAPI):
    def __init__(self):
        BaseGlobalAvgPoolAPI.__init__(self)
        BaseGlobalMaxPoolAPI.__init__(self)

    def global_pool_out_shape(self, in_shape):
        return in_shape[0],
