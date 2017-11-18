from .global_pool import BaseGlobalPoolAPI
from .pad import BasePadAPI
from .pool import BasePoolAPI


class BaseShapeAPI(BaseGlobalPoolAPI, BasePadAPI, BasePoolAPI):
    def __init__(self):
        BaseGlobalPoolAPI.__init__(self)
        BasePadAPI.__init__(self)
        BasePoolAPI.__init__(self)
