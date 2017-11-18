from .pad import BasePadAPI
from .pool import BasePoolAPI


class BaseShapeAPI(BasePadAPI, BasePoolAPI):
    def __init__(self):
        BasePadAPI.__init__(self)
        BasePoolAPI.__init__(self)
