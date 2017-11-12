from .activation import BaseActivationAPI
from .dense import BaseDenseAPI


class BaseLayerAPI(BaseActivationAPI, BaseDenseAPI):
    def __init__(self):
        BaseActivationAPI.__init__(self)
        BaseDenseAPI.__init__(self)
