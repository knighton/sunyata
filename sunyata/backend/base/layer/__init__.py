from .activation import BaseActivationAPI
from .dot import BaseDotAPI


class BaseLayerAPI(BaseActivationAPI, BaseDotAPI):
    def __init__(self):
        BaseActivationAPI.__init__(self)
        BaseDotAPI.__init__(self)
