from ...base.layer import BaseLayerAPI
from .activation import MXNetActivationAPI
from .dot import MXNetDotAPI


class MXNetLayerAPI(BaseLayerAPI, MXNetActivationAPI, MXNetDotAPI):
    def __init__(self):
        BaseLayerAPI.__init__(self)
        MXNetActivationAPI.__init__(self)
        MXNetDotAPI.__init__(self)
