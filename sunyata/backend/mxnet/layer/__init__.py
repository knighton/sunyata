from .activation import MXNetActivationAPI
from .dense import MXNetDenseAPI


class MXNetLayerAPI(MXNetActivationAPI, MXNetDenseAPI):
    def __init__(self):
        MXNetActivationAPI.__init__(self)
        MXNetDenseAPI.__init__(self)
