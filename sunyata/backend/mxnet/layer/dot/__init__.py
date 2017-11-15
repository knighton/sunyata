from ....base.layer.dot import BaseDotAPI
from .conv import MXNetConvAPI
from .dense import MXNetDenseAPI


class MXNetDotAPI(BaseDotAPI, MXNetConvAPI, MXNetDenseAPI):
    def __init__(self):
        BaseDotAPI.__init__(self)
        MXNetConvAPI.__init__(self)
        MXNetDenseAPI.__init__(self)
