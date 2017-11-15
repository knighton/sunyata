from ....base.layer.dot import BaseDotAPI
from .conv import ChainerConvAPI
from .dense import ChainerDenseAPI


class ChainerDotAPI(BaseDotAPI, ChainerConvAPI, ChainerDenseAPI):
    def __init__(self):
        BaseDotAPI.__init__(self)
        ChainerConvAPI.__init__(self)
        ChainerDenseAPI.__init__(self)
