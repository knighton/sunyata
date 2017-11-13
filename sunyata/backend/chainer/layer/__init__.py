from .activation import ChainerActivationAPI
from .dense import ChainerDenseAPI


class ChainerLayerAPI(ChainerActivationAPI, ChainerDenseAPI):
    def __init__(self):
        ChainerActivationAPI.__init__(self)
        ChainerDenseAPI.__init__(self)
