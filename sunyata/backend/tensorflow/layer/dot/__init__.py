from ....base.layer.dot import BaseDotAPI
from .conv import TensorFlowConvAPI
from .dense import TensorFlowDenseAPI


class TensorFlowDotAPI(BaseDotAPI, TensorFlowConvAPI, TensorFlowDenseAPI):
    def __init__(self):
        BaseDotAPI.__init__(self)
        TensorFlowConvAPI.__init__(self)
        TensorFlowDenseAPI.__init__(self)
