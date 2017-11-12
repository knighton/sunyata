from .activation import TensorFlowActivationAPI
from .dense import TensorFlowDenseAPI


class TensorFlowLayerAPI(TensorFlowActivationAPI, TensorFlowDenseAPI):
    def __init__(self):
        TensorFlowActivationAPI.__init__(self)
        TensorFlowDenseAPI.__init__(self)
