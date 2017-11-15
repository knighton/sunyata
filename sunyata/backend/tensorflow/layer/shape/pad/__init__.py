from .....base.layer.shape.pad import BasePadAPI
from .constant import TensorFlowConstantPadAPI
from .reflect import TensorFlowReflectPadAPI


class TensorFlowPadAPI(BasePadAPI, TensorFlowConstantPadAPI,
                       TensorFlowReflectPadAPI):
    def __init__(self):
        BasePadAPI.__init__(self)
        TensorFlowConstantPadAPI.__init__(self)
        TensorFlowReflectPadAPI.__init__(self)
