from ....base.layer.shape import BaseShapeAPI
from .pad import TensorFlowPadAPI


class TensorFlowShapeAPI(BaseShapeAPI, TensorFlowPadAPI):
    def __init__(self):
        BaseShapeAPI.__init__(self)
        TensorFlowPadAPI.__init__(self)
