from ....base.layer.shape import BaseShapeAPI
from .pad import TensorFlowPadAPI
from .upsample import TensorFlowUpsampleAPI


class TensorFlowShapeAPI(BaseShapeAPI, TensorFlowPadAPI, TensorFlowUpsampleAPI):
    def __init__(self):
        BaseShapeAPI.__init__(self)
        TensorFlowPadAPI.__init__(self)
        TensorFlowUpsampleAPI.__init__(self)
