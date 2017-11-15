from .....base.layer.shape.pad import BasePadAPI
from .constant import MXNetConstantPadAPI
from .edge import MXNetEdgePadAPI
from .reflect import MXNetReflectPadAPI


class MXNetPadAPI(BasePadAPI, MXNetConstantPadAPI, MXNetEdgePadAPI,
                  MXNetReflectPadAPI):
    def __init__(self):
        BasePadAPI.__init__(self)
        MXNetConstantPadAPI.__init__(self)
        MXNetEdgePadAPI.__init__(self)
        MXNetReflectPadAPI.__init__(self)
