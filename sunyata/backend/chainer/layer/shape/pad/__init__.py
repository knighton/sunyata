from .....base.layer.shape.pad import BasePadAPI
from .constant import ChainerConstantPadAPI
from .edge import ChainerEdgePadAPI
from .reflect import ChainerReflectPadAPI


class ChainerPadAPI(BasePadAPI, ChainerConstantPadAPI, ChainerEdgePadAPI,
                    ChainerReflectPadAPI):
    def __init__(self):
        BasePadAPI.__init__(self)
        ChainerConstantPadAPI.__init__(self)
        ChainerEdgePadAPI.__init__(self)
        ChainerReflectPadAPI.__init__(self)
