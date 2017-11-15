from .constant import BaseConstantPadAPI
from .edge import BaseEdgePadAPI
from .reflect import BaseReflectPadAPI


class BasePadAPI(BaseConstantPadAPI, BaseEdgePadAPI, BaseReflectPadAPI):
    def __init__(self):
        BaseConstantPadAPI.__init__(self)
        BaseEdgePadAPI.__init__(self)
        BaseReflectPadAPI.__init__(self)
