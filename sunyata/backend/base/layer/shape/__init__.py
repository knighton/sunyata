from .global_pool import BaseGlobalPoolAPI
from .pad import BasePadAPI
from .pool import BasePoolAPI
from .upsample import BaseUpsampleAPI


class BaseShapeAPI(BaseGlobalPoolAPI, BasePadAPI, BasePoolAPI, BaseUpsampleAPI):
    def __init__(self):
        BaseGlobalPoolAPI.__init__(self)
        BasePadAPI.__init__(self)
        BasePoolAPI.__init__(self)
        BaseUpsampleAPI.__init__(self)
