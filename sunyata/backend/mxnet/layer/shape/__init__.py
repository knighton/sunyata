from ....base.layer.shape import BaseShapeAPI
from .pad import MXNetPadAPI
from .pool import MXNetPoolAPI
from .upsample import MXNetUpsampleAPI


class MXNetShapeAPI(BaseShapeAPI, MXNetPadAPI, MXNetPoolAPI, MXNetUpsampleAPI):
    def __init__(self):
        BaseShapeAPI.__init__(self)
        MXNetPadAPI.__init__(self)
        MXNetPoolAPI.__init__(self)
        MXNetUpsampleAPI.__init__(self)
