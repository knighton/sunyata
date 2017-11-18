from .....base.layer.shape.upsample import BaseUpsampleAPI
from .nearest import MXNetNearestUpsampleAPI


class MXNetUpsampleAPI(BaseUpsampleAPI, MXNetNearestUpsampleAPI):
    def __init__(self):
        BaseUpsampleAPI.__init__(self)
        MXNetNearestUpsampleAPI.__init__(self)
