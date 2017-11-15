from ....base.layer.shape import BaseShapeAPI
from .pad import MXNetPadAPI


class MXNetShapeAPI(BaseShapeAPI, MXNetPadAPI):
    def __init__(self):
        BaseShapeAPI.__init__(self)
        MXNetPadAPI.__init__(self)
