from ....base.layer.shape import BaseShapeAPI
from .pad import ChainerPadAPI


class ChainerShapeAPI(BaseShapeAPI, ChainerPadAPI):
    def __init__(self):
        BaseShapeAPI.__init__(self)
        ChainerPadAPI.__init__(self)
