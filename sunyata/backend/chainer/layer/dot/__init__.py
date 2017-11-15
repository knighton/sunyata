from ....base.layer.dot import BaseDotAPI
from .dense import ChainerDenseAPI


class ChainerDotAPI(BaseDotAPI, ChainerDenseAPI):
    def __init__(self):
        BaseDotAPI.__init__(self)
        ChainerDenseAPI.__init__(self)
