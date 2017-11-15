from .conv import BaseConvAPI
from .dense import BaseDenseAPI


class BaseDotAPI(BaseConvAPI, BaseDenseAPI):
    def __init__(self):
        BaseConvAPI.__init__(self)
        BaseDenseAPI.__init__(self)
