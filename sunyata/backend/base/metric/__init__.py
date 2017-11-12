from .accuracy import BaseAccuracyAPI
from .loss import BaseLossAPI


class BaseMetricAPI(BaseAccuracyAPI, BaseLossAPI):
    def __init__(self):
        BaseAccuracyAPI.__init__(self)
        BaseLossAPI.__init__(self)
