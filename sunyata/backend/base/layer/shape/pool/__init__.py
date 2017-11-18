from .avg import BaseAvgPoolAPI
from .max import BaseMaxPoolAPI


class BasePoolAPI(BaseAvgPoolAPI, BaseMaxPoolAPI):
    def __init__(self):
        BaseAvgPoolAPI.__init__(self)
        BaseMaxPoolAPI.__init__(self)
