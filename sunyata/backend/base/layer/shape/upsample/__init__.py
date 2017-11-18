from .linear import BaseLinearUpsampleAPI
from .nearest import BaseNearestUpsampleAPI


class BaseUpsampleAPI(BaseLinearUpsampleAPI, BaseNearestUpsampleAPI):
    def __init__(self):
        BaseLinearUpsampleAPI.__init__(self)
        BaseNearestUpsampleAPI.__init__(self)
