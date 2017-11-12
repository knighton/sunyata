from ..base import APIBase


class BaseDenseAPI(APIBase):
    def __init__(self):
        APIBase.__init__(self)

    def dense(self, x, kernel, bias):
        raise NotImplementedError
