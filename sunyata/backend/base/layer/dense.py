from ..base import APIBase


class BaseDenseAPI(APIBase):
    def dense(self, x, kernel, bias):
        raise NotImplementedError
