from ...base import APIMixin


class BaseDenseAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def dense(self, x, kernel, bias):
        raise NotImplementedError
