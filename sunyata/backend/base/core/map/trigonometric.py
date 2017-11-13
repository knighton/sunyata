from ...base import APIMixin


class BaseTrigonometricAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def sin(self, x):
        raise NotImplementedError

    def cos(self, x):
        raise NotImplementedError

    def tan(self, x):
        raise NotImplementedError

    def arcsin(self, x):
        raise NotImplementedError

    def arccos(self, x):
        raise NotImplementedError

    def arctan(self, x):
        raise NotImplementedError
