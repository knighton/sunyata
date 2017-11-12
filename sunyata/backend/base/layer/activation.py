from ..base import APIMixin


class BaseActivationAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def relu(self, x):
        return self.clip(x, min=0)

    def softmax(self, x):
        raise NotImplementedError
