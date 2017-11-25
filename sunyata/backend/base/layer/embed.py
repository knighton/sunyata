from ..base import APIMixin


class BaseEmbedAPI(APIMixin):
    def __init__(self):
        APIMixin.__init__(self)

    def embed(self, x, reference):
        raise NotImplementedError
