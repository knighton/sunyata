class Model(object):
    """
    Object containing parameters which forward/backpropagates.
    """

    def __init__(self):
        self._model_is_built = False

    def is_built(self):
        return self._model_is_built

    def build_inner(self):
        """
        ->
        """
        raise NotImplementedError

    def build(self):
        if self._model_is_built:
            return
        self.build_inner()
        self._model_is_built = True

    def params(self):
        """
        -> list of Variable
        """
        raise NotImplementedError

    def forward(self, xx, is_training):
        """
        xx, is_training -> yy
        """
        raise NotImplementedError
