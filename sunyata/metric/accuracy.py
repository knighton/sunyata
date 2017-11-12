from .. import backend as Z
from .base import Metric


class CategoricalAccuracy(Metric):
    def __call__(self, true, pred):
        return Z.categorical_accuracy(true, pred)
