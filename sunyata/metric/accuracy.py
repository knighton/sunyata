from .. import backend as Z
from .base import Metric


class BinaryAccuracy(Metric):
    def __call__(self, true, pred):
        return Z.binary_accuracy(true, pred)


class CategoricalAccuracy(Metric):
    def __call__(self, true, pred):
        return Z.categorical_accuracy(true, pred)
