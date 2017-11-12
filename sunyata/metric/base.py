class Metric(object):
    def __init__(self, importance=1):
        self.importance = importance

    def __call__(self, true, pred):
        raise NotImplementedError
