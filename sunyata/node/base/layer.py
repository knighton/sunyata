class Layer(object):
    def params(self):
        return []

    def forward_multi(self, xx):
        raise NotImplementedError


class MergeLayer(Layer):
    pass


class TransformLayer(Layer):
    def forward_one(self, x):
        raise NotImplementedError

    def forward_multi(self, xx):
        assert len(xx) == 1
        x, = xx
        x = self.forward_one(x)
        return [x]
