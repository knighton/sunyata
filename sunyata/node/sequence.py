from .base import TransformLayer, TransformSpec


class SequenceLayer(TransformLayer):
    def __init__(self, layers):
        for layer in layers:
            assert isinstance(layer, TransformLayer)
        self.layers = layers

    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params()
        return params

    def forward_one(self, x):
        for layer in self.layers:
            x = layer.forward_one(x)
        return x


class SequenceSpec(TransformSpec):
    def __init__(self, specs):
        self.specs = specs

    def build_one(self, form):
        layers = []
        for spec in self.specs:
            layer, form = spec.build_one(form)
            layers.append(layer)
        return SequenceLayer(layers), form
