from .base import TransformLayer, TransformSpec


class SequenceLayer(TransformLayer):
    def __init__(self, layers):
        super().__init__()
        for layer in layers:
            assert isinstance(layer, TransformLayer)
        self.layers = layers

    def params(self):
        params = []
        for layer in self.layers:
            params += layer.params()
        return params

    def forward_one(self, x, is_training):
        for layer in self.layers:
            x = layer.forward_one(x, is_training)
        return x


class SequenceSpec(TransformSpec):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs

    def build_one(self, form):
        layers = []
        for spec in self.specs:
            layer, form = spec.build_one(form)
            layers.append(layer)
        return SequenceLayer(layers), form
