from copy import deepcopy

from ..layer.base import Layer, Spec
from .model_or_node import ModelOrNode


class Link(ModelOrNode):
    """
    A model/node realized as a single layer node (with predecessors).
    """

    def __init__(self, spec, _parents=None):
        parents = self.normalize_parents(_parents)
        ModelOrNode.__init__(self, parents)
        assert isinstance(spec, Spec)
        self._spec = spec
        self._layer = None

    def pseudo_node_to_pretty(self):
        return self._spec.__class__.__name__[:-4]

    def __call__(self, *parents):
        assert parents
        return Link(deepcopy(self._spec), parents)

    def child_build_inner(self, forms):
        self._layer, forms = self._spec.build(forms)
        assert isinstance(self._layer, Layer)
        return forms

    def child_params_inner(self, nodes_seen, params_seen, params):
        for param in self._layer.params():
            if param in params_seen:
                continue
            params_seen.add(param)
            params.append(param)

    def child_forward_inner(self, xx, train):
        return self._layer.forward(xx, train)
