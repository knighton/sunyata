from copy import deepcopy

from ..node.base.layer import Layer
from ..node.base.spec import Spec
from .base.model_or_nexus import ModelOrNexus


class Atom(ModelOrNexus):
    def __init__(self, spec, _parents=None):
        parents = self.normalize_parents(_parents)
        ModelOrNexus.__init__(self, parents)
        assert isinstance(spec, Spec)
        self._spec = spec
        self._layer = None

    def __call__(self, *parents):
        assert not parents
        return Atom(deepcopy(self._spec), parents)

    def node_build_inner(self, forms):
        self._layer, forms = self._spec.build(forms)
        assert isinstance(self._layer, Layer)
        return forms

    def node_params_inner(self, nodes_seen, params_seen, params):
        for param in self._layer.params():
            if param in params_seen:
                continue
            params_seen.add(param)
            params.append(param)

    def node_forward_inner(self, xx, is_training):
        return self._layer.forward(xx, is_training)
