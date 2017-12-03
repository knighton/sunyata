from copy import deepcopy

from .base.model_node import ModelNode
from .base.pseudo_node import PseudoNode


class LayerNode(ModelNode):
    @classmethod
    def _init_parents(cls, parents):
        if parents is None:
            parents = []
        else:
            assert parents
            assert isinstance(parents, tuple)
            for parent in parents:
                assert isinstance(parent, PseudoNode)
            parents = list(parents)
        return parents

    def __init__(self, spec, _parents=None):
        parents = self._init_parents(_parents)
        for parent in parents:
            self.adopt_parent(parent)
        self._spec = spec
        self._layer = None

    def __call__(self, *parents):
        assert not parents
        return LayerNode(deepcopy(self._spec), parents)

    def node_build_inner(self, forms):
        self._layer, forms = self._spec.build(forms)
        return forms

    def node_params_inner(self, nodes_seen, params_seen, params):
        for param in self._layer.params():
            if param in params_seen:
                continue
            params_seen.add(param)
            params.append(param)

    def node_forward_inner(self, xx, is_training):
        return self._layer.forward(xx, is_training)
