from copy import deepcopy

from ..base.layer import Layer
from ..base.spec import Spec
from .base.link_or_model import LinkOrModel


class Atom(LinkOrModel):
    def __init__(self, spec, _parents=None):
        parents = self.normalize_parents(_parents)
        LinkOrModel.__init__(self, parents)
        assert isinstance(spec, Spec)
        self._spec = spec
        self._layer = None

    def __call__(self, *parents):
        assert not parents
        return Atom(deepcopy(self._spec), parents)

    def link_build_inner(self, forms):
        self._layer, forms = self._spec.build(forms)
        assert isinstance(self._layer, Layer)
        return forms

    def link_params_inner(self, nodes_seen, params_seen, params):
        for param in self._layer.params():
            if param in params_seen:
                continue
            params_seen.add(param)
            params.append(param)

    def link_forward_inner(self, xx, is_training):
        return self._layer.forward(xx, is_training)
