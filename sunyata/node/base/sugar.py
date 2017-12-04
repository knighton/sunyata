from copy import deepcopy

from ..network.atom import Atom
from .pseudo_node import PseudoNode
from .spec import Spec


class Sugar(PseudoNode):
    """
    Syntactic sugar for creating layer nodes.

    A spec factory with default arguments.  Returns orphan layer nodes.
    """

    def __init__(self, spec_class, default_kwargs=None):
        default_kwargs = default_kwargs or {}
        assert isinstance(default_kwargs, dict)
        assert issubclass(spec_class, Spec)
        self.spec_class = spec_class
        self.default_kwargs = default_kwargs or {}

    def __call__(self, *args, **kwargs):
        kw = deepcopy(self.default_kwargs)
        kw.update(deepcopy(kwargs))
        spec = self.spec_class(*args, **kw)
        return Atom(spec)
