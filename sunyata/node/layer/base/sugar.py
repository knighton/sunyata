from copy import deepcopy

from ...base.pseudo_node import PseudoNode


class LinkBuilder(PseudoNode):
    """
    Syntactic sugar for creating layer nodes.

    A spec factory with default arguments.  Returns orphan layer nodes.
    """

    def __init__(self, spec_class, default_kwargs=None):
        from .spec import Spec
        default_kwargs = default_kwargs or {}
        assert isinstance(default_kwargs, dict)
        assert issubclass(spec_class, Spec)
        self.spec_class = spec_class
        self.default_kwargs = default_kwargs or {}

    def __call__(self, *args, **kwargs):
        from ..network import Link
        kw = deepcopy(self.default_kwargs)
        kw.update(deepcopy(kwargs))
        spec = self.spec_class(*args, **kw)
        return Link(spec)
