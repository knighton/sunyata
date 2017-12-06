from copy import deepcopy
import inspect

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


def _normalize_ndims(ndims):
    if isinstance(ndims, (list, tuple)):
        assert len(set(ndims)) == len(ndims)
    else:
        ndims = [ndims]
    for ndim in ndims:
        assert ndim in {None, 0, 1, 2, 3}
    return ndims


def _spec_class_to_link_builder_base_name(spec_class):
    assert spec_class.__name__.endswith('Spec')
    name = spec_class.__name__[:-4]
    assert name
    return name


def _link_builder_name(base_name, ndim):
    if ndim is None:
        name = base_name
    elif isinstance(ndim, int):
        name = '%s%dD' % (base_name, ndim)
    else:
        assert False
    return name


def _link_builder(spec_class, ndim):
    default_kwargs = {'ndim': ndim}
    return LinkBuilder(spec_class, default_kwargs)


def _each_link_builder(spec_class, ndims):
    ndims = _normalize_ndims(ndims)
    base_name = _spec_class_to_link_builder_base_name(spec_class)
    for ndim in ndims:
        name = _link_builder_name(base_name, ndim)
        builder = _link_builder(spec_class, ndim)
        yield name, builder


def node_wrap(spec_class, ndims=None):
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    for builder_name, builder in _each_link_builder(spec_class, ndims):
        assert not hasattr(caller_module, builder_name)
        setattr(caller_module, builder_name, builder)
