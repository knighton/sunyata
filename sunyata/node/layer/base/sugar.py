from copy import deepcopy
import inspect

from ...base.pseudo_node import PseudoNode


class LinkBuilder(PseudoNode):
    """
    Syntactic sugar for creating layer nodes.

    A spec factory with default arguments.  Returns orphan layer nodes.
    """

    def __init__(self, spec_class, default_kwargs=None):
        from . import Spec
        default_kwargs = default_kwargs or {}
        assert isinstance(default_kwargs, dict)
        assert issubclass(spec_class, Spec)
        self.spec_class = spec_class
        self.default_kwargs = default_kwargs or {}

    def __call__(self, *args, **kwargs):
        from ... import Link
        kwargs = deepcopy(self.default_kwargs)
        kwargs.update(deepcopy(kwargs))
        spec = self.spec_class(*args, **kwargs)
        return Link(spec)

    def desugar(self):
        return self.__call__()

    def pseudo_node_to_pretty(self):
        return 'Sugar(%s)' % self.spec_class.__name__[:-4]


def _normalize_ndims(spatial_ndims):
    if isinstance(spatial_ndims, (list, tuple)):
        assert len(set(spatial_ndims)) == len(spatial_ndims)
    else:
        spatial_ndims = [spatial_ndims]
    for spatial_ndim in spatial_ndims:
        assert spatial_ndim in {None, 0, 1, 2, 3}
    return spatial_ndims


def _spec_class_to_link_builder_base_name(spec_class):
    assert spec_class.__name__.endswith('Spec')
    name = spec_class.__name__[:-4]
    assert name
    return name


def _link_builder_name(base_name, spatial_ndim):
    if spatial_ndim is None:
        name = base_name
    elif isinstance(spatial_ndim, int):
        name = '%s%dD' % (base_name, spatial_ndim)
    else:
        assert False
    return name


def _link_builder(spec_class, spatial_ndim):
    default_kwargs = {'spatial_ndim': spatial_ndim}
    return LinkBuilder(spec_class, default_kwargs)


def _each_link_builder(spec_class, spatial_ndims):
    spatial_ndims = _normalize_ndims(spatial_ndims)
    base_name = _spec_class_to_link_builder_base_name(spec_class)
    for spatial_ndim in spatial_ndims:
        name = _link_builder_name(base_name, spatial_ndim)
        builder = _link_builder(spec_class, spatial_ndim)
        yield name, builder


def node_wrap(spec_class, spatial_ndims=None):
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    for builder_name, builder in _each_link_builder(spec_class, spatial_ndims):
        assert not hasattr(caller_module, builder_name)
        setattr(caller_module, builder_name, builder)
