from ..base import ChildNode, Node
from .base import ModelOrNode


class Chain(ModelOrNode):
    """
    A link/model realized as a sequence of nodes.
    """

    @classmethod
    def _seq_init_head_steps(cls, nodes, has_parents):
        assert nodes
        assert isinstance(nodes, list)
        if has_parents:
            for node in nodes:
                assert isinstance(node, ChildNode)
                assert not node.parents()
                assert not node.children()
            head = nodes[0]
            steps = nodes[1:]
        else:
            head = nodes[0]
            assert isinstance(head, Node)
            assert not head.children()
            steps = nodes[1:]
            for step in steps:
                assert isinstance(step, ChildNode)
                assert not step.parents()
                assert not step.children()
        return head, steps

    def __init__(self, *nodes, _parents=None):
        parents = self.normalize_parents(_parents)
        head, steps = self._seq_init_steps(nodes, bool(parents))
        for parent in parents:
            head.adopt_parent(parent)
        ModelOrNode.__init__(self, [head])
        self._steps = steps

    def link_build_inner(self, forms):
        for step in self._steps:
            forms = step.link_build_inner(forms)
        return forms

    def link_params_inner(self, nodes_seen, params_seen, params):
        for step in self._steps:
            step.link_params_inner(nodes_seen, params_seen, params)

    def link_forward_inner(self, xx, is_training):
        for step in self._steps:
            xx = step.link_forward_inner(xx, is_training)
        return xx
