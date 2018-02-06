from copy import deepcopy

from .child_node import ChildNode
from .model_or_node import ModelOrNode


class Sequence(ModelOrNode):
    """
    A model/node realized as a sequence of nodes (with predecessors).
    """

    @classmethod
    def _desugar_nodes(cls, nodes):
        assert nodes
        assert isinstance(nodes, (list, tuple))
        return list(map(lambda node: node.desugar(), nodes))

    def __init__(self, *nodes, _parents=None):
        parents_via_call = self.normalize_parents(_parents)
        nodes = self._desugar_nodes(nodes)
        sequence_head = nodes[0]
        if parents_via_call:
            # Connected to previous node(s) via __call__().
            assert not sequence_head.parents()
            eject_head = False
            parents = parents_via_call
        elif sequence_head.model_inputs():
            # Connected to previous via the first node of its sequence.
            assert 2 <= len(nodes)
            eject_head = True
            parents = [sequence_head]
        else:
            # Orphan sequences.
            eject_head = False
            parents = None
        ModelOrNode.__init__(self, parents)
        steps = nodes[int(eject_head):]
        for step in steps:
            assert isinstance(step, ChildNode)
            assert not step.parents()
            assert not step.children()
        self._steps = steps

    def pseudo_node_to_pretty(self):
        ss = []
        for step in self._steps:
            ss.append(step.pseudo_node_to_pretty())
        return '[%s]' % ' > '.join(ss)

    def __call__(self, *parents):
        assert parents
        nodes = deepcopy(self._steps)
        return Sequence(*nodes, _parents=parents)

    def child_build_inner(self, forms):
        for step in self._steps:
            forms = step.child_build_inner(forms)
        return forms

    def child_params_inner(self, nodes_seen, params_seen, params):
        for step in self._steps:
            step.child_params_inner(nodes_seen, params_seen, params)

    def child_forward_inner(self, xx, train):
        for step in self._steps:
            xx = step.child_forward_inner(xx, train)
        return xx
