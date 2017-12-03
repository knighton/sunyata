class Sequence(ModelNode):
    """
    A model/node realized as a sequence of nodes.
    """

    @classmethod
    def _seq_init_parents(cls, parents):
        if parents is None:
            parents = []
        else:
            assert parents
            assert isinstance(parents, tuple)
            for parent in parents:
                assert isinstance(parent, PseudoNode)
            parents = list(parents)
        return parents

    @classmethod
    def _seq_init_head_steps(cls, nodes, has_parents):
        assert nodes
        assert isinstance(nodes, list)
        if has_parents:
            for node in nodes:
                assert isinstance(node, Node)
                assert not node.parents()
                assert not node.children()
            head = nodes[0]
            steps = nodes[1:]
        else:
            head = nodes[0]
            assert isinstance(head, PseudoNode)
            assert not head.children()
            steps = nodes[1:]
            for step in steps:
                assert isinstance(step, Node)
                assert not step.parents()
                assert not step.children()
        return head, steps

    def __init__(self, *nodes, _parents=None):
        parents = self._seq_init_parents(_parents)
        head, steps = self._seq_init_steps(nodes, bool(parents))
        for parent in parents:
            head.adopt_parent(parent)
        self.adopt_parent(head)
        self._steps = steps

    def node_build_inner(self, forms):
        for step in self._steps:
            forms = step.node_build_inner(forms)
        return forms

    def node_params_inner(self, nodes_seen, params_seen, params):
        for step in self._steps:
            step.node_params_inner(nodes_seen, params_seen, params)

    def node_forward_inner(self, xx, is_training):
        for step in self._steps:
            xx = step.node_forward_inner(xx, is_training)
        return xx
