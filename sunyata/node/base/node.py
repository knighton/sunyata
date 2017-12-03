from .pseudo_node import PseudoNode


class Node(PseudoNode):
    """
    A non-input node of a computational graph, that connects inputs to outputs.

    The forms of its inputs and outputs are fixed, but how it turns one into the
    other is up to the implementations.
    """

    @classmethod
    def _normalize_parents(cls, parents):
        parents = parents or []
        assert isinstance(parents, list)
        for parent in parents:
            assert isinstance(parent, PseudoNode)
        return parents

    def adopt_parent(self, parent):
        assert isinstance(parent, PseudoNode)
        index_of_child = parent.adopt_child(self)
        self._parents.append(parent)
        self._parent_indices.append(index_of_child)

    def __init__(self, parents):
        inputs_as_model = self.collect_inputs_as_model(parents)
        PseudoNode.__init__(self, inputs_as_model)
        self._parents = []
        self._parent_indices = []
        self._parents_ready_to_build = 0
        self._parents_ready_to_forward = 0
        for parent in self._normalize_parents(parents):
            self.adopt_parent(parent)

    def node_is_built(self):
        return self._parents_ready_to_build is None

    def node_build_inner(self, forms):
        """
        forms -> forms
        """
        raise NotImplementedError

    def node_build(self):
        self._parents_ready_to_build += 1
        if self._parents_ready_to_build < len(self._parents):
            return
        forms = []
        for parent in self._parents:
            forms += parent.forms()
        forms = self.node_build_inner(forms)
        self.initialize_forms(forms)
        for child in self.children():
            child.node_build()
        self._parents_ready_to_build = None

    def node_params_inner(self, nodes_seen, params_seen, params):
        """
        Node set, Variable set, Variable list ->
        """
        raise NotImplementedError

    def node_params(self, nodes_seen, params_seen, params):
        if self in nodes_seen:
            return
        nodes_seen.add(self)
        self.node_params_inner(nodes_seen, params_seen, params)
        for child in self.children():
            child.node_params(nodes_seen, params_seen, params)

    def node_forward_inner(self, xx, is_training):
        """
        xx, is_training -> yy
        """
        raise NotImplementedError

    def node_forward(self, is_training):
        self._parents_ready_to_forward += 1
        if self._parents_ready_to_forward < len(self._parents):
            return
        xx = []
        for parent in self.parents():
            xx += parent.data()
        yy = self.node_forward_inner(xx, is_training)
        self.set_data(yy)
        for child in self.children(is_training):
            child.node_forward(is_training)
        self._parents_ready_to_forward = 0

    @classmethod
    def as_node_list(cls, x):
        nodes = cls.as_list(x)
        for node in nodes:
            assert isinstance(node, Node)
        return nodes
