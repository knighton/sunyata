from .node import Node


class ChildNode(Node):
    """
    A non-input node of a computational graph, that links inputs to outputs.

    The forms of its inputs and outputs are fixed, but how it turns one into the
    other is up to the implementation.
    """

    @classmethod
    def normalize_parents(cls, parents):
        if parents is None:
            parents = []
        else:
            assert isinstance(parents, (list, tuple))
            for parent in parents:
                assert isinstance(parent, Node)
            parents = list(parents)
        return parents

    def adopt_parent(self, parent):
        assert isinstance(parent, Node)
        index_of_child = parent.adopt_child(self)
        self._parents.append(parent)
        self._parent_indices.append(index_of_child)

    def __init__(self, parents):
        parents = self.as_list(parents)
        model_inputs = self.collect_model_inputs(parents)
        Node.__init__(self, model_inputs)
        self._parents = []
        self._parent_indices = []
        self._parents_ready_to_build = 0
        self._parents_ready_to_forward = 0
        for parent in parents:
            self.adopt_parent(parent)

    def parents(self):
        return self._parents

    def child_is_built(self):
        return self._parents_ready_to_build is None

    def child_build_inner(self, forms):
        """
        forms -> forms
        """
        raise NotImplementedError

    def child_build(self):
        self._parents_ready_to_build += 1
        if self._parents_ready_to_build < len(self._parents):
            return
        input_forms = []
        for parent in self._parents:
            input_forms += parent.output_forms()
        output_forms = self.child_build_inner(input_forms)
        self.initialize_output_forms(output_forms)
        for child in self.children():
            child.child_build()
        self._parents_ready_to_build = None

    def child_params_inner(self, nodes_seen, params_seen, params):
        """
        Node set, Variable set, Variable list ->
        """
        raise NotImplementedError

    def child_params(self, nodes_seen, params_seen, params):
        if self in nodes_seen:
            return
        nodes_seen.add(self)
        self.child_params_inner(nodes_seen, params_seen, params)
        for child in self.children():
            child.child_params(nodes_seen, params_seen, params)

    def child_forward_inner(self, xx, is_training):
        """
        xx, is_training -> yy
        """
        raise NotImplementedError

    def child_forward(self, is_training):
        self._parents_ready_to_forward += 1
        if self._parents_ready_to_forward < len(self._parents):
            return
        xx = []
        for parent in self.parents():
            xx += parent.output_data()
        yy = self.child_forward_inner(xx, is_training)
        self.set_output_data(yy)
        for child in self.children(is_training):
            child.child_forward(is_training)
        self._parents_ready_to_forward = 0

    @classmethod
    def as_children(cls, x):
        children = cls.as_list(x)
        for child in children:
            assert isinstance(child, ChildNode)
        return children
