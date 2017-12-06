from .base import ModelOrNode


class Graph(ModelOrNode):
    """
    A link/model realized as a static computational graph.
    """

    def __init__(self, inputs, outputs, _parents=None):
        parents = self.normalize_parents(_parents)
        ModelOrNode.__init__(self, parents)
        self._internal_inputs = self.collect_model_inputs(self.as_nodes(inputs))
        self._internal_outputs = self.as_children(outputs)

    def link_build_inner(self, forms):
        for input_ in self._internal_inputs:
            input_.input_build()
        forms = []
        for output in self._internal_outputs:
            assert output.node_is_built()
            forms += output.output_forms()
        return forms

    def link_params_inner(self, nodes_seen, params_seen, params):
        for input_ in self._internal_inputs:
            input_.input_params(nodes_seen, params_seen, params)

    def link_forward_inner(self, xx, is_training):
        assert len(self._internal_inputs) == len(xx)
        for input_, x in zip(self._internal_inputs, xx):
            input_.input_forward(x, is_training)
        yy = []
        for output in self._internal_outputs:
            yy += output.output_data()
        return yy
