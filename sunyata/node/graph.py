from .model_node import ModelNode


class Graph(ModelNode):
    """
    A model/node realized as a static computational graph.
    """

    def __init__(self, inputs, outputs, _parents=None):
        parents = self.normalize_parents(_parents)
        ModelNode.__init__(self, parents)
        self._node_inputs = \
            self.collect_model_inputs(self.as_pseudo_node_list(inputs))
        self._node_outputs = self.as_node_list(outputs)

    def node_build_inner(self, forms):
        for input_ in self._node_inputs:
            input_.input_build()
        forms = []
        for output in self._node_outputs:
            assert output.node_is_built()
            forms += output.output_forms()
        return forms

    def node_params_inner(self, nodes_seen, params_seen, params):
        for input_ in self._node_inputs:
            input_.input_params(nodes_seen, params_seen, params)

    def node_forward_inner(self, xx, is_training):
        assert len(self._node_inputs) == len(xx)
        for input_, x in zip(self._node_inputs, xx):
            input_.input_forward(x, is_training)
        yy = []
        for output in self._node_outputs:
            yy += output.output_data()
        return yy
