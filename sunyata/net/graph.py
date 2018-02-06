from .model_or_node import ModelOrNode


class Graph(ModelOrNode):
    """
    A model/node realized as a static computational graph (with predecessors).
    """

    def __init__(self, inputs, outputs, _parents=None):
        parents = self.normalize_parents(_parents)
        ModelOrNode.__init__(self, parents)
        self._internal_inputs = self.collect_model_inputs(self.as_nodes(inputs))
        self._internal_outputs = self.as_children(outputs)

    def pseudo_node_to_pretty(self):
        inputs = []
        for input_ in self._internal_inputs:
            inputs.append(input_.to_pretty())
        outputs = []
        for output in self._internal_outputs:
            outputs.append(output.to_pretty())
        return 'Graph(%s -> %s)' % (', '.join(inputs), ', '.join(outputs))

    def __call__(self, *parents):
        raise NotImplementedError  # TODO

    def child_build_inner(self, forms):
        for input_ in self._internal_inputs:
            input_.input_build()
        forms = []
        for output in self._internal_outputs:
            assert output.node_is_built()
            forms += output.output_forms()
        return forms

    def child_params_inner(self, nodes_seen, params_seen, params):
        for input_ in self._internal_inputs:
            input_.input_params(nodes_seen, params_seen, params)

    def child_forward_inner(self, xx, train):
        assert len(self._internal_inputs) == len(xx)
        for input_, x in zip(self._internal_inputs, xx):
            input_.input_forward(x, train)
        yy = []
        for output in self._internal_outputs:
            yy += output.output_data()
        return yy
