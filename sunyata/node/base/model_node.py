from ...model.model import Model
from .node import Node


class ModelNode(Model, Node):
    """
    A node that also can be treated as a model to refer to its entire network.

    Examples are Sequences and Graphs.

    That is, this node is all the outputs of the network, and is descended from
    all its inputs.  Nodes track all their original inputs.  If treated as a
    model, it will construct and be the network from the beginning.  If called
    implicitly during model construction with node methods, it will only
    construct itself (without predecessors).
    """

    def __init__(self, parents):
        Model.__init__(self)
        Node.__init__(self, parents)

    def build_inner(self):
        for input_ in self.model_inputs():
            input_.input_build()

    def params(self):
        self.ensure_built()
        nodes_seen = set()
        params_seen = set()
        params = []
        for input_ in self.model_inputs():
            input_.input_params(nodes_seen, params_seen, params)
        return params

    def forward(self, xx, is_training):
        inputs = self.model_inputs()
        assert len(xx) == len(inputs)
        for input_, x in zip(inputs, xx):
            input_.input_forward(x, is_training)
        return self.output_data()
