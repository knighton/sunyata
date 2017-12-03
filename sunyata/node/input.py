from .. import backend as Z
from .base.pseudo_node import PseudoNode
from .base.spec import Form


class Input(PseudoNode):
    """
    Placeholder for model input data within a computational graph.

    Input is one of the two kinds of PseudoNode (everything else is a Node).
    """

    def __init__(self, shape, dtype=None):
        model_inputs = [self]
        PseudoNode.__init__(self, model_inputs)
        dtype = Z.dtype(dtype)
        form = Form(shape, dtype)
        self.initialize_forms([form])

    def input_build(self):
        for child in self.children():
            child.node_build()

    def input_params(self, nodes_seen, params_seen, params):
        for child in self.children():
            child.node_params(nodes_seen, params_seen, params)

    def input_forward(self, x, is_training):
        self.set_data([x])
        for child in self.children():
            child.node_forward(is_training)
