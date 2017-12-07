from .. import backend as Z
from .base import Node
from .layer.base import Form


class Input(Node):
    """
    Placeholder for model input data within a computational graph.

    Input is one of the two kinds of Node (everything else is a Link).
    """

    def __init__(self, shape, dtype=None):
        model_inputs = [self]
        Node.__init__(self, model_inputs)
        dtype = Z.dtype(dtype)
        form = Form(shape, dtype)
        self.initialize_output_forms([form])

    def pseudo_node_to_pretty(self):
        form, = self._forms
        return 'Input(%s %s)' % (form.batch_shape, form.dtype)

    def input_build(self):
        for child in self.children():
            child.child_build()

    def input_params(self, nodes_seen, params_seen, params):
        for child in self.children():
            child.child_params(nodes_seen, params_seen, params)

    def input_forward(self, x, train):
        self.set_output_data([x])
        for child in self.children():
            child.child_forward(train)
