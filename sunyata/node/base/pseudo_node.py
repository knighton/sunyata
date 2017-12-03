from ... import backend as Z
from .spec import Form


class PseudoNode(object):
    """
    A node of a computational graph.

    There are two types:
    * Input (placeholder for model input)
    * Node (everything else: receives inputs and broadcasts outputs)

    Tracks its outputs.  Also caches references to the inputs nodes of all its
    ancestors, which are needed by models.
    """

    def __init__(self, model_inputs):
        self._model_inputs = model_inputs
        self._forms = None
        self._data = None
        self._children = []

    def model_inputs(self):
        return self._model_inputs

    @classmethod
    def collect_model_inputs(cls, pseudo_nodes):
        inputs_set = set()
        inputs = []
        for node in pseudo_nodes:
            for input_ in node.model_inputs():
                if input_ in inputs_set:
                    continue
                inputs_set.add(input_)
                inputs.append(input_)
        return inputs

    def forms(self):
        return self._forms

    @classmethod
    def validate_forms(cls, forms):
        assert forms
        assert isinstance(forms, list)
        for form in forms:
            assert isinstance(form, Form)

    def initialize_forms(self, forms):
        self.validate_forms(forms)
        assert self._forms is None
        self._forms = forms

    def data(self):
        return self._data

    def set_data(self, data):
        assert len(self._forms) == len(data)
        for form, x in zip(self._forms, data):
            assert Z.shape(x)[1:] == form.shape
            assert Z.dtype_of(x) == form.dtype
        self._data = data

    def children(self):
        return self._children

    def adopt_child(self, child):
        index = len(self._children)
        self._children.append(child)
        return index

    @classmethod
    def as_list(cls, x):
        if isinstance(x, list):
            nodes = x
        else:
            nodes = [x]
        return nodes

    @classmethod
    def as_pseudo_node_list(cls, x):
        nodes = cls.as_list(x)
        for node in nodes:
            assert isinstance(node, PseudoNode)
        return nodes
