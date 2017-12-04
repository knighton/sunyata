from ... import backend as Z
from .form import Form


class Node(object):
    """
    A node of a computational graph.

    There are two types:
    * Input (placeholder for model input)
    * Link (everything else: receives inputs and broadcasts outputs)

    Tracks its outputs.  Also caches references to the inputs nodes of all its
    ancestors, which are needed by models.
    """

    def __init__(self, model_inputs):
        self._model_inputs = model_inputs
        self._output_forms = None
        self._output_data = None
        self._children = []

    def model_inputs(self):
        return self._model_inputs

    @classmethod
    def collect_model_inputs(cls, nodes):
        inputs_set = set()
        inputs = []
        for node in nodes:
            for input_ in node.model_inputs():
                if input_ in inputs_set:
                    continue
                inputs_set.add(input_)
                inputs.append(input_)
        return inputs

    def forms(self):
        return self._output_forms

    @classmethod
    def validate_forms(cls, forms):
        assert forms
        assert isinstance(forms, list)
        for form in forms:
            assert isinstance(form, Form)

    def initialize_output_forms(self, forms):
        self.validate_forms(forms)
        assert self._output_forms is None
        self._output_forms = forms

    def output_data(self):
        return self._output_data

    def set_output_data(self, data):
        assert len(self._output_forms) == len(data)
        for form, x in zip(self._output_forms, data):
            assert Z.shape(x)[1:] == form.shape
            assert Z.dtype_of(x) == form.dtype
        self._output_data = data

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
    def as_nodes(cls, x):
        nodes = cls.as_list(x)
        for node in nodes:
            assert isinstance(node, Node)
        return nodes
