from copy import deepcopy

from ..layer.base import Form
from .pseudo_node import PseudoNode


class Node(PseudoNode):
    """
    A node of a computational graph.

    There are two types:
    * Input (placeholder for model input)
    * ChildNode (everything else: receives inputs and broadcasts outputs)

    Tracks its outputs.  Also caches references to the inputs nodes of all its
    ancestors, which are needed by models.
    """

    def __init__(self, model_inputs):
        self._model_inputs = model_inputs
        self._output_forms = None
        self._output_data = None
        self._children = []

    def __mul__(self, count):
        from .sequence import Sequence
        assert isinstance(count, int)
        assert 1 <= count
        steps = [deepcopy(self) for i in range(count)]
        return Sequence(*steps)

    def desugar(self):
        return self

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

    def output_forms(self):
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
            form.check(x)
        self._output_data = data

    def children(self):
        return self._children

    def adopt_child(self, child):
        index = len(self._children)
        self._children.append(child)
        return index

    @classmethod
    def as_list(cls, x):
        if x is None:
            nodes = []
        elif isinstance(x, list):
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
