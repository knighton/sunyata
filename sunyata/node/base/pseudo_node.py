

class PseudoNode(object):
    """
    A node of a computational graph.

    There are two types:
    * Input (placeholder for model input)
    * Node (everything else: receives inputs and broadcasts outputs)

    Tracks its outputs.  Also caches references to the inputs nodes of all its
    ancestors, which are needed by models.
    """

    def __init__(self, inputs):
        self._inputs_as_model = inputs
        self._forms = None
        self._data = None
        self._children = []

    def inputs_as_model(self):
        return self._inputs_as_model

    def forms(self):
        return self._forms

    def initialize_forms(self, forms):
        self.validate_forms(forms)
        assert self._forms is None
        self._forms = forms

    def data(self):
        return self._data

    def set_data(self, data):
        assert len(self._forms) == len(data)
        for form, tensor in zip(self._forms, data):
            assert Z.shape(tensor)[1:] == form.shape
            assert Z.dtype_of(tensor) == form.dtype
        self._data = data

    def children(self):
        return self._children

    def adopt_child(self, child):
        index = len(self._children)
        self._children.append(child)
        return index
