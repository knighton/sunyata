class LayerSpec(object):
    def __init__(self, spatial_ndim=None):
        if spatial_ndim is None:
            ndim = None
        else:
            assert spatial_ndim in {0, 1, 2, 3}
            ndim = spatial_ndim + 2
        self._x_ndim = ndim

    def build_inner(self, forms):
        raise NotImplementedError

    def build(self, forms):
        assert forms
        if self._x_ndim is None:
            x_ndim = forms[0].batch_ndim + 1
            start = 1
        else:
            x_ndim = self._x_ndim
            start = 0
        for form in forms[start:]:
            assert form.batch_ndim + 1 == x_ndim
        layer, forms = self.build_inner(self, forms)
        layer.initialize_input_ndim(x_ndim)
        return layer, forms


class TransformSpec(LayerSpec):
    def build_change(self, form):
        raise NotImplementedError

    def build_inner(self, forms):
        assert len(forms) == 1
        form, = forms
        layer, form = self.build_change(form)
        return layer, [form]


class MergeSpec(LayerSpec):
    def build_merge(self, forms):
        raise NotImplementedError

    def build_inner(self, forms):
        layer, form = self.build_merge(forms)
        return layer, [form]


class ForkSpec(LayerSpec):
    def build_fork(self, forms):
        raise NotImplementedError

    def build_inner(self, forms):
        assert len(forms) == 1
        form, = forms
        layer, forms = self.build_fork(form)
        return layer, forms


class FlexSpec(LayerSpec):
    def build_flex(self, forms):
        raise NotImplementedError

    def build_inner(self, forms):
        return self.build_flex(forms)
